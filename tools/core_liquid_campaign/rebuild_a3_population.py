from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .production_cache import SourcePart
from .production_inputs import _load_daily_bars
from .production_population_tables import A3PopulationTableCompiler, PopulationTableError


UTC = timezone.utc
EXPECTED_SYMBOLS = 187
PROTECTED_START_MS = 1_767_225_600_000


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise PopulationTableError(f"{label} is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PopulationTableError(f"{label} is not an object")
    return payload


def build_a3_population_from_virtual_cache(
    *,
    virtual_cache_root: Path,
    virtual_cache_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Rebuild only the sparse A3 table from the already verified virtual cache."""
    manifest = _read_json(virtual_cache_manifest_path, label="virtual cache manifest")
    if manifest.get("schema") != "stage23_production_semantic_cache_manifest_v1":
        raise PopulationTableError("virtual cache manifest schema differs")
    artifacts = list(manifest.get("artifacts", ()))
    if manifest.get("artifact_inventory_sha256") != canonical_hash(artifacts):
        raise PopulationTableError("virtual cache artifact inventory differs")
    pit_records = [row for row in artifacts if row.get("path") == "pit/PIT_DAILY_MEMBERSHIP.jsonl"]
    symbol_records = [row for row in artifacts if row.get("symbol")]
    if len(pit_records) != 1 or len(symbol_records) != EXPECTED_SYMBOLS:
        raise PopulationTableError("virtual cache does not contain PIT plus 187 symbol indexes")

    pit_path = virtual_cache_root / str(pit_records[0]["path"])
    if pit_path.stat().st_size != int(pit_records[0]["bytes"]) or sha256_file(pit_path) != pit_records[0]["sha256"]:
        raise PopulationTableError("PIT membership bytes differ from virtual cache authority")
    pit_rows = [json.loads(line) for line in pit_path.read_text(encoding="utf-8").splitlines() if line]
    if any(int(row["day_open_ms"]) >= PROTECTED_START_MS for row in pit_rows):
        raise PopulationTableError("PIT membership contains a protected day")

    parts_by_symbol: dict[str, tuple[SourcePart, ...]] = {}
    required_source_parts: dict[str, Mapping[str, Any]] = {}
    for record in sorted(symbol_records, key=lambda row: str(row["symbol"])):
        index_path = virtual_cache_root / str(record["path"])
        if index_path.stat().st_size != int(record["bytes"]) or sha256_file(index_path) != record["sha256"]:
            raise PopulationTableError("virtual symbol index bytes differ")
        index = _read_json(index_path, label="virtual symbol index")
        source_parts = list(index.get("source_parts", ()))
        if (
            index.get("schema") != "stage23_production_symbol_cache_index_v1"
            or index.get("symbol") != record["symbol"]
            or index.get("source_part_inventory_sha256") != canonical_hash(source_parts)
        ):
            raise PopulationTableError("virtual symbol index contract differs")
        trade_parts = []
        for raw in source_parts:
            if raw.get("dataset") != "historical_trade_candles_5m":
                continue
            part = SourcePart(**{key: raw[key] for key in SourcePart.__dataclass_fields__})
            if part.symbol != record["symbol"]:
                raise PopulationTableError("trade source symbol differs from its virtual index")
            prior = required_source_parts.get(part.path)
            if prior is not None and prior != raw:
                raise PopulationTableError("trade source path has conflicting identities")
            required_source_parts[part.path] = raw
            trade_parts.append(part)
        if not trade_parts:
            raise PopulationTableError("registered symbol has no trade source parts")
        parts_by_symbol[str(record["symbol"])] = tuple(trade_parts)

    for raw_path, record in sorted(required_source_parts.items()):
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size <= 0 or sha256_file(path) != record["sha256"]:
            raise PopulationTableError("required trade source bytes differ from authority")

    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    daily_by_symbol = {
        symbol: _load_daily_bars(parts, start, end)
        for symbol, parts in sorted(parts_by_symbol.items())
    }
    build = A3PopulationTableCompiler(output_root, parts_by_symbol, pit_rows, daily_by_symbol).build()
    report = {
        "schema": "stage24_a3_population_targeted_rebuild_v1",
        "status": "pass",
        "virtual_cache_manifest": {
            "path": str(virtual_cache_manifest_path.resolve()),
            "bytes": virtual_cache_manifest_path.stat().st_size,
            "sha256": sha256_file(virtual_cache_manifest_path),
        },
        "pit_membership_sha256": sha256_file(pit_path),
        "symbols": len(parts_by_symbol),
        "required_trade_source_parts": len(required_source_parts),
        "required_trade_source_inventory_sha256": canonical_hash(list(required_source_parts.values())),
        "source_bytes_physically_verified": True,
        "registered_breakout_lookback_days": [5, 10, 20, 60],
        "population_manifest": {
            "path": str(Path(build["manifest_path"]).resolve()),
            "bytes": Path(build["manifest_path"]).stat().st_size,
            "sha256": build["manifest_sha256"],
        },
        "features": len(build["features"]),
        "rows": sum(int(record["rows"]) for record in build["features"].values()),
        "protected_rows": int(build["protected_rows"]),
        "economic_outcomes_opened": False,
    }
    report["rebuild_identity_sha256"] = canonical_hash(report)
    atomic_write_json(output_root / "A3_TARGETED_REBUILD.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Rebuild only the frozen Stage24 A3 sparse population authority")
    result.add_argument("--virtual-cache-root", type=Path, required=True)
    result.add_argument("--virtual-cache-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    report = build_a3_population_from_virtual_cache(
        virtual_cache_root=args.virtual_cache_root,
        virtual_cache_manifest_path=args.virtual_cache_manifest,
        output_root=args.output,
    )
    print(json.dumps({"status": report["status"], "population_manifest": report["population_manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_a3_population_from_virtual_cache", "main"]
