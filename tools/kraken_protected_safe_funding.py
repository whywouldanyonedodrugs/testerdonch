#!/usr/bin/env python3
"""Protected-safe Kraken funding metadata guard and synthetic arithmetic.

This module never performs a broad Parquet table read. Source payload is
deserialized only by explicit, previously admitted row-group index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RANKABLE_START = pd.Timestamp("2023-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
REQUIRED_COLUMNS = ("timestamp", "fundingRate")
SAFE = "safe_rankable_row_group"
PROTECTED = "protected_row_group"
MIXED = "mixed_row_group"
UNKNOWN = "unknown_row_group"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _statistics_bounds(statistics: Any) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if statistics is None or not bool(getattr(statistics, "has_min_max", False)):
        return None, None
    try:
        minimum = _utc(statistics.min)
        maximum = _utc(statistics.max)
    except Exception:
        return None, None
    if pd.isna(minimum) or pd.isna(maximum) or maximum < minimum:
        return None, None
    return minimum, maximum


def classify_bounds(minimum: pd.Timestamp | None, maximum: pd.Timestamp | None) -> str:
    if minimum is None or maximum is None:
        return UNKNOWN
    if maximum < PROTECTED_START:
        return SAFE
    if minimum >= PROTECTED_START:
        return PROTECTED
    return MIXED


@dataclass(frozen=True)
class SourceAuthority:
    platform: str
    purpose: str
    source_manifest_path: str
    source_manifest_sha256: str
    file_path: str
    file_sha256: str


def inspect_source_file(authority: SourceAuthority) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if authority.platform != "kraken_derivatives":
        raise RuntimeError("funding source platform is not Kraken derivatives")
    if authority.purpose != "rankable_exact_funding":
        raise RuntimeError("funding source purpose is not rankable exact funding")
    path = Path(authority.file_path)
    if sha256_file(path) != authority.file_sha256:
        raise RuntimeError(f"funding source hash mismatch: {path}")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(f"invalid Parquet footer: {path}") from exc
    names = parquet.schema_arrow.names
    missing = sorted(set(REQUIRED_COLUMNS) - set(names))
    if missing:
        raise RuntimeError(f"funding source schema missing {missing}: {path}")
    timestamp_index = names.index("timestamp")
    schema_payload = [(field.name, str(field.type), field.nullable) for field in parquet.schema_arrow]
    footer_payload = {
        "created_by": parquet.metadata.created_by,
        "file_size": path.stat().st_size,
        "num_rows": parquet.metadata.num_rows,
        "num_row_groups": parquet.metadata.num_row_groups,
        "schema_sha256": canonical_sha256(schema_payload),
    }
    rows: list[dict[str, Any]] = []
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        minimum, maximum = _statistics_bounds(group.column(timestamp_index).statistics)
        rows.append({
            "file_path": str(path),
            "file_sha256": authority.file_sha256,
            "row_group_index": index,
            "row_count": group.num_rows,
            "min_timestamp": minimum.isoformat() if minimum is not None else "",
            "max_timestamp": maximum.isoformat() if maximum is not None else "",
            "classification": classify_bounds(minimum, maximum),
            "payload_requested": False,
        })
    file_record = {
        **authority.__dict__,
        **footer_payload,
        "footer_identity_sha256": canonical_sha256(footer_payload),
        "timestamp_column": "timestamp",
        "absolute_rate_column": "fundingRate",
        "protected_start": PROTECTED_START.isoformat(),
    }
    return file_record, rows


class GuardedRowGroupReader:
    """Read only row groups already classified SAFE by trusted footer stats."""

    def __init__(self, audit_rows: list[dict[str, Any]], read_method: Callable[..., pa.Table] | None = None):
        self.audit_rows = audit_rows
        self.requests: list[dict[str, Any]] = []
        self._read_method = read_method

    def read(self, path: Path, row_group_index: int, columns: Iterable[str] = REQUIRED_COLUMNS) -> pd.DataFrame:
        matches = [
            row for row in self.audit_rows
            if row["file_path"] == str(path) and int(row["row_group_index"]) == int(row_group_index)
        ]
        if len(matches) != 1 or matches[0]["classification"] != SAFE:
            raise RuntimeError("non-safe funding row group payload request rejected before deserialization")
        audit = matches[0]
        requested_columns = list(dict.fromkeys([*columns, "timestamp"]))
        self.requests.append({
            "file_path": str(path), "row_group_index": int(row_group_index),
            "classification": SAFE, "columns": requested_columns, "status": "requested",
        })
        parquet = pq.ParquetFile(path)
        method = self._read_method or parquet.read_row_group
        table = method(row_group_index, columns=requested_columns)
        frame = table.to_pandas()
        stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if stamps.isna().any() or (stamps < RANKABLE_START).any() or (stamps >= PROTECTED_START).any():
            self.requests[-1]["status"] = "payload_footer_contradiction_global_stop"
            raise RuntimeError("funding payload contradicts admitted rankable footer bounds")
        footer_min = _utc(audit["min_timestamp"])
        footer_max = _utc(audit["max_timestamp"])
        if (stamps < footer_min).any() or (stamps > footer_max).any():
            self.requests[-1]["status"] = "payload_footer_contradiction_global_stop"
            raise RuntimeError("funding payload timestamps contradict footer statistics")
        audit["payload_requested"] = True
        self.requests[-1]["status"] = "read_and_payload_asserted"
        frame["timestamp"] = stamps
        return frame


def overlap_fraction(entry_ts: Any, exit_ts: Any, period_start: Any) -> float:
    entry, exit_, start = _utc(entry_ts), _utc(exit_ts), _utc(period_start)
    end = start + pd.Timedelta(hours=1)
    seconds = max(0.0, (min(exit_, end) - max(entry, start)).total_seconds())
    return min(1.0, seconds / 3600.0)


def exact_funding_cashflow_bps(
    position_sign: int, absolute_funding_rate: float, held_fraction: float, entry_trade_open: float
) -> float:
    if position_sign not in {-1, 1}:
        raise ValueError("position_sign must be -1 or +1")
    values = [absolute_funding_rate, held_fraction, entry_trade_open]
    if not all(math.isfinite(float(value)) for value in values) or entry_trade_open <= 0 or not 0 <= held_fraction <= 1:
        raise ValueError("invalid exact funding arithmetic input")
    return -position_sign * absolute_funding_rate * held_fraction / entry_trade_open * 10000.0


def adverse_allowance_cost_bps(allowance_bps_per_hour: float, held_fraction: float) -> float:
    if not math.isfinite(allowance_bps_per_hour) or allowance_bps_per_hour < 0 or not 0 <= held_fraction <= 1:
        raise ValueError("invalid adverse allowance input")
    return -allowance_bps_per_hour * held_fraction


def calibrate_allowances(calibration: pd.DataFrame, minimum_rows: int = 720) -> pd.DataFrame:
    required = {"symbol", "absolute_hourly_funding_bps_on_mark_notional"}
    if not required.issubset(calibration.columns):
        raise ValueError("calibration fields missing")
    work = calibration[list(required)].copy()
    work["absolute_hourly_funding_bps_on_mark_notional"] = pd.to_numeric(
        work["absolute_hourly_funding_bps_on_mark_notional"], errors="coerce"
    )
    values = work["absolute_hourly_funding_bps_on_mark_notional"]
    work = work[np.isfinite(values) & (values >= 0)]
    eligible_rows: list[dict[str, Any]] = []
    all_symbols = sorted(set(map(str, calibration["symbol"].dropna())))
    for symbol, group in work.groupby("symbol", sort=True):
        if len(group) >= minimum_rows:
            arr = group["absolute_hourly_funding_bps_on_mark_notional"].to_numpy(float)
            eligible_rows.append({
                "symbol": str(symbol), "exact_observations": len(arr),
                "base_adverse_allowance_bps_per_hour": float(np.quantile(arr, 0.95, method="linear")),
                "stress_adverse_allowance_bps_per_hour": float(np.quantile(arr, 0.99, method="linear")),
                "allowance_source": "symbol_type7",
            })
    if not eligible_rows:
        raise RuntimeError("no eligible symbols for adverse funding allowance calibration")
    base_fallback = float(np.mean([row["base_adverse_allowance_bps_per_hour"] for row in eligible_rows]))
    stress_fallback = float(np.mean([row["stress_adverse_allowance_bps_per_hour"] for row in eligible_rows]))
    by_symbol = {row["symbol"]: row for row in eligible_rows}
    output = []
    for symbol in all_symbols:
        row = by_symbol.get(symbol)
        if row is None:
            output.append({
                "symbol": symbol,
                "exact_observations": int((work["symbol"].astype(str) == symbol).sum()),
                "base_adverse_allowance_bps_per_hour": base_fallback,
                "stress_adverse_allowance_bps_per_hour": stress_fallback,
                "allowance_source": "equal_symbol_weighted_eligible_symbol_quantile_fallback",
            })
        else:
            output.append(row)
    result = pd.DataFrame(output).sort_values("symbol", kind="mergesort").reset_index(drop=True)
    if (result["stress_adverse_allowance_bps_per_hour"] < result["base_adverse_allowance_bps_per_hour"]).any():
        raise RuntimeError("q99 allowance is below q95")
    return result


def selection_funding_metrics(gross_return_bps: float, held_fractions: Iterable[float], base: float, stress: float) -> dict[str, float]:
    fractions = list(held_fractions)
    base_cost = sum(adverse_allowance_cost_bps(base, value) for value in fractions)
    stress_cost = sum(adverse_allowance_cost_bps(stress, value) for value in fractions)
    return {
        "pre_funding_diagnostic_bps": gross_return_bps - 14.0,
        "uniform_adverse_allowance_primary_bps": gross_return_bps - 14.0 + base_cost,
        "uniform_adverse_allowance_stress_bps": gross_return_bps - 32.0 + stress_cost,
    }


def validate_campaign_funding_source(package_root: Path) -> dict[str, Any]:
    manifest_path = package_root / "RANKABLE_EXACT_FUNDING_SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_physical_rankable_exact_funding_source":
        raise RuntimeError("campaign funding source is not a completed physical rankable package")
    if manifest.get("protected_payload_rows_opened") != 0:
        raise RuntimeError("campaign funding source records protected payload access")
    relative = manifest.get("physical_rankable_parquet")
    expected = manifest.get("physical_rankable_parquet_sha256")
    if not relative or not expected:
        raise RuntimeError("campaign funding source identity is incomplete")
    path = package_root / relative
    if sha256_file(path) != expected:
        raise RuntimeError("campaign funding source physical hash mismatch")
    parquet = pq.ParquetFile(path)
    if "timestamp" not in parquet.schema_arrow.names:
        raise RuntimeError("campaign funding source timestamp missing")
    timestamp_index = parquet.schema_arrow.names.index("timestamp")
    for index in range(parquet.metadata.num_row_groups):
        minimum, maximum = _statistics_bounds(parquet.metadata.row_group(index).column(timestamp_index).statistics)
        if classify_bounds(minimum, maximum) != SAFE:
            raise RuntimeError("campaign funding source is not physically rankable-only")
    return manifest


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rankable_package(source_manifest: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root exists and is nonempty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(source_manifest)
    source = pd.read_csv(source_manifest, low_memory=False)
    required_manifest = {"dataset", "symbol", "parquet_path", "parquet_sha256", "status"}
    if not required_manifest.issubset(source.columns):
        raise RuntimeError("source acquisition manifest schema incomplete")
    source = source[(source["dataset"] == "funding") & (source["status"] == "downloaded")].copy()
    file_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for row in source.sort_values(["symbol", "parquet_path"], kind="mergesort").itertuples(index=False):
        authority = SourceAuthority(
            platform="kraken_derivatives", purpose="rankable_exact_funding",
            source_manifest_path=str(source_manifest), source_manifest_sha256=manifest_sha,
            file_path=str(row.parquet_path), file_sha256=str(row.parquet_sha256),
        )
        file_record, groups = inspect_source_file(authority)
        file_record["symbol"] = str(row.symbol)
        file_rows.append(file_record)
        for group in groups:
            group["symbol"] = str(row.symbol)
            group_rows.append(group)
    safe = [row for row in group_rows if row["classification"] == SAFE]
    skipped = [row for row in group_rows if row["classification"] != SAFE]
    source_fields = [
        "symbol", "file_path", "file_sha256", "row_group_index", "row_count",
        "min_timestamp", "max_timestamp", "classification", "payload_requested",
    ]
    _write_csv(output_root / "SOURCE_FILE_AND_ROW_GROUP_LEDGER.csv", group_rows, source_fields)
    _write_csv(output_root / "SKIPPED_ROW_GROUP_LEDGER.csv", skipped, source_fields)
    _write_json(output_root / "SCHEMA.json", {
        "required_source_columns": list(REQUIRED_COLUMNS),
        "output_columns": ["symbol", "timestamp", "fundingRate", "source_file_sha256", "source_row_group_index"],
        "timestamp_timezone": "UTC", "rankable_interval": [RANKABLE_START.isoformat(), PROTECTED_START.isoformat()],
    })
    counts = pd.Series([row["classification"] for row in group_rows]).value_counts().to_dict()
    _write_csv(output_root / "COVERAGE.csv", [{
        "source_files": len(file_rows), "source_row_groups": len(group_rows),
        "safe_rankable_row_groups": len(safe), "protected_row_groups": counts.get(PROTECTED, 0),
        "mixed_row_groups": counts.get(MIXED, 0), "unknown_row_groups": counts.get(UNKNOWN, 0),
        "payload_row_groups_read": 0,
    }], ["source_files", "source_row_groups", "safe_rankable_row_groups", "protected_row_groups", "mixed_row_groups", "unknown_row_groups", "payload_row_groups_read"])
    status = "ready_for_safe_payload_materialization" if safe else "blocked_no_safe_rankable_absolute_funding_row_groups"
    audit = {
        "status": status, "protected_start": PROTECTED_START.isoformat(),
        "source_manifest_path": str(source_manifest), "source_manifest_sha256": manifest_sha,
        "source_files_verified": len(file_rows), "row_group_classification_counts": counts,
        "payload_row_groups_read": 0, "protected_payload_rows_opened": 0,
        "mixed_payload_rows_opened": 0, "unknown_payload_rows_opened": 0,
        "broad_table_read_used": False,
    }
    _write_json(output_root / "PROTECTED_AUDIT.json", audit)
    source_manifest_value = {
        **audit, "platform": "kraken_derivatives", "purpose": "rankable_exact_funding",
        "file_authorities_sha256": canonical_sha256(file_rows),
        "safe_payload_materialized": False,
        "physical_rankable_parquet": None,
    }
    _write_json(output_root / "RANKABLE_EXACT_FUNDING_SOURCE_MANIFEST.json", source_manifest_value)
    _write_json(output_root / "PROTECTED_SAFE_FUNDING_READER_CONTRACT.json", {
        "version": "stage18_v1", "admission": {
            "safe": "max_timestamp < protected_start",
            "protected": "min_timestamp >= protected_start",
            "mixed": "min_timestamp < protected_start <= max_timestamp",
            "unknown": "missing_invalid_or_untrusted_statistics",
        },
        "payload_rule": "only safe row groups may be requested; post-read timestamps must remain within footer and rankable bounds",
        "campaign_source_rule": "only a completed physical rankable package with a passing manifest may be consumed",
    })
    (output_root / "README.md").write_text(
        "# Protected-Safe Exact Funding Source\n\n"
        f"Status: `{status}`. Footer metadata was inspected and source hashes were verified. "
        "No funding payload row group was deserialized. A physical exact-funding Parquet was not created because no row group passed the protected-safe admission rule.\n",
        encoding="utf-8",
    )
    artifacts = []
    for path in sorted(output_root.iterdir()):
        if path.name == "ARTIFACT_MANIFEST.json" or not path.is_file():
            continue
        artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    _write_json(output_root / "ARTIFACT_MANIFEST.json", {
        "status": status, "economic_outputs_computed": False,
        "protected_payload_rows_opened": 0, "files": artifacts,
    })
    return source_manifest_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = build_rankable_package(Path(args.source_manifest), Path(args.output_root))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready_for_safe_payload_materialization" else 3


if __name__ == "__main__":
    raise SystemExit(main())
