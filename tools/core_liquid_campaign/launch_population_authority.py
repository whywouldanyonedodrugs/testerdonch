from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file


UTC = timezone.utc
DAY_MS = 86_400_000
PROTECTED_START_MS = 1_767_225_600_000
TOP_N = (10, 20, 40)
EXPECTED_SYMBOLS = 187
EXPECTED_PARTITIONS = 132
EXPECTED_INNER_PARTITIONS = 124
EXPECTED_RAW_SOURCE_PARTS = 81_906
EXPECTED_SOURCE_BYTES = 3_528_014_133
EXPECTED_SOURCE_ROWS = 81_128_690
EXPECTED_FUNDING_ROWS = 3_373_194
EXPECTED_PHYSICAL_COUNTS = {
    "all": {"symbol_days": 134_151, "decisions_5m": 38_582_680},
    "10": {"symbol_days": 10_660, "decisions_5m": 3_067_210},
    "20": {"symbol_days": 21_320, "decisions_5m": 6_134_420},
    "40": {"symbol_days": 42_640, "decisions_5m": 12_268_840},
}
EXPECTED_EXPANDED_COUNTS = {
    "all": {"symbol_days": 518_360, "decisions_5m": 149_234_872, "a4_8h": 1_554_712, "a4_1d": 518_360},
    "10": {"symbol_days": 45_140, "decisions_5m": 12_997_450, "a4_8h": 135_400, "a4_1d": 45_140},
    "20": {"symbol_days": 90_280, "decisions_5m": 25_994_900, "a4_8h": 270_800, "a4_1d": 90_280},
    "40": {"symbol_days": 180_560, "decisions_5m": 51_989_800, "a4_8h": 541_600, "a4_1d": 180_560},
}
EXPECTED_A3_COUNTS = {
    "raw_signature_rows": {"all": 179_689, "10": 13_759, "20": 27_309, "40": 54_732},
    "fold_expanded_signature_rows": {"all": 706_464, "10": 61_579, "20": 123_641, "40": 245_886},
    "unique_crossing_keys_without_atr": {"all": 45_266, "10": 3_456, "20": 6_860, "40": 13_780},
    "fold_expanded_unique_crossing_keys_without_atr": {"all": 177_763, "10": 15_401, "20": 30_944, "40": 61_654},
}


class LaunchPopulationAuthorityError(RuntimeError):
    pass


def _utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    return int(parsed.timestamp() * 1000)


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _verify_record(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise LaunchPopulationAuthorityError(f"{label} bytes differ from authority")


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LaunchPopulationAuthorityError(f"required authority file is absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _partitions(fold_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outer_folds = fold_graph.get("outer_folds", ())
    if len(outer_folds) != 8 or sum(len(row.get("inner_folds", ())) for row in outer_folds) != EXPECTED_INNER_PARTITIONS:
        raise LaunchPopulationAuthorityError("fold graph is not the frozen eight-outer/124-inner graph")
    for outer in outer_folds:
        rows.append({
            "phase": "outer_evaluation", "outer_fold_id": str(outer["outer_fold_id"]), "inner_fold_id": None,
            "evaluation_start_ms": _utc_ms(outer["outer_evaluation_start"]),
            "evaluation_end_exclusive_ms": _utc_ms(outer["outer_evaluation_end_exclusive"]),
        })
        for inner in outer["inner_folds"]:
            rows.append({
                "phase": "inner_validation", "outer_fold_id": str(outer["outer_fold_id"]),
                "inner_fold_id": str(inner["inner_fold_id"]),
                "evaluation_start_ms": _utc_ms(inner["validation_start"]),
                "evaluation_end_exclusive_ms": _utc_ms(inner["validation_end_exclusive"]),
            })
    if len(rows) != EXPECTED_PARTITIONS or any(row["evaluation_start_ms"] >= row["evaluation_end_exclusive_ms"] for row in rows):
        raise LaunchPopulationAuthorityError("fold graph partition inventory is invalid")
    return rows


def census_pit(path: Path, partitions: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    membership: dict[tuple[str, int], dict[str, Any]] = {}
    symbols: set[str] = set()
    physical = {key: Counter() for key in ("all", "10", "20", "40")}
    partition_rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            symbol = str(row["symbol"]); day = int(row["day_open_ms"]); decisions = int(row["decision_count_5m"])
            if (
                day < 1_672_531_200_000
                or day >= PROTECTED_START_MS
                or day % DAY_MS
                or decisions not in {1, 288}
                or (decisions == 1 and day != PROTECTED_START_MS - DAY_MS)
            ):
                raise LaunchPopulationAuthorityError("PIT row has an unsupported boundary or partial-day schedule")
            key = (symbol, day)
            if key in membership:
                raise LaunchPopulationAuthorityError("PIT membership contains a duplicate symbol-day")
            normalized = dict(row); membership[key] = normalized; symbols.add(symbol)
            for top in ("all", "10", "20", "40"):
                if top == "all" or bool(row[f"top_{top}"]):
                    physical[top].update(symbol_days=1, decisions_5m=decisions, a4_8h=3 if decisions == 288 else 1, a4_1d=1)
    for partition in partitions:
        counts = {key: Counter() for key in physical}
        start = int(partition["evaluation_start_ms"]); end = int(partition["evaluation_end_exclusive_ms"])
        selected_symbols = {key: set() for key in physical}
        for (symbol, day), row in membership.items():
            if not start <= day < end:
                continue
            decisions = int(row["decision_count_5m"])
            for top in counts:
                if top == "all" or bool(row[f"top_{top}"]):
                    counts[top].update(symbol_days=1, decisions_5m=decisions, a4_8h=3 if decisions == 288 else 1, a4_1d=1)
                    selected_symbols[top].add(symbol)
        partition_rows.append({
            **dict(partition),
            "counts": {key: {**dict(counts[key]), "symbols": len(selected_symbols[key])} for key in counts},
        })
    expanded = {
        key: {field: sum(int(row["counts"][key][field]) for row in partition_rows) for field in ("symbol_days", "decisions_5m", "a4_8h", "a4_1d")}
        for key in physical
    }
    report = {
        "symbols": len(symbols), "membership_rows": len(membership),
        "physical": {key: dict(value) for key, value in physical.items()},
        "fold_expanded": expanded, "partitions": partition_rows,
        "membership_content_sha256": canonical_hash(list(membership.values())),
    }
    return report, membership


def census_a3(manifest: Mapping[str, Any], cache_root: Path, membership: Mapping[tuple[str, int], Mapping[str, Any]], partitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    codes = {int(value): str(key) for key, value in manifest["symbol_codes"].items()}
    raw = Counter(); expanded = Counter(); by_phase = Counter()
    unique: dict[str, set[tuple[str, int, int, int]]] = {key: set() for key in ("all", "10", "20", "40")}
    expected_names = {
        f"A3_breakout:lookback={lookback}:atr={atr}:side={side}"
        for lookback in (20, 60, 120, 250) for atr in (10, 20, 40, 60) for side in (-1, 1)
    }
    if set(manifest.get("features", {})) != expected_names:
        raise LaunchPopulationAuthorityError("A3 feature signature inventory differs from the frozen 32 signatures")
    for name, record in manifest["features"].items():
        fields = dict(item.split("=", 1) for item in name.split(":")[1:])
        lookback = int(fields["lookback"]); side = int(fields["side"])
        arrays = {}
        for field, dtype in (("timestamps", "<i8"), ("symbols", "<u2"), ("deciles", "u1"), ("values", "<f8")):
            path = cache_root / str(record[f"{field}_path"])
            _verify_record(path, {"bytes": record[f"{field}_bytes"], "sha256": record[f"{field}_sha256"]}, label=f"A3 {field}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1 or len(array) != int(record["rows"]) or array.dtype != np.dtype(dtype):
                raise LaunchPopulationAuthorityError("A3 array shape or dtype differs")
            if field == "values" and not np.isfinite(array).all():
                raise LaunchPopulationAuthorityError("A3 values contain a non-finite observation")
            arrays[field] = array
        for timestamp, code, decile in zip(arrays["timestamps"], arrays["symbols"], arrays["deciles"]):
            decision = int(timestamp)
            if decision >= PROTECTED_START_MS:
                raise LaunchPopulationAuthorityError("A3 authority contains a protected decision")
            symbol = codes.get(int(code)); source_day = (decision - 1) // DAY_MS * DAY_MS
            pit = membership.get((str(symbol), source_day))
            if pit is None:
                raise LaunchPopulationAuthorityError("A3 crossing is absent from PIT membership")
            expected_decile = 1 + min(9, int((float(pit["average_liquidity_rank"]) - 1) * 10 / int(pit["eligible_population"])))
            if int(decile) != expected_decile:
                raise LaunchPopulationAuthorityError("A3 crossing liquidity decile differs from PIT authority")
            containing = [row for row in partitions if int(row["evaluation_start_ms"]) <= decision < int(row["evaluation_end_exclusive_ms"])]
            for top in ("all", "10", "20", "40"):
                if top != "all" and not bool(pit[f"top_{top}"]):
                    continue
                raw[top] += 1; expanded[top] += len(containing)
                unique[top].add((str(symbol), decision, lookback, side))
                for partition in containing:
                    by_phase[(top, str(partition["phase"]))] += 1
    unique_expanded = {
        top: sum(sum(int(row["evaluation_start_ms"]) <= key[1] < int(row["evaluation_end_exclusive_ms"]) for row in partitions) for key in keys)
        for top, keys in unique.items()
    }
    return {
        "raw_signature_rows": dict(raw), "fold_expanded_signature_rows": dict(expanded),
        "unique_crossing_keys_without_atr": {key: len(value) for key, value in unique.items()},
        "fold_expanded_unique_crossing_keys_without_atr": unique_expanded,
        "fold_expanded_by_phase": {
            top: {phase: by_phase[(top, phase)] for phase in ("inner_validation", "outer_evaluation")}
            for top in ("all", "10", "20", "40")
        },
    }


def validate_frozen_counts(payload: Mapping[str, Any]) -> None:
    population = payload["population_census"]
    physical = {key: {field: int(population["physical"][key][field]) for field in ("symbol_days", "decisions_5m")} for key in EXPECTED_PHYSICAL_COUNTS}
    if physical != EXPECTED_PHYSICAL_COUNTS:
        raise LaunchPopulationAuthorityError("physical PIT population differs from frozen authority")
    expanded = {key: {field: int(population["fold_expanded"][key][field]) for field in EXPECTED_EXPANDED_COUNTS[key]} for key in EXPECTED_EXPANDED_COUNTS}
    if expanded != EXPECTED_EXPANDED_COUNTS:
        raise LaunchPopulationAuthorityError("fold-expanded population differs from frozen authority")
    a3 = payload["a3_sparse_census"]
    for field, expected in EXPECTED_A3_COUNTS.items():
        if {key: int(a3[field][key]) for key in expected} != expected:
            raise LaunchPopulationAuthorityError("A3 sparse population differs from frozen authority")


def validate_launch_population_authority(payload: Mapping[str, Any], *, verify_files: bool = True) -> None:
    if payload.get("schema") != "stage24_launch_population_authority_v1" or payload.get("status") != "bound_outcome_free":
        raise LaunchPopulationAuthorityError("launch population authority schema or status differs")
    firewall = payload.get("outcome_firewall", {})
    if firewall != {"economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False}:
        raise LaunchPopulationAuthorityError("launch population authority outcome firewall is not closed")
    benchmark = payload.get("benchmark_semantic_frames", {})
    if (
        benchmark.get("classification") != "benchmark_probe_only"
        or benchmark.get("launch_input_authority") is not False
        or int(benchmark.get("artifacts", -1)) != 567
    ):
        raise LaunchPopulationAuthorityError("benchmark semantic frames are not isolated from launch authority")
    source = payload.get("source_authority", {})
    if (
        int(source.get("physical_parts", -1)) != EXPECTED_RAW_SOURCE_PARTS
        or int(source.get("source_bytes", -1)) != EXPECTED_SOURCE_BYTES
        or int(source.get("source_rows", -1)) != EXPECTED_SOURCE_ROWS
        or source.get("raw_source_parts_physically_verified") is not True
    ):
        raise LaunchPopulationAuthorityError("raw source population authority differs")
    funding = payload.get("funding_authority", {})
    if int(funding.get("symbols", -1)) != EXPECTED_SYMBOLS or int(funding.get("rows", -1)) != EXPECTED_FUNDING_ROWS:
        raise LaunchPopulationAuthorityError("funding population authority differs")
    validate_frozen_counts(payload)
    expected_inventory = canonical_hash({key: value for key, value in payload.items() if key != "authority_inventory_sha256"})
    if payload.get("authority_inventory_sha256") != expected_inventory:
        raise LaunchPopulationAuthorityError("launch population authority inventory hash differs")
    if verify_files:
        for key in (
            "execution_input_authority", "fold_graph", "pit_membership", "a1_population_table",
            "a3_population_table", "benchmark_semantic_frames",
        ):
            record = payload[key]
            _verify_record(Path(str(record["path"])), record, label=key)
        virtual = payload["virtual_cache"]
        _verify_record(Path(str(virtual["path"])), virtual, label="virtual_cache")


def build_launch_population_authority(
    *, repository_root: Path, execution_authority_path: Path, virtual_cache_root: Path,
    virtual_cache_manifest_path: Path, fold_graph_path: Path, a1_manifest_path: Path,
    a3_manifest_path: Path, benchmark_cache_manifest_path: Path,
) -> dict[str, Any]:
    execution = json.loads(execution_authority_path.read_text(encoding="utf-8")); execution_sha = sha256_file(execution_authority_path)
    for record in execution.get("source_records", ()):
        _verify_record(_resolve(repository_root, str(record["path"])), record, label=f"execution source {record.get('role')}")
    fold_graph = json.loads(fold_graph_path.read_text(encoding="utf-8"))
    if sha256_file(fold_graph_path) != execution.get("fold_graph_sha256"):
        raise LaunchPopulationAuthorityError("fold graph differs from execution authority")
    partitions = _partitions(fold_graph)
    virtual = json.loads(virtual_cache_manifest_path.read_text(encoding="utf-8"))
    source_verification = virtual.get("source_verification", {})
    funding = virtual.get("funding", {})
    if virtual.get("schema") != "stage23_production_semantic_cache_manifest_v1" or virtual.get("authority_sha256") != execution_sha or virtual.get("campaign_symbols") != EXPECTED_SYMBOLS:
        raise LaunchPopulationAuthorityError("Stage23 virtual cache authority binding differs")
    if (
        int(source_verification.get("physical_parts", -1)) != EXPECTED_RAW_SOURCE_PARTS
        or int(source_verification.get("source_bytes", -1)) != EXPECTED_SOURCE_BYTES
        or int(source_verification.get("source_rows", -1)) != EXPECTED_SOURCE_ROWS
        or int(funding.get("symbols", -1)) != EXPECTED_SYMBOLS
        or int(funding.get("rows", -1)) != EXPECTED_FUNDING_ROWS
    ):
        raise LaunchPopulationAuthorityError("Stage23 source or funding inventory differs")
    source_roles = {str(row.get("role")) for row in execution.get("source_records", ())}
    if not {"price_and_instrument_source_manifest", "funding_partition_manifest", "rankable_funding_package"} <= source_roles:
        raise LaunchPopulationAuthorityError("execution authority omits a required price/funding source binding")
    artifacts = list(virtual.get("artifacts", ()))
    if len(artifacts) != EXPECTED_SYMBOLS + 1:
        raise LaunchPopulationAuthorityError("virtual cache must contain PIT plus 187 symbol indexes")
    pit_records = [row for row in artifacts if row.get("path") == "pit/PIT_DAILY_MEMBERSHIP.jsonl"]
    symbol_records = [row for row in artifacts if row.get("symbol")]
    if len(pit_records) != 1 or len(symbol_records) != EXPECTED_SYMBOLS or len({row["symbol"] for row in symbol_records}) != EXPECTED_SYMBOLS:
        raise LaunchPopulationAuthorityError("virtual cache PIT/symbol inventory is incomplete")
    for record in artifacts:
        _verify_record(virtual_cache_root / str(record["path"]), record, label="virtual cache artifact")
    raw_parts: dict[str, Mapping[str, Any]] = {}
    for record in symbol_records:
        index = json.loads((virtual_cache_root / str(record["path"])).read_text(encoding="utf-8"))
        if index.get("schema") != "stage23_production_symbol_cache_index_v1" or index.get("symbol") != record["symbol"] or index.get("protected_cutoff") != "2026-01-01T00:00:00Z":
            raise LaunchPopulationAuthorityError("virtual symbol index contract differs")
        if index.get("source_part_inventory_sha256") != canonical_hash(index.get("source_parts", ())):
            raise LaunchPopulationAuthorityError("virtual symbol source-part inventory hash differs")
        for part in index["source_parts"]:
            if part.get("symbol") != record["symbol"] or int(part.get("rows", 0)) < 0:
                raise LaunchPopulationAuthorityError("source part identity is invalid")
            identity = str(part["path"])
            prior = raw_parts.get(identity)
            if prior is not None and prior != part:
                raise LaunchPopulationAuthorityError("source part path has conflicting identities")
            raw_parts[identity] = part
    if len(raw_parts) != EXPECTED_RAW_SOURCE_PARTS:
        raise LaunchPopulationAuthorityError("raw source part inventory is incomplete")
    for raw, record in raw_parts.items():
        path = Path(raw)
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise LaunchPopulationAuthorityError("raw source part differs from virtual cache authority")
    pit_path = virtual_cache_root / str(pit_records[0]["path"])
    pit_census, membership = census_pit(pit_path, partitions)
    if pit_census["symbols"] != EXPECTED_SYMBOLS or pit_census["membership_content_sha256"] != virtual["pit_membership"]["membership_content_sha256"]:
        raise LaunchPopulationAuthorityError("PIT census differs from virtual cache manifest")
    virtual_fold_by_key = {(row["phase"], row["outer_fold_id"], row.get("inner_fold_id")): row for row in virtual["folds"]["partitions"]}
    for row in pit_census["partitions"]:
        expected = virtual_fold_by_key.get((row["phase"], row["outer_fold_id"], row["inner_fold_id"]))
        if expected is None or int(expected["eligible_decisions_5m"]) != int(row["counts"]["all"]["decisions_5m"]):
            raise LaunchPopulationAuthorityError("PIT fold count differs from virtual cache reconciliation")
    a1 = json.loads(a1_manifest_path.read_text(encoding="utf-8"))
    if a1.get("schema") != "stage24_a1_exact_pit_population_table_v1" or a1.get("rows") != EXPECTED_PHYSICAL_COUNTS["all"]["decisions_5m"] or a1.get("symbols") != EXPECTED_SYMBOLS or a1.get("pit_content_sha256") != pit_census["membership_content_sha256"] or a1.get("protected_rows") != 0:
        raise LaunchPopulationAuthorityError("A1 full population table does not bind the PIT census")
    a1_root = a1_manifest_path.parent.parent.parent
    for record in [*a1["common"].values(), *a1["features"].values()]:
        _verify_record(a1_root / str(record["path"]), record, label="A1 population component")
        if "daily_counts_path" in record:
            _verify_record(a1_root / str(record["daily_counts_path"]), {"bytes": record["daily_counts_bytes"], "sha256": record["daily_counts_sha256"]}, label="A1 daily-count component")
    a3 = json.loads(a3_manifest_path.read_text(encoding="utf-8")); a3_root = a3_manifest_path.parent.parent.parent
    if a3.get("schema") != "stage24_a3_exact_pit_first_crossing_table_v1" or a3.get("symbols") != EXPECTED_SYMBOLS or a3.get("protected_rows") != 0 or len(a3.get("shards", ())) != EXPECTED_SYMBOLS:
        raise LaunchPopulationAuthorityError("A3 sparse population manifest is invalid")
    for record in a3["shards"]:
        _verify_record(a3_root / str(record["path"]), record, label="A3 symbol shard")
    a3_census = census_a3(a3, a3_root, membership, partitions)
    benchmark = json.loads(benchmark_cache_manifest_path.read_text(encoding="utf-8"))
    benchmark_artifacts = list(benchmark.get("artifacts", ()))
    kda = sum(row.get("campaign_partition", {}).get("phase") == "kda02b_adjudication" for row in benchmark_artifacts)
    if len(benchmark_artifacts) != 567 or kda != 171:
        raise LaunchPopulationAuthorityError("benchmark semantic frame inventory is not the reviewed 396+171 probe")
    payload = {
        "schema": "stage24_launch_population_authority_v1", "status": "bound_outcome_free",
        "execution_input_authority": _file_binding(execution_authority_path),
        "fold_graph": _file_binding(fold_graph_path),
        "virtual_cache": {**_file_binding(virtual_cache_manifest_path), "root": str(virtual_cache_root.resolve()), "artifacts": len(artifacts), "symbol_indexes": len(symbol_records)},
        "source_authority": {
            "physical_parts": len(raw_parts), "source_bytes": int(source_verification["source_bytes"]),
            "source_rows": int(source_verification["source_rows"]),
            "source_record_inventory_sha256": str(virtual["semantic_contract"]["source_record_inventory_sha256"]),
            "raw_source_parts_physically_verified": True,
        },
        "funding_authority": {
            "symbols": int(funding["symbols"]), "rows": int(funding["rows"]),
            "inventory_sha256": str(funding["inventory_sha256"]),
            "manifest_sha256": str(virtual["semantic_contract"]["funding_manifest_sha256"]),
        },
        "pit_membership": _file_binding(pit_path),
        "a1_population_table": {**_file_binding(a1_manifest_path), "rows": int(a1["rows"]), "features": len(a1["features"]), "reuse": "existing_hash_bound_mmap_no_copy"},
        "a3_population_table": {**_file_binding(a3_manifest_path), "features": len(a3["features"]), "shards": len(a3["shards"]), "reuse": "existing_hash_bound_sparse_shards_no_copy"},
        "population_census": pit_census, "a3_sparse_census": a3_census,
        "family_routes": {
            "A1_COMPRESSION_V2": "A1 mmap decisions/features plus lazy raw-source FamilyInput construction",
            "A3_STARTER_RETEST_V3": "A3 sparse first-crossing shards plus lazy raw-source FamilyInput construction",
            "A4_TSMOM_V7": "PIT membership plus exact registered 8h/1d schedule and lazy raw-source FamilyInput construction",
            "A2_PRIOR_HIGH_RS_CONTEXT_V1": "parent-event identities only; context materialized lazily after exact parent binding",
        },
        "benchmark_semantic_frames": {**_file_binding(benchmark_cache_manifest_path), "artifacts": 567, "base_frames": 396, "kda02b_frames": 171, "classification": "benchmark_probe_only", "launch_input_authority": False},
        "outcome_firewall": {"economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False},
    }
    payload["authority_inventory_sha256"] = canonical_hash({key: value for key, value in payload.items() if key != "authority_inventory_sha256"})
    validate_launch_population_authority(payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build the immutable outcome-free Stage24 launch population authority")
    result.add_argument("--repository-root", type=Path, required=True)
    result.add_argument("--execution-authority", type=Path, required=True)
    result.add_argument("--virtual-cache-root", type=Path, required=True)
    result.add_argument("--virtual-cache-manifest", type=Path, required=True)
    result.add_argument("--fold-graph", type=Path, required=True)
    result.add_argument("--a1-manifest", type=Path, required=True)
    result.add_argument("--a3-manifest", type=Path, required=True)
    result.add_argument("--benchmark-cache-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    payload = build_launch_population_authority(
        repository_root=args.repository_root, execution_authority_path=args.execution_authority,
        virtual_cache_root=args.virtual_cache_root, virtual_cache_manifest_path=args.virtual_cache_manifest,
        fold_graph_path=args.fold_graph, a1_manifest_path=args.a1_manifest, a3_manifest_path=args.a3_manifest,
        benchmark_cache_manifest_path=args.benchmark_cache_manifest,
    )
    atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
