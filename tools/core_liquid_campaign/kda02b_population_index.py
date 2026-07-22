from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, canonical_json_bytes, sha256_file
from .family_engines.kda02b_adjudication import cell_contract


UTC = timezone.utc
KDA_FAMILY = "KDA02B_SURVIVOR_ADJUDICATION_V1"
TAPE_FAMILY = "KDA02B"
LOCAL_UNAVAILABLE_REASON = "stage14_kda02b_final_eligible_false"
SELECTED_MODELS = (
    "Q_2023Q4", "Q_2024Q1", "Q_2024Q2", "Q_2024Q3", "Q_2024Q4",
    "Q_2025Q1", "Q_2025Q2", "Q_2025Q3", "Q_2025Q4",
)
REQUIRED_FEATURE_COLUMNS = (
    "timestamp_utc", "trade_return_1h", "mark_return_1h", "oi_log_change_1h",
    "liquidation_base_units_1h", "liquidation_intensity_robust_z",
    "liquidation_normalization_valid", "eligible", "known_lifecycle_mask",
    "trade_coverage", "mark_coverage", "analytics_coverage",
)


class KDA02BPopulationIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class PopulationExpectations:
    configurations: int
    cells: int
    variants_per_cell: int
    folds: int
    symbols: int
    eligible_symbols: int
    unavailable_symbols: int
    event_rows: int
    eligible_event_rows: int
    unavailable_event_rows: int
    eligible_unique_symbol_decisions: int
    unavailable_unique_symbol_decisions: int
    eligible_dispatch_units: int
    unavailable_dispatch_units: int
    eligible_coverage_positions: int
    unavailable_coverage_positions: int
    unavailable_symbols_with_events: int
    unavailable_symbols_without_events: int


PRODUCTION_EXPECTATIONS = PopulationExpectations(
    configurations=209,
    cells=19,
    variants_per_cell=11,
    folds=9,
    symbols=187,
    eligible_symbols=117,
    unavailable_symbols=70,
    event_rows=482_919,
    eligible_event_rows=466_348,
    unavailable_event_rows=16_571,
    eligible_unique_symbol_decisions=113_607,
    unavailable_unique_symbol_decisions=4_539,
    eligible_dispatch_units=5_129_828,
    unavailable_dispatch_units=182_281,
    eligible_coverage_positions=220_077,
    unavailable_coverage_positions=131_670,
    unavailable_symbols_with_events=61,
    unavailable_symbols_without_events=9,
)


@dataclass(frozen=True)
class PopulationSources:
    authority_path: Path
    execution_registry_path: Path
    event_manifest_path: Path
    feature_manifest_path: Path
    universe_path: Path
    retention_boundary_path: Path
    fold_thresholds_path: Path


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise KDA02BPopulationIndexError("KDA02B timestamp is timezone-naive")
    return result.astimezone(UTC)


def _resolve_record_path(raw: object, repository_root: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repository_root / path


def sources_from_authority(
    authority_path: Path,
    execution_registry_path: Path,
    repository_root: Path,
) -> PopulationSources:
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    records = [*authority.get("source_records", ()), *authority.get("kda02b_authority_records", ())]
    roles: dict[str, Path] = {}
    for record in records:
        role = str(record.get("role", ""))
        path = _resolve_record_path(record.get("path", ""), repository_root)
        if role in roles:
            raise KDA02BPopulationIndexError(f"duplicate execution-input authority role: {role}")
        if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise KDA02BPopulationIndexError(f"execution-input authority drift: {role}")
        roles[role] = path
    required = {
        "stage20_kda02b_event_tape_manifest",
        "stage20_kda02b_fold_local_thresholds",
        "stage8a_feature_cache_manifest",
        "campaign_universe_reconciliation",
        "stage14_kda02b_retention_boundary",
    }
    if not required <= set(roles):
        raise KDA02BPopulationIndexError(f"KDA02B source roles are absent: {sorted(required - set(roles))}")
    if not execution_registry_path.is_file():
        raise KDA02BPopulationIndexError("execution registry is absent")
    return PopulationSources(
        authority_path=authority_path,
        execution_registry_path=execution_registry_path,
        event_manifest_path=roles["stage20_kda02b_event_tape_manifest"],
        feature_manifest_path=roles["stage8a_feature_cache_manifest"],
        universe_path=roles["campaign_universe_reconciliation"],
        retention_boundary_path=roles["stage14_kda02b_retention_boundary"],
        fold_thresholds_path=roles["stage20_kda02b_fold_local_thresholds"],
    )


def _registry(path: Path, expected: PopulationExpectations) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = [row for row in rows if row.get("family_id") == KDA_FAMILY]
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        config = row.get("config")
        if not isinstance(config, Mapping):
            raise KDA02BPopulationIndexError("KDA02B registry configuration is absent")
        by_cell.setdefault(str(config.get("stage20_cell_id")), []).append(row)
    variants = {
        cell: {str(row["config"].get("adjudication_variant")) for row in cell_rows}
        for cell, cell_rows in by_cell.items()
    }
    if (
        len(selected) != expected.configurations
        or len(by_cell) != expected.cells
        or any(len(rows) != expected.variants_per_cell for rows in by_cell.values())
        or any(len(items) != expected.variants_per_cell for items in variants.values())
    ):
        raise KDA02BPopulationIndexError("frozen KDA02B configuration/cell/variant registry differs")
    return selected, by_cell


def _universe(sources: PopulationSources, expected: PopulationExpectations) -> tuple[dict[str, bool], dict[str, str]]:
    with sources.universe_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    campaign = [row for row in rows if str(row.get("final_campaign_eligible", "")).lower() == "true"]
    eligibility = {
        str(row["PF_symbol"]): str(row.get("KDA02B_final_eligible", "")).lower() == "true"
        for row in campaign
    }
    with sources.retention_boundary_path.open("r", encoding="utf-8", newline="") as handle:
        retention_rows = {str(row["symbol"]): dict(row) for row in csv.DictReader(handle)}
    if set(eligibility) != set(retention_rows):
        raise KDA02BPopulationIndexError("Stage14 KDA02B retention boundary and campaign universe differ")
    if (
        len(eligibility) != expected.symbols
        or sum(eligibility.values()) != expected.eligible_symbols
        or len(eligibility) - sum(eligibility.values()) != expected.unavailable_symbols
    ):
        raise KDA02BPopulationIndexError("KDA02B 117/70 symbol boundary differs")
    universe_by_symbol = {str(row["PF_symbol"]): row for row in campaign}
    boundary_hashes = {
        symbol: canonical_hash({
            "universe_record": universe_by_symbol[symbol],
            "retention_boundary_record": retention_rows[symbol],
        })
        for symbol in sorted(eligibility)
    }
    return eligibility, boundary_hashes


def _thresholds(path: Path, selected_models: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models")
    if not isinstance(models, Mapping) or not set(selected_models) <= set(models):
        raise KDA02BPopulationIndexError("KDA02B fold-local threshold models are incomplete")
    return {model: models[model] for model in selected_models}


def _feature_partitions(path: Path, symbols: set[str]) -> dict[str, Mapping[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("partitions")
    if not isinstance(rows, list):
        raise KDA02BPopulationIndexError("Stage8A feature manifest has no partitions")
    by_symbol = {str(row.get("symbol")): row for row in rows}
    if set(by_symbol) != symbols:
        raise KDA02BPopulationIndexError("Stage8A feature partitions differ from the campaign symbol boundary")
    return by_symbol


def _inventory_hasher() -> tuple[Any, Any]:
    digest = sha256()
    def add(row: Mapping[str, Any]) -> None:
        digest.update(canonical_json_bytes(dict(row)))
        digest.update(b"\n")
    return digest, add


def _event_schema():
    import pyarrow as pa
    return pa.schema([
        ("event_id", pa.string()), ("translation_id", pa.string()),
        ("cell_id", pa.string()), ("model_id", pa.string()), ("outer_fold_id", pa.string()),
        ("symbol", pa.string()), ("decision_ts", pa.timestamp("us", tz="UTC")),
        ("onset_ts", pa.timestamp("us", tz="UTC")), ("side", pa.string()), ("horizon", pa.string()),
        ("status", pa.string()), ("unavailable_reason", pa.string()),
        ("event_tape_sha256", pa.string()), ("source_selected_row_ordinal", pa.int64()),
        ("feature_partition_sha256", pa.string()), ("feature_row_ordinal", pa.int64()),
        ("feature_timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("feature_available_ts", pa.timestamp("us", tz="UTC")),
        ("retention_boundary_identity_sha256", pa.string()),
        ("event_locator_sha256", pa.string()),
    ])


def _coverage_schema():
    import pyarrow as pa
    return pa.schema([
        ("executable_attempt_id", pa.string()), ("canonical_economic_address_sha256", pa.string()),
        ("cell_id", pa.string()), ("adjudication_variant", pa.string()),
        ("outer_fold_id", pa.string()), ("symbol", pa.string()), ("status", pa.string()),
        ("unavailable_reason", pa.string()), ("registered_event_rows", pa.int64()),
        ("retention_boundary_identity_sha256", pa.string()),
        ("coverage_identity_sha256", pa.string()),
    ])


def _atomic_parquet_writer(path: Path, schema: Any):
    import pyarrow.parquet as pq
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise KDA02BPopulationIndexError(f"stale temporary output exists: {temporary}")
    return temporary, pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True)


def _write_rows(writer: Any, schema: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import pyarrow as pa
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()


def _feature_frame(partition: Mapping[str, Any]):
    import pandas as pd
    import pyarrow.parquet as pq
    path = Path(str(partition.get("path", "")))
    if not path.is_file() or sha256_file(path) != partition.get("sha256"):
        raise KDA02BPopulationIndexError(f"Stage8A feature partition hash differs: {partition.get('symbol')}")
    parquet = pq.ParquetFile(path)
    if not set(REQUIRED_FEATURE_COLUMNS) <= set(parquet.schema_arrow.names):
        raise KDA02BPopulationIndexError(f"Stage8A feature schema is incomplete: {partition.get('symbol')}")
    frame = pq.read_table(path, columns=list(REQUIRED_FEATURE_COLUMNS)).to_pandas()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if len(frame) != int(partition.get("rows", -1)) or not frame["timestamp_utc"].is_monotonic_increasing or frame["timestamp_utc"].duplicated().any():
        raise KDA02BPopulationIndexError(f"Stage8A feature timestamps/count differ: {partition.get('symbol')}")
    if (frame["timestamp_utc"] >= datetime(2026, 1, 1, tzinfo=UTC)).any():
        raise KDA02BPopulationIndexError(f"protected Stage8A feature row rejected: {partition.get('symbol')}")
    return frame


def _validate_feature_row(row: Mapping[str, Any], *, eligible_symbol: bool) -> None:
    required_finite = (
        "trade_return_1h", "mark_return_1h", "oi_log_change_1h", "liquidation_base_units_1h",
    )
    if eligible_symbol:
        if not all(bool(row[key]) for key in ("eligible", "known_lifecycle_mask", "trade_coverage", "mark_coverage", "analytics_coverage")):
            raise KDA02BPopulationIndexError("eligible KDA02B event lacks exact point-in-time coverage")
        if any(not math.isfinite(float(row[key])) for key in required_finite):
            raise KDA02BPopulationIndexError("eligible KDA02B event lacks a required finite raw feature")
        if bool(row["liquidation_normalization_valid"]) and not math.isfinite(float(row["liquidation_intensity_robust_z"])):
            raise KDA02BPopulationIndexError("eligible KDA02B event lacks valid liquidation normalization")


def _expected_side_and_horizon(cell_id: str) -> tuple[str, str]:
    axes = cell_contract(cell_id)["axes"]
    price_side = "short" if axes["price_state"] == "negative" else "long"
    side = price_side if axes["branch"] == "continuation" else ("long" if price_side == "short" else "short")
    return side, str(axes["horizon"])


def _check_counts(actual: Mapping[str, int], expected: PopulationExpectations) -> None:
    expected_counts = {
        "configurations": expected.configurations,
        "cells": expected.cells,
        "folds": expected.folds,
        "symbols": expected.symbols,
        "eligible_symbols": expected.eligible_symbols,
        "unavailable_symbols": expected.unavailable_symbols,
        "event_rows": expected.event_rows,
        "eligible_event_rows": expected.eligible_event_rows,
        "unavailable_event_rows": expected.unavailable_event_rows,
        "eligible_unique_symbol_decisions": expected.eligible_unique_symbol_decisions,
        "unavailable_unique_symbol_decisions": expected.unavailable_unique_symbol_decisions,
        "eligible_dispatch_units": expected.eligible_dispatch_units,
        "unavailable_dispatch_units": expected.unavailable_dispatch_units,
        "eligible_coverage_positions": expected.eligible_coverage_positions,
        "unavailable_coverage_positions": expected.unavailable_coverage_positions,
        "unavailable_symbols_with_events": expected.unavailable_symbols_with_events,
        "unavailable_symbols_without_events": expected.unavailable_symbols_without_events,
    }
    if dict(actual) != expected_counts:
        differences = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected_counts.items() if actual.get(key) != value}
        raise KDA02BPopulationIndexError(f"KDA02B population reconciliation differs: {differences}")


def build_kda02b_lazy_population_index(
    *,
    sources: PopulationSources,
    output_root: Path,
    expectations: PopulationExpectations = PRODUCTION_EXPECTATIONS,
    selected_models: Sequence[str] = SELECTED_MODELS,
) -> dict[str, Any]:
    """Compile the complete pre-outcome KDA02B event/coverage authority.

    The index contains locators only. It neither copies Stage8A partitions nor
    opens any post-entry price, funding, or economic-outcome value.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    if len(selected_models) != expectations.folds or len(set(selected_models)) != len(selected_models):
        raise KDA02BPopulationIndexError("selected KDA02B folds differ")
    selected, by_cell = _registry(sources.execution_registry_path, expectations)
    eligibility, boundary_hashes = _universe(sources, expectations)
    thresholds = _thresholds(sources.fold_thresholds_path, selected_models)
    feature_partitions = _feature_partitions(sources.feature_manifest_path, set(eligibility))
    event_manifest = json.loads(sources.event_manifest_path.read_text(encoding="utf-8"))
    if event_manifest.get("status") != "pass" or event_manifest.get("economic_outcome_reader_opened") is not False or event_manifest.get("protected_rows_opened") != 0:
        raise KDA02BPopulationIndexError("Stage20 event-tape manifest is not outcome-firewalled")
    files = event_manifest.get("files")
    if not isinstance(files, list) or {str(row.get("symbol")) for row in files} != set(eligibility):
        raise KDA02BPopulationIndexError("Stage20 event-tape symbol inventory differs")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise KDA02BPopulationIndexError(f"population-index output already exists: {output_root}")
    raw_temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        event_path = raw_temporary / "KDA02B_EVENT_INDEX.parquet"
        event_schema = _event_schema()
        event_tmp, event_writer = _atomic_parquet_writer(event_path, event_schema)
        event_digest, add_event_hash = _inventory_hasher()
        event_counts: Counter[tuple[str, str, str, str]] = Counter()
        row_status = Counter()
        eligible_pairs: set[tuple[str, str]] = set()
        unavailable_pairs: set[tuple[str, str]] = set()
        unavailable_symbols_with_events: set[str] = set()
        seen_event_locators: set[tuple[str, str, str, str, str]] = set()
        batch: list[dict[str, Any]] = []
        try:
            for file_record in sorted(files, key=lambda row: str(row.get("symbol"))):
                symbol = str(file_record["symbol"])
                tape_path = Path(str(file_record["path"]))
                if not tape_path.is_file() or tape_path.stat().st_size != int(file_record["bytes"]) or sha256_file(tape_path) != file_record["sha256"]:
                    raise KDA02BPopulationIndexError(f"Stage20 event tape hash differs: {symbol}")
                feature = _feature_frame(feature_partitions[symbol])
                timestamp_to_row = {timestamp.to_pydatetime(): int(index) for index, timestamp in enumerate(feature["timestamp_utc"])}
                table = pq.read_table(
                    tape_path,
                    columns=["event_id", "translation_id", "cell_id", "family", "symbol", "decision_ts", "side", "horizon", "model_id", "onset_ts"],
                    filters=[("family", "=", TAPE_FAMILY)],
                )
                tape_rows = [
                    row for row in table.to_pylist()
                    if str(row["cell_id"]) in by_cell and str(row["model_id"]) in selected_models
                ]
                tape_rows.sort(key=lambda row: (_utc(row["decision_ts"]), str(row["cell_id"]), str(row["model_id"]), str(row["event_id"])))
                for source_ordinal, row in enumerate(tape_rows):
                    cell = str(row["cell_id"]); model = str(row["model_id"]); decision = _utc(row["decision_ts"])
                    if str(row["symbol"]) != symbol:
                        raise KDA02BPopulationIndexError("event tape contains a cross-symbol row")
                    expected_side, expected_horizon = _expected_side_and_horizon(cell)
                    if str(row["side"]) != expected_side or str(row["horizon"]) != expected_horizon:
                        raise KDA02BPopulationIndexError("Stage20 side/horizon differs from the frozen cell grammar")
                    model_contract = thresholds[model]
                    evaluation_start = _utc(model_contract["evaluation_start"])
                    evaluation_end = _utc(model_contract["evaluation_end"])
                    if not evaluation_start <= decision < evaluation_end:
                        raise KDA02BPopulationIndexError("KDA02B event is outside its exact outer fold")
                    feature_timestamp = decision - timedelta(minutes=5)
                    feature_ordinal = timestamp_to_row.get(feature_timestamp)
                    if feature_ordinal is None:
                        raise KDA02BPopulationIndexError(f"KDA02B event lacks its exact decision-time feature row: {symbol}")
                    feature_row = feature.iloc[feature_ordinal]
                    _validate_feature_row(feature_row, eligible_symbol=eligibility[symbol])
                    status = "eligible" if eligibility[symbol] else "typed_unavailable"
                    reason = "" if eligibility[symbol] else LOCAL_UNAVAILABLE_REASON
                    locator_identity = {
                        "event_id": str(row["event_id"]), "cell_id": cell, "model_id": model,
                        "symbol": symbol, "decision_ts": decision.isoformat(),
                    }
                    locator_tuple = tuple(str(locator_identity[key]) for key in ("event_id", "cell_id", "model_id", "symbol", "decision_ts"))
                    if locator_tuple in seen_event_locators:
                        raise KDA02BPopulationIndexError("duplicate KDA02B event locator")
                    seen_event_locators.add(locator_tuple)
                    output = {
                        **locator_identity,
                        "translation_id": str(row["translation_id"]),
                        "outer_fold_id": model.removeprefix("Q_"),
                        "decision_ts": decision,
                        # Stage20 preserves a null onset for event grammars that
                        # have no separately defined onset.  It is optional
                        # locator metadata, so retain null exactly rather than
                        # inventing a timestamp or rejecting an eligible event.
                        "onset_ts": None if row["onset_ts"] is None else _utc(row["onset_ts"]),
                        "side": str(row["side"]), "horizon": str(row["horizon"]),
                        "status": status, "unavailable_reason": reason,
                        "event_tape_sha256": str(file_record["sha256"]),
                        "source_selected_row_ordinal": source_ordinal,
                        "feature_partition_sha256": str(feature_partitions[symbol]["sha256"]),
                        "feature_row_ordinal": feature_ordinal,
                        "feature_timestamp_utc": feature_timestamp,
                        "feature_available_ts": decision,
                        "retention_boundary_identity_sha256": boundary_hashes[symbol],
                        "event_locator_sha256": canonical_hash(locator_identity),
                    }
                    batch.append(output)
                    hash_row = {
                        key: value.isoformat() if isinstance(value, datetime) else value
                        for key, value in output.items()
                    }
                    add_event_hash(hash_row)
                    row_status[status] += 1
                    event_counts[(cell, model.removeprefix("Q_"), symbol, status)] += 1
                    pair = (symbol, decision.isoformat())
                    (eligible_pairs if status == "eligible" else unavailable_pairs).add(pair)
                    if status == "typed_unavailable":
                        unavailable_symbols_with_events.add(symbol)
                    if len(batch) >= 8192:
                        _write_rows(event_writer, event_schema, batch)
            _write_rows(event_writer, event_schema, batch)
        finally:
            event_writer.close()
        os.replace(event_tmp, event_path)

        coverage_path = raw_temporary / "KDA02B_COVERAGE_GRID.parquet"
        coverage_schema = _coverage_schema()
        coverage_tmp, coverage_writer = _atomic_parquet_writer(coverage_path, coverage_schema)
        coverage_digest, add_coverage_hash = _inventory_hasher()
        coverage_status = Counter()
        coverage_batch: list[dict[str, Any]] = []
        try:
            for row in sorted(selected, key=lambda item: str(item["executable_attempt_id"])):
                cell = str(row["config"]["stage20_cell_id"])
                variant = str(row["config"]["adjudication_variant"])
                for model in selected_models:
                    fold = model.removeprefix("Q_")
                    for symbol in sorted(eligibility):
                        status = "eligible" if eligibility[symbol] else "typed_unavailable"
                        reason = "" if eligibility[symbol] else LOCAL_UNAVAILABLE_REASON
                        identity = {
                            "executable_attempt_id": str(row["executable_attempt_id"]),
                            "cell_id": cell, "adjudication_variant": variant,
                            "outer_fold_id": fold, "symbol": symbol,
                        }
                        output = {
                            **identity,
                            "canonical_economic_address_sha256": str(row["canonical_economic_address_sha256"]),
                            "status": status, "unavailable_reason": reason,
                            "registered_event_rows": event_counts[(cell, fold, symbol, status)],
                            "retention_boundary_identity_sha256": boundary_hashes[symbol],
                            "coverage_identity_sha256": canonical_hash(identity),
                        }
                        coverage_batch.append(output); add_coverage_hash(output); coverage_status[status] += 1
                        if len(coverage_batch) >= 8192:
                            _write_rows(coverage_writer, coverage_schema, coverage_batch)
            _write_rows(coverage_writer, coverage_schema, coverage_batch)
        finally:
            coverage_writer.close()
        os.replace(coverage_tmp, coverage_path)

        eligible_dispatch = sum(
            count * len(by_cell[cell])
            for (cell, _fold, _symbol, status), count in event_counts.items()
            if status == "eligible"
        )
        unavailable_dispatch = sum(
            count * len(by_cell[cell])
            for (cell, _fold, _symbol, status), count in event_counts.items()
            if status == "typed_unavailable"
        )
        actual_counts = {
            "configurations": len(selected), "cells": len(by_cell), "folds": len(selected_models),
            "symbols": len(eligibility), "eligible_symbols": sum(eligibility.values()),
            "unavailable_symbols": len(eligibility) - sum(eligibility.values()),
            "event_rows": sum(row_status.values()), "eligible_event_rows": row_status["eligible"],
            "unavailable_event_rows": row_status["typed_unavailable"],
            "eligible_unique_symbol_decisions": len(eligible_pairs),
            "unavailable_unique_symbol_decisions": len(unavailable_pairs),
            "eligible_dispatch_units": eligible_dispatch,
            "unavailable_dispatch_units": unavailable_dispatch,
            "eligible_coverage_positions": coverage_status["eligible"],
            "unavailable_coverage_positions": coverage_status["typed_unavailable"],
            "unavailable_symbols_with_events": len(unavailable_symbols_with_events),
            "unavailable_symbols_without_events": expectations.unavailable_symbols - len(unavailable_symbols_with_events),
        }
        _check_counts(actual_counts, expectations)
        authority = json.loads(sources.authority_path.read_text(encoding="utf-8"))
        source_records = {
            "execution_input_authority": {"path": str(sources.authority_path.resolve()), "sha256": sha256_file(sources.authority_path)},
            "execution_registry": {"path": str(sources.execution_registry_path.resolve()), "sha256": sha256_file(sources.execution_registry_path)},
            "event_manifest": {"path": str(sources.event_manifest_path.resolve()), "sha256": sha256_file(sources.event_manifest_path)},
            "feature_manifest": {"path": str(sources.feature_manifest_path.resolve()), "sha256": sha256_file(sources.feature_manifest_path)},
            "universe": {"path": str(sources.universe_path.resolve()), "sha256": sha256_file(sources.universe_path)},
            "retention_boundary": {"path": str(sources.retention_boundary_path.resolve()), "sha256": sha256_file(sources.retention_boundary_path)},
            "fold_thresholds": {"path": str(sources.fold_thresholds_path.resolve()), "sha256": sha256_file(sources.fold_thresholds_path)},
        }
        files_manifest = [
            {"role": "event_index", "path": event_path.name, "bytes": event_path.stat().st_size, "sha256": sha256_file(event_path), "row_inventory_sha256": event_digest.hexdigest()},
            {"role": "coverage_grid", "path": coverage_path.name, "bytes": coverage_path.stat().st_size, "sha256": sha256_file(coverage_path), "row_inventory_sha256": coverage_digest.hexdigest()},
        ]
        manifest = {
            "schema": "stage24_kda02b_lazy_population_index_v1",
            "family_id": KDA_FAMILY,
            "selected_models": list(selected_models),
            "counts": actual_counts,
            "files": files_manifest,
            "file_inventory_sha256": canonical_hash(files_manifest),
            "source_records": source_records,
            "source_inventory_sha256": canonical_hash(source_records),
            "execution_input_authority_sha256": canonical_hash(authority),
            "event_status_contract": {
                "eligible": "causally eligible for lazy FamilyInput construction and shadow/economic dispatch",
                "typed_unavailable": "registered but not economically tested; skip only this exact local observation",
                "typed_unavailable_reason": LOCAL_UNAVAILABLE_REASON,
            },
            "feature_locator_contract": "feature_timestamp_utc=decision_ts-5m; feature_available_ts=decision_ts; exact Stage8A partition hash and row ordinal",
            "dispatch_contract": "209 bounded attempt jobs stream their exact cell rows across nine folds; 11 variants per cell; no per-opportunity supervisor jobs",
            "large_artifacts_copied": False,
            "economic_outcomes_opened": False,
            "protected_rows_opened": 0,
            "capitalcom_payload_opened": False,
            "status": "pass",
        }
        atomic_write_json(raw_temporary / "KDA02B_LAZY_POPULATION_MANIFEST.json", manifest)
        os.replace(raw_temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(raw_temporary, ignore_errors=True)
        raise


def validate_kda02b_lazy_population_index(root: Path, expected: PopulationExpectations = PRODUCTION_EXPECTATIONS) -> dict[str, Any]:
    import pyarrow.parquet as pq
    manifest_path = root / "KDA02B_LAZY_POPULATION_MANIFEST.json"
    if not manifest_path.is_file():
        raise KDA02BPopulationIndexError("KDA02B lazy population manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "stage24_kda02b_lazy_population_index_v1" or manifest.get("status") != "pass":
        raise KDA02BPopulationIndexError("KDA02B lazy population manifest schema/status differs")
    _check_counts(manifest.get("counts", {}), expected)
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_inventory_sha256") != canonical_hash(files):
        raise KDA02BPopulationIndexError("KDA02B lazy population file inventory differs")
    expected_rows = {"event_index": expected.event_rows, "coverage_grid": expected.eligible_coverage_positions + expected.unavailable_coverage_positions}
    for record in files:
        path = root / str(record.get("path", ""))
        role = str(record.get("role", ""))
        if role not in expected_rows or not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise KDA02BPopulationIndexError(f"KDA02B lazy population file drift: {role}")
        if pq.ParquetFile(path).metadata.num_rows != expected_rows[role]:
            raise KDA02BPopulationIndexError(f"KDA02B lazy population row count differs: {role}")
    event_path = root / next(str(row["path"]) for row in files if row["role"] == "event_index")
    coverage_path = root / next(str(row["path"]) for row in files if row["role"] == "coverage_grid")
    event_status = Counter()
    event_pairs: dict[str, set[tuple[str, str]]] = {"eligible": set(), "typed_unavailable": set()}
    event_locators: set[str] = set()
    unavailable_symbols: set[str] = set()
    for batch in pq.ParquetFile(event_path).iter_batches(columns=[
        "event_id", "cell_id", "model_id", "symbol", "decision_ts", "status",
        "unavailable_reason", "feature_timestamp_utc", "feature_available_ts",
        "event_locator_sha256", "retention_boundary_identity_sha256",
    ]):
        for row in batch.to_pylist():
            status = str(row["status"])
            if status not in event_pairs:
                raise KDA02BPopulationIndexError("KDA02B event index has an unknown status")
            decision = _utc(row["decision_ts"])
            if _utc(row["feature_timestamp_utc"]) + timedelta(minutes=5) != decision or _utc(row["feature_available_ts"]) != decision:
                raise KDA02BPopulationIndexError("KDA02B event index feature availability differs")
            if status == "typed_unavailable":
                if row["unavailable_reason"] != LOCAL_UNAVAILABLE_REASON:
                    raise KDA02BPopulationIndexError("KDA02B local-unavailable reason differs")
                unavailable_symbols.add(str(row["symbol"]))
            elif row["unavailable_reason"]:
                raise KDA02BPopulationIndexError("eligible KDA02B event carries an unavailable reason")
            locator_identity = {
                "event_id": str(row["event_id"]), "cell_id": str(row["cell_id"]),
                "model_id": str(row["model_id"]), "symbol": str(row["symbol"]),
                "decision_ts": decision.isoformat(),
            }
            locator = str(row["event_locator_sha256"])
            if locator != canonical_hash(locator_identity) or locator in event_locators:
                raise KDA02BPopulationIndexError("KDA02B event locator identity differs or duplicates")
            boundary = str(row["retention_boundary_identity_sha256"])
            if len(boundary) != 64 or any(character not in "0123456789abcdef" for character in boundary):
                raise KDA02BPopulationIndexError("KDA02B retention-boundary identity is absent")
            event_locators.add(locator); event_status[status] += 1
            event_pairs[status].add((str(row["symbol"]), decision.isoformat()))
    coverage_status = Counter()
    for batch in pq.ParquetFile(coverage_path).iter_batches(columns=["status", "unavailable_reason"]):
        for row in batch.to_pylist():
            status = str(row["status"])
            if status not in event_pairs or (status == "typed_unavailable") != (row["unavailable_reason"] == LOCAL_UNAVAILABLE_REASON):
                raise KDA02BPopulationIndexError("KDA02B coverage-grid status/reason differs")
            coverage_status[status] += 1
    if (
        event_status != Counter({"eligible": expected.eligible_event_rows, "typed_unavailable": expected.unavailable_event_rows})
        or len(event_pairs["eligible"]) != expected.eligible_unique_symbol_decisions
        or len(event_pairs["typed_unavailable"]) != expected.unavailable_unique_symbol_decisions
        or coverage_status != Counter({"eligible": expected.eligible_coverage_positions, "typed_unavailable": expected.unavailable_coverage_positions})
        or len(unavailable_symbols) != expected.unavailable_symbols_with_events
    ):
        raise KDA02BPopulationIndexError("KDA02B physical event/coverage reconciliation differs")
    source_records = manifest.get("source_records")
    if not isinstance(source_records, Mapping) or manifest.get("source_inventory_sha256") != canonical_hash(source_records):
        raise KDA02BPopulationIndexError("KDA02B population source inventory differs")
    for role, record in source_records.items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise KDA02BPopulationIndexError(f"KDA02B population source drift: {role}")
    if manifest.get("economic_outcomes_opened") is not False or manifest.get("protected_rows_opened") != 0 or manifest.get("large_artifacts_copied") is not False:
        raise KDA02BPopulationIndexError("KDA02B lazy population safety declaration differs")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build the complete Stage24 KDA02B lazy pre-outcome population index")
    result.add_argument("--repository-root", type=Path, default=Path.cwd())
    result.add_argument("--execution-input-authority", type=Path, required=True)
    result.add_argument("--execution-registry", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    sources = sources_from_authority(args.execution_input_authority.resolve(), args.execution_registry.resolve(), args.repository_root.resolve())
    manifest = build_kda02b_lazy_population_index(sources=sources, output_root=args.output.resolve())
    print(json.dumps({"status": "pass", "counts": manifest["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "KDA02BPopulationIndexError", "LOCAL_UNAVAILABLE_REASON", "PopulationExpectations",
    "PopulationSources", "PRODUCTION_EXPECTATIONS", "SELECTED_MODELS",
    "build_kda02b_lazy_population_index", "sources_from_authority",
    "validate_kda02b_lazy_population_index",
]
