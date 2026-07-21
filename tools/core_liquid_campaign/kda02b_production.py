from __future__ import annotations

import csv
import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cache import SemanticCacheWriter
from .canonical import atomic_write_json, canonical_hash, sha256_file
from .engine_types import ContextInputs, FamilyInput, FundingInput, KRAKEN_PLATFORM, SignalBar
from .family_engines.kda02b_adjudication import cell_contract, cell_contract_sha256
from .production_cache import (
    ALLOWED_CANDLE_COLUMNS,
    KDA02B_FEATURE_SOURCE_ROLES,
    ProductionCacheCompiler,
    ProductionCacheError,
    SourcePart,
)
from .production_inputs import _funding_rows
from .synthetic import with_source_authority


UTC = timezone.utc
PROTECTED_START = datetime(2026, 1, 1, tzinfo=UTC)
KDA_FAMILY = "KDA02B_SURVIVOR_ADJUDICATION_V1"
FEATURE_CONTRACT_SHA256 = "4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4"
ANALYTICS_CONTENT_SHA256 = "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d"
KDA_FEATURE_COLUMNS = (
    "timestamp_utc", "trade_close", "trade_return_1h", "mark_return_1h",
    "oi_log_change_1h", "liquidation_base_units_1h",
    "liquidation_intensity_robust_z", "liquidation_normalization_valid",
    "eligible", "known_lifecycle_mask", "trade_coverage", "mark_coverage",
    "analytics_coverage",
)
KDA_REQUIRED_FIELDS_AND_UNITS = {
    "trade_return_1h": "decimal_log_return; engine_multiplies_by_10000_to_bps",
    "mark_return_1h": "decimal_log_return; engine_multiplies_by_10000_to_bps",
    "oi_log_change_1h": "dimensionless_log_fraction",
    "liquidation_base_units_1h": "unsigned_base_unit_contract_quantity",
    "liquidation_normalization_valid": "boolean",
    "liquidation_intensity_robust_z": "dimensionless_robust_z",
    "basis_bps": "available_decimal_basis_times_10000; not_consumed_by_frozen_KDA02B_engine",
    "funding": "absolute_USD_per_contract_unit_at_exact_UTC_hour",
    "price": "Kraken_native_linear_PF_trade_and_mark",
    "eligibility": "frozen_Stage14_KDA02B_causal_OI_retention_boundary",
}
KDA_THRESHOLD_FIELDS = (
    "trade_abs_q80", "trade_abs_q100", "mark_abs_q80", "mark_abs_q100",
    "oi_q0", "oi_q20", "liquidation_q80", "liquidation_q100",
)


class KDA02BProductionError(RuntimeError):
    pass


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _record(path: Path, role: str, repository_root: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def supplemental_authority_records(repository_root: Path) -> list[dict[str, Any]]:
    stage8 = repository_root / "docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1"
    stage14 = repository_root / "docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1"
    paths = {
        "stage7c_analytics_data_manifest": Path("/opt/testerdonch/results/rebaseline/phase_kraken_futures_analytics_stage7c_20260717_v1_20260717_204226/KRAKEN_ANALYTICS_DATA_MANIFEST.json"),
        "stage8a_feature_cache_manifest": stage8 / "KDA_FEATURE_CACHE_MANIFEST.json",
        "stage8a_shared_feature_schema": stage8 / "KDA_SHARED_FEATURE_SCHEMA.json",
        "stage8a_semantic_contract": stage8 / "ANALYTICS_SEMANTIC_CONTRACT.json",
        "stage8a_validation_summary": stage8 / "KDA_VALIDATION_SUMMARY.json",
        "stage14_kda02b_retention_boundary": stage14 / "KDA02B_RETENTION_BOUNDARY.csv",
    }
    if set(paths) != KDA02B_FEATURE_SOURCE_ROLES:
        raise KDA02BProductionError("code-owned KDA02B supplemental role inventory differs")
    for role, path in paths.items():
        if not path.is_file():
            raise KDA02BProductionError(f"required KDA02B authority is absent: {role}")
    return [_record(paths[role], role, repository_root) for role in sorted(paths)]


def extend_execution_input_authority(base_path: Path, output_path: Path, repository_root: Path) -> dict[str, Any]:
    authority = json.loads(base_path.read_text(encoding="utf-8"))
    records = supplemental_authority_records(repository_root)
    result = {
        **authority,
        "schema": "stage24_execution_input_authority_v2",
        "kda02b_authority_records": records,
        "kda02b_authority_inventory_sha256": canonical_hash(records),
        "kda02b_feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "kda02b_analytics_content_sha256": ANALYTICS_CONTENT_SHA256,
        "kda02b_required_fields_and_units": KDA_REQUIRED_FIELDS_AND_UNITS,
        "kda02b_authority_binding": "supplemental exact source authority; base A1-A4 source composite remains byte-identical",
    }
    atomic_write_json(output_path, result)
    return result


def _verify_feature_authority(roles: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    analytics = json.loads(roles["stage7c_analytics_data_manifest"].read_text(encoding="utf-8"))
    feature_manifest = json.loads(roles["stage8a_feature_cache_manifest"].read_text(encoding="utf-8"))
    schema = json.loads(roles["stage8a_shared_feature_schema"].read_text(encoding="utf-8"))
    semantic = json.loads(roles["stage8a_semantic_contract"].read_text(encoding="utf-8"))
    if analytics.get("content_hash") != ANALYTICS_CONTENT_SHA256 or int(analytics.get("file_count", -1)) != 3672:
        raise KDA02BProductionError("Stage7C analytics manifest identity differs")
    if feature_manifest.get("analytics_manifest_hash") != ANALYTICS_CONTENT_SHA256 or feature_manifest.get("feature_contract_hash") != FEATURE_CONTRACT_SHA256:
        raise KDA02BProductionError("Stage8A cache does not bind the expected Stage7C analytics and feature contract")
    partitions = feature_manifest.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 187 or len({row.get("symbol") for row in partitions}) != 187:
        raise KDA02BProductionError("Stage8A cache does not contain exactly 187 symbol partitions")
    if schema.get("feature_contract_hash") != FEATURE_CONTRACT_SHA256 or schema.get("outcome_columns") != []:
        raise KDA02BProductionError("Stage8A shared feature schema differs or contains outcome columns")
    columns = set(schema.get("columns", ()))
    required = set(KDA_FEATURE_COLUMNS) | {"basis_bps"}
    if not required <= columns:
        raise KDA02BProductionError(f"Stage8A shared schema lacks KDA02B fields: {sorted(required - columns)}")
    if semantic.get("economic_use_gate", {}).get("permitted_now") is None:
        raise KDA02BProductionError("Stage8A semantic contract lacks its explicit use gate")
    verified_rows = 0
    for row in partitions:
        path = Path(str(row["path"])); manifest_path = Path(str(row["manifest_path"]))
        if not path.is_file() or path.stat().st_size <= 0 or sha256_file(path) != row.get("sha256"):
            raise KDA02BProductionError(f"Stage8A feature partition hash differs: {row.get('symbol')}")
        if not manifest_path.is_file():
            raise KDA02BProductionError(f"Stage8A partition manifest is absent: {row.get('symbol')}")
        verified_rows += int(row["rows"])
    if verified_rows != 39_279_314:
        raise KDA02BProductionError("Stage8A feature row reconciliation differs")
    return analytics, feature_manifest, schema


def _eligibility(universe_path: Path, retention_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    with universe_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = tuple(sorted(row["PF_symbol"] for row in rows if str(row["KDA02B_final_eligible"]).lower() == "true"))
    unavailable = tuple(sorted(row["PF_symbol"] for row in rows if str(row["final_campaign_eligible"]).lower() == "true" and str(row["KDA02B_final_eligible"]).lower() != "true"))
    if (len(eligible), len(unavailable)) != (117, 70) or len(set(eligible) | set(unavailable)) != 187:
        raise KDA02BProductionError("Stage14 KDA02B 117/70 retention boundary differs")
    retention_hash = sha256_file(retention_path)
    if retention_hash != "ab506f14ebaa5247cca7e1f5224d2b82490c5f788a2f4f451937e700b676e26c":
        raise KDA02BProductionError("Stage14 KDA02B retention-boundary bytes differ")
    return eligible, unavailable, retention_hash


def _registered_kda(execution_path: Path, stage19_search_space_path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    execution = [json.loads(line) for line in execution_path.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in execution if row["family_id"] == KDA_FAMILY]
    cells = tuple(sorted({str(row["config"]["stage20_cell_id"]) for row in rows}))
    variants = Counter(str(row["config"]["adjudication_variant"]) for row in rows)
    if len(rows) != 209 or len(cells) != 19 or set(variants.values()) != {19} or len(variants) != 11:
        raise KDA02BProductionError("frozen 209-address KDA02B registry differs")
    search = json.loads(stage19_search_space_path.read_text(encoding="utf-8"))
    by_cell = {str(row["cell_id"]): row for row in search.get("cells", ()) if row.get("family") == "KDA02B"}
    for cell in cells:
        if cell not in by_cell or cell_contract(cell)["axes"] != by_cell[cell].get("search_axes"):
            raise KDA02BProductionError(f"Stage20 KDA02B cell grammar differs: {cell}")
    return rows, cells


def _scan_event_tapes(
    event_manifest: Mapping[str, Any],
    cells: Sequence[str],
    models: Sequence[str],
    eligible: set[str],
    unavailable: set[str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    wanted = {(cell, model) for cell in cells for model in models}
    representatives: dict[tuple[str, str], dict[str, Any]] = {}
    eligible_rows = 0; unavailable_rows = 0
    eligible_pairs: set[tuple[str, str]] = set(); unavailable_pairs: set[tuple[str, str]] = set()
    unavailable_symbols_with_events: set[str] = set()
    for file_row in sorted(event_manifest["files"], key=lambda row: str(row["symbol"])):
        symbol = str(file_row["symbol"])
        table = pq.read_table(
            file_row["path"],
            columns=["event_id", "translation_id", "cell_id", "family", "symbol", "decision_ts", "side", "horizon", "model_id"],
            filters=[("family", "=", "KDA02B")],
        )
        for row in table.to_pylist():
            key = (str(row["cell_id"]), str(row["model_id"]))
            if key not in wanted:
                continue
            decision = _utc(row["decision_ts"])
            pair = (symbol, decision.isoformat())
            if symbol in eligible:
                eligible_rows += 1; eligible_pairs.add(pair)
                prior = representatives.get(key)
                candidate = {**row, "decision_ts": decision, "symbol": symbol}
                if prior is None or (symbol, decision, str(row["event_id"])) < (prior["symbol"], prior["decision_ts"], str(prior["event_id"])):
                    representatives[key] = candidate
            elif symbol in unavailable:
                unavailable_rows += 1; unavailable_pairs.add(pair); unavailable_symbols_with_events.add(symbol)
            else:
                raise KDA02BProductionError(f"event tape symbol is outside frozen 117/70 boundary: {symbol}")
    if set(representatives) != wanted:
        missing = sorted(wanted - set(representatives))
        raise KDA02BProductionError(f"one or more KDA02B cell/fold identities lack an eligible representative: {missing[:3]}")
    if (eligible_rows, len(eligible_pairs), unavailable_rows, len(unavailable_pairs)) != (466_348, 113_607, 16_571, 4_539):
        raise KDA02BProductionError("Stage20 selected-cell KDA02B event identity reconciliation differs")
    return representatives, {
        "eligible_stage20_cell_address_decision_rows": eligible_rows,
        "eligible_unique_symbol_decisions": len(eligible_pairs),
        "unavailable_stage20_cell_address_decision_rows": unavailable_rows,
        "unavailable_unique_symbol_decisions": len(unavailable_pairs),
        "unavailable_symbols_with_events": len(unavailable_symbols_with_events),
        "unavailable_symbols_without_events": len(unavailable) - len(unavailable_symbols_with_events),
    }


def _guarded_feature_frame(partition: Mapping[str, Any]):
    import pandas as pd
    import pyarrow.parquet as pq

    path = Path(str(partition["path"])); parquet = pq.ParquetFile(path)
    if not set(KDA_FEATURE_COLUMNS) <= set(parquet.schema_arrow.names):
        raise KDA02BProductionError(f"KDA02B feature partition lacks required columns: {partition['symbol']}")
    time_index = parquet.schema_arrow.names.index("timestamp_utc")
    for group_index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(group_index).column(time_index).statistics
        if stats is None or not stats.has_min_max or _utc(stats.max) >= PROTECTED_START:
            raise KDA02BProductionError(f"protected or unverifiable KDA02B feature row group rejected: {partition['symbol']}")
    frame = pd.read_parquet(path, columns=list(KDA_FEATURE_COLUMNS))
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if (frame.timestamp_utc >= PROTECTED_START).any() or len(frame) != int(partition["rows"]):
        raise KDA02BProductionError(f"KDA02B feature partition boundary/count differs: {partition['symbol']}")
    return frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)


def _feature_history(frame: Any, start: datetime, decision: datetime) -> tuple[dict[str, Any], ...]:
    source_end = decision - timedelta(minutes=5)
    selected = frame.loc[frame.timestamp_utc.ge(start - timedelta(minutes=5)) & frame.timestamp_utc.le(source_end)]
    if selected.empty or _utc(selected.timestamp_utc.iloc[-1]) != source_end:
        raise KDA02BProductionError("KDA02B exact decision-time feature row is absent")
    history: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        source_close = _utc(row.pop("timestamp_utc")) + timedelta(minutes=5)
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"eligible", "known_lifecycle_mask", "trade_coverage", "mark_coverage", "analytics_coverage", "liquidation_normalization_valid"}:
                converted[key] = bool(value) if value is not None else False
            elif value is None or (isinstance(value, float) and value != value):
                converted[key] = None
            else:
                converted[key] = float(value)
        history.append({**converted, "source_close_ts": source_close, "feature_available_ts": source_close})
    return tuple(history)


def _schedule_bars(parts: Sequence[SourcePart], decision: datetime, last_known_close: float) -> tuple[SignalBar, ...]:
    import pyarrow.parquet as pq

    start_ms = int(decision.timestamp() * 1000); end_ms = int((decision + timedelta(minutes=75)).timestamp() * 1000)
    times: set[int] = set()
    for part in parts:
        if part.dataset != "historical_trade_candles_5m":
            continue
        parquet = pq.ParquetFile(part.path)
        if set(parquet.schema_arrow.names) != ALLOWED_CANDLE_COLUMNS:
            raise KDA02BProductionError("KDA02B execution-schedule candle schema differs")
        table = pq.read_table(part.path, columns=["time"], filters=[("time", ">=", start_ms), ("time", "<", end_ms)])
        times.update(int(value) for value in table["time"].to_pylist())
    ordered = sorted(times)
    if len(ordered) < 13 or ordered[0] != start_ms or any(right - left != 300_000 for left, right in zip(ordered, ordered[1:])):
        raise KDA02BProductionError("KDA02B real execution-open schedule is absent or gapped")
    price = float(last_known_close)
    return tuple(
        SignalBar(
            datetime.fromtimestamp(value / 1000, tz=UTC), datetime.fromtimestamp(value / 1000, tz=UTC) + timedelta(minutes=5),
            price, price, price, price,
            datetime.fromtimestamp(value / 1000, tz=UTC) + timedelta(minutes=5),
            datetime.fromtimestamp(value / 1000, tz=UTC) + timedelta(minutes=5),
            True, True, None,
        )
        for value in ordered
    )


def _hardlink_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise KDA02BProductionError(f"superseding cache target already exists: {target}")
    shutil.copytree(source, target, copy_function=os.link)


def _merge_cache_manifests(
    base_root: Path,
    kda_root: Path,
    output_root: Path,
    authority: Mapping[str, Any],
    frame_records: Sequence[Mapping[str, Any]],
) -> Path:
    _hardlink_tree(base_root, output_root)
    base = json.loads((base_root / "SEMANTIC_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    kda = json.loads((kda_root / "SEMANTIC_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    for directory in ("frames", "components"):
        source_dir = kda_root / directory; target_dir = output_root / directory
        for source in source_dir.iterdir():
            target = target_dir / source.name
            if target.exists():
                if sha256_file(target) != sha256_file(source):
                    raise KDA02BProductionError("KDA02B cache component collides with different bytes")
            else:
                os.link(source, target)
    identity_by_path = {str(row["path"]): row for row in frame_records}
    kda_artifacts = [{
        **row,
        "kda02b_stage20_cell_id": identity_by_path[str(row["path"])]["cell_id"],
        "kda02b_model_id": identity_by_path[str(row["path"])]["model_id"],
        "kda02b_stage20_event_id": identity_by_path[str(row["path"])]["stage20_event_id"],
    } for row in kda["artifacts"]]
    artifacts = sorted([*base["artifacts"], *kda_artifacts], key=lambda row: (str(row["fold_id"]), str(row["symbol"]), str(row["decision_ts"]), str(row["frame_content_sha256"])))
    component_by_path = {str(row["path"]): row for row in base.get("components", ())}
    for row in kda.get("components", ()):
        prior = component_by_path.get(str(row["path"]))
        if prior is not None and prior != row:
            raise KDA02BProductionError("KDA02B component manifest identity conflicts")
        component_by_path[str(row["path"])] = row
    components = sorted(component_by_path.values(), key=lambda row: str(row["path"]))
    manifest = {
        **base,
        "artifacts": artifacts,
        "artifact_inventory_sha256": canonical_hash(artifacts),
        "components": components,
        "component_inventory_sha256": canonical_hash(components),
        "typed_unavailable": [row for row in base.get("typed_unavailable", ()) if row.get("family_id") != KDA_FAMILY],
        "kda02b_authority_inventory_sha256": authority["kda02b_authority_inventory_sha256"],
        "kda02b_artifact_inventory_sha256": canonical_hash(kda_artifacts),
        "kda02b_production_frames": len(kda_artifacts),
    }
    path = output_root / "SEMANTIC_CACHE_MANIFEST.json"
    atomic_write_json(path, manifest)
    return path


def build_kda02b_production_cache(
    *,
    repository_root: Path,
    packet_root: Path,
    base_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    authority_path = output_root / "EXECUTION_INPUT_AUTHORITY.json"
    authority = extend_execution_input_authority(packet_root / "EXECUTION_INPUT_AUTHORITY.json", authority_path, repository_root)
    verifier = ProductionCacheCompiler(authority_path, output_root / "source_audit", repository_root)
    verified_authority, roles = verifier._authority()
    _, feature_manifest, schema = _verify_feature_authority(roles)
    unit = verifier._read_json(roles["price_and_instrument_source_manifest"], "price_unit_manifest")
    symbols = verifier._symbols(unit, roles["campaign_universe_reconciliation"])
    source_parts = verifier._source_parts(roles["kraken_acquisition_manifest"], set(symbols))
    verifier._funding(roles["rankable_funding_package"], symbols)
    kda_preoutcome = verifier._kda02b(roles)
    eligible, unavailable, retention_hash = _eligibility(roles["campaign_universe_reconciliation"], roles["stage14_kda02b_retention_boundary"])
    kda_rows, cells = _registered_kda(
        packet_root / "FINAL_EXECUTION_REGISTRY.jsonl",
        repository_root / "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/ECONOMIC_TRANSLATION_REGISTRY.json",
    )
    thresholds_doc = json.loads(roles["stage20_kda02b_fold_local_thresholds"].read_text(encoding="utf-8"))
    models = tuple(sorted(name for name in thresholds_doc["models"] if name.startswith("Q_")))
    if models != ("Q_2023Q4", "Q_2024Q1", "Q_2024Q2", "Q_2024Q3", "Q_2024Q4", "Q_2025Q1", "Q_2025Q2", "Q_2025Q3", "Q_2025Q4"):
        raise KDA02BProductionError("frozen nine Stage20 KDA02B outer folds differ")
    for name in models:
        model_thresholds = thresholds_doc["models"][name].get("thresholds", {})
        if not set(KDA_THRESHOLD_FIELDS) <= set(model_thresholds):
            raise KDA02BProductionError(f"KDA02B fold-local threshold schema differs: {name}")
    event_manifest = json.loads(roles["stage20_kda02b_event_tape_manifest"].read_text(encoding="utf-8"))
    representatives, event_counts = _scan_event_tapes(event_manifest, cells, models, set(eligible), set(unavailable))
    partition_by_symbol = {str(row["symbol"]): row for row in feature_manifest["partitions"]}
    representative_symbols = tuple(sorted({str(row["symbol"]) for row in representatives.values()}))
    feature_frames = {symbol: _guarded_feature_frame(partition_by_symbol[symbol]) for symbol in representative_symbols}
    funding = {symbol: _funding_rows(roles["rankable_funding_package"], symbol) for symbol in representative_symbols}
    kda_cache_root = output_root / "kda02b_cache_build"
    writer = SemanticCacheWriter(kda_cache_root, verified_authority, authority_root=repository_root, synthetic_only=False)
    frame_records: list[dict[str, Any]] = []
    for cell in cells:
        for model_id in models:
            event = representatives[(cell, model_id)]
            symbol = str(event["symbol"]); decision = _utc(event["decision_ts"])
            model = thresholds_doc["models"][model_id]
            evaluation_start = _utc(model["evaluation_start"]); evaluation_end = _utc(model["evaluation_end"])
            history = _feature_history(feature_frames[symbol], evaluation_start, decision)
            final = history[-1]
            if final.get("eligible") is not True:
                raise KDA02BProductionError("selected KDA02B representative is not causally eligible")
            bars = _schedule_bars(source_parts[symbol], decision, float(final["trade_close"]))
            partition = {
                "phase": "kda02b_adjudication", "outer_fold_id": model_id.removeprefix("Q_"), "inner_fold_id": None,
                "training_start": _utc(model["training_start"]), "training_end_exclusive": _utc(model["training_end"]),
                "evaluation_start": evaluation_start, "evaluation_end_exclusive": evaluation_end,
            }
            metadata = {
                "production_input": True,
                "shadow_no_outcome_execution_schedule_only": True,
                "real_post_entry_price_rows_opened": 0,
                "economic_outcomes_opened": False,
                "protected_rows": 0,
                "campaign_partition": partition,
                "evaluation_start": evaluation_start,
                "evaluation_end_exclusive": evaluation_end,
                "eligible_days": int((evaluation_end - evaluation_start).days),
                "eligible_symbol_seconds": float((evaluation_end - evaluation_start).total_seconds() * len(eligible)),
                "base_gap_allowance_bps_per_hour": 0.25,
                "stress_gap_allowance_bps_per_hour": 0.50,
                "stage20_cell_id": cell,
                "stage20_cell_contract_sha256": cell_contract_sha256(cell),
                "stage20_model_id": model_id,
                "stage20_event_id": str(event["event_id"]),
                "stage20_translation_id": str(event["translation_id"]),
                "stage20_tape_side": str(event["side"]),
                "stage20_tape_horizon": str(event["horizon"]),
                "kda02b_feature_history": history,
                "fold_thresholds": {key: float(value) for key, value in model["thresholds"].items() if isinstance(value, (int, float))},
                "kda02b_feature_partition_sha256": str(partition_by_symbol[symbol]["sha256"]),
                "kda02b_feature_schema_sha256": sha256_file(roles["stage8a_shared_feature_schema"]),
                "kda02b_feature_contract_sha256": FEATURE_CONTRACT_SHA256,
                "kda02b_required_fields_and_units": KDA_REQUIRED_FIELDS_AND_UNITS,
                "kda02b_authority_inventory_sha256": authority["kda02b_authority_inventory_sha256"],
                "execution_schedule_identity": canonical_hash({"symbol": symbol, "open_ts": [bar.open_ts.isoformat() for bar in bars], "source": "verified_trade_timestamp_column_only"}),
            }
            frame = FamilyInput(
                KRAKEN_PLATFORM, symbol, model_id, decision, bars, (),
                tuple(FundingInput(timestamp, timestamp, absolute) for timestamp, absolute, _ in funding[symbol] if timestamp <= decision),
                {}, ContextInputs(), metadata,
            )
            frame = with_source_authority(frame, verified_authority)
            frame.validate()
            record = writer.add(frame)
            frame_records.append({"cell_id": cell, "model_id": model_id, "symbol": symbol, "decision_ts": decision.isoformat(), "stage20_event_id": str(event["event_id"]), **record})
    kda_manifest = writer.finalize()
    final_cache_root = output_root / "semantic_cache"
    final_manifest = _merge_cache_manifests(base_cache_root, kda_cache_root, final_cache_root, authority, frame_records)
    base_build_path = base_cache_root.parent / "PRODUCTION_FAMILY_INPUT_BUILD.json"
    base_build = json.loads(base_build_path.read_text(encoding="utf-8"))
    base_matrix = [row for row in base_build.get("family_fold_input_matrix", ()) if row.get("family") != KDA_FAMILY]
    kda_matrix = [{
        "family": KDA_FAMILY, "phase": "kda02b_adjudication", "outer_fold_id": row["model_id"].removeprefix("Q_"),
        "inner_fold_id": None, "stage20_cell_id": row["cell_id"], "symbol": row["symbol"], "status": "available",
        "frame_content_sha256": row["frame_content_sha256"],
    } for row in frame_records]
    build_report = {
        **base_build,
        "schema": "stage24_production_family_input_build_v2",
        "cache_manifest": str(final_manifest),
        "cache_manifest_sha256": sha256_file(final_manifest),
        "family_fold_input_matrix": [*base_matrix, *kda_matrix],
        "family_fold_rows": len(base_matrix) + len(kda_matrix),
        "available_rows": len(base_matrix) + len(kda_matrix),
        "typed_unavailable_rows": 0,
        "kda02b": {
            "literal_prior_unavailable_reason": "authorized Stage20 event tape has event identities but no raw decision-time derivative feature columns",
            "classification": ["omitted_authority_binding", "missing_CacheAuthority_to_FamilyInput_adapter", "missing_eligible_validity_check", "wrong_fold_binding", "hardcoded_gate_unavailability"],
            "genuine_source_or_coverage_unavailability": False,
            "required_fields_and_units": KDA_REQUIRED_FIELDS_AND_UNITS,
            "basis_available_but_not_consumed": True,
            "registered_configurations": len(kda_rows),
            "cells": len(cells), "folds": len(models), "symbols": len(symbols),
            "causally_eligible_symbols": len(eligible), "locally_unavailable_symbols": len(unavailable),
            "eligible_configuration_fold_symbol_observations": len(kda_rows) * len(models) * len(eligible),
            "locally_unavailable_configuration_fold_symbol_observations": len(kda_rows) * len(models) * len(unavailable),
            "representative_real_frames": len(frame_records),
            "representative_symbols": list(representative_symbols),
            "retention_boundary_sha256": retention_hash,
            "feature_manifest_sha256": sha256_file(roles["stage8a_feature_cache_manifest"]),
            "feature_schema_sha256": sha256_file(roles["stage8a_shared_feature_schema"]),
            "semantic_contract_sha256": sha256_file(roles["stage8a_semantic_contract"]),
            "analytics_manifest_sha256": sha256_file(roles["stage7c_analytics_data_manifest"]),
            **event_counts,
        },
        "source_verification_sha256": canonical_hash(verifier.accessed),
        "economic_outcomes_opened": False,
        "protected_rows": 0,
        "capitalcom_payload_opened": False,
        "large_artifacts_reused_by_hardlink": True,
    }
    atomic_write_json(output_root / "PRODUCTION_FAMILY_INPUT_BUILD.json", build_report)
    atomic_write_json(output_root / "KDA02B_RECONCILIATION.json", build_report["kda02b"])
    atomic_write_json(output_root / "KDA02B_FRAME_INVENTORY.json", {"schema": "stage24_kda02b_frame_inventory_v1", "frames": frame_records, "inventory_sha256": canonical_hash(frame_records)})
    return build_report


__all__ = [
    "KDA02BProductionError", "KDA_REQUIRED_FIELDS_AND_UNITS", "build_kda02b_production_cache",
    "extend_execution_input_authority", "supplemental_authority_records",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build the authority-bound Stage24 KDA02B production cache adapter")
    result.add_argument("--repository-root", type=Path, default=Path.cwd())
    result.add_argument("--packet-root", type=Path, required=True)
    result.add_argument("--base-cache-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    report = build_kda02b_production_cache(
        repository_root=args.repository_root.resolve(), packet_root=args.packet_root.resolve(),
        base_cache_root=args.base_cache_root.resolve(), output_root=args.output.resolve(),
    )
    print(json.dumps({"status": "pass", "cache_manifest_sha256": report["cache_manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
