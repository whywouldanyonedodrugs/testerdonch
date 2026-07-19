#!/usr/bin/env python3
"""Build the outcome-free Stage 8B1 KDA01 contract-closure artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import build_kraken_c01_foundation as c01
from tools.qlmg_kda01_contract_closure import (
    CONTRACT_VERSION,
    MAX_ENTRY_DELAY,
    MAX_EXIT_DELAY,
    attach_market_cluster_identity,
    execution_records,
    frozen_contract_hash,
    normalized_bar_times,
)
from tools.qlmg_kda01_v2 import (
    FEATURE_EXTENSION_HASH,
    GENERATOR_HASH,
    TRANSLATION_ID,
)
from tools.qlmg_kraken_derivatives_state import PROTECTED_START, TRAIN_START, sha256_file, stable_hash
from tools.telegram_notify import TelegramNotifier


TASK_ID = "donch_bt_stage_8b1_kda01_contract_closure_20260719_v1"
STARTING_COMMIT = "2a3d38545600eb39f70f91180fb237bc436a1ece"
SOURCE_LEVEL3_HASH = "2eef5efb631e49014ea239eef5b90d4f2d5932fdcd33e97ec26067b5288ef938"
SOURCE_MANIFEST_CONTENT_HASH = "569bb1c53ba58767091c42dbf96ead63b3df5a45ae7cad1b4f6d4425e5ef61d5"
SOURCE_MANIFEST_FILE_SHA256 = "ee7db729eac30363c6147984658777ed92dc06f1d436527a56efae5bb997f669"
SOURCE_PARENT_SHA256 = "ae470d06c7da049d145ac8e083d0c4711c9c2dd6e4ec32fedf79224e8885ddbd"
SOURCE_EVENT_SHA256 = "7c9e682380681aa6fc83161ae29dafab9a047bdb19fe1fa8ade3ba5a4eb9c2e5"
TIMESTAMP_ONLY_COLUMNS = (
    "time",
    "venue_symbol",
    "resolution",
    "rankable_pre_holdout",
    "contains_protected_period",
)
CONTROL_CLASSES = (
    "price_progress_path_without_oi_or_basis",
    "material_oi_without_basis",
    "directional_basis_without_oi",
    "structural_failure_after_price_only_parent",
    "ordinary_oi_basis_matched_parent_episodes",
    "btc_eth_parent_state",
    "kda01_v1_overlap_ablation_non_rescue",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def verify_stage8b(archive: Path) -> dict[str, Any]:
    manifest_path = archive / "ARTIFACT_MANIFEST.json"
    if sha256_file(manifest_path) != SOURCE_MANIFEST_FILE_SHA256:
        raise ValueError("Stage 8B artifact-manifest file SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    content = {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    if stable_hash(content) != manifest.get("manifest_content_hash") or manifest.get("manifest_content_hash") != SOURCE_MANIFEST_CONTENT_HASH:
        raise ValueError("Stage 8B artifact-manifest content hash mismatch")
    missing: list[str] = []
    mismatched: list[str] = []
    for item in manifest["files"]:
        path = archive / item["path"]
        if not path.is_file():
            missing.append(str(path))
        elif path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            mismatched.append(str(path))
    for item in manifest["local_cache_files"]:
        path = Path(item["path"])
        if not path.is_file():
            missing.append(str(path))
        elif path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            mismatched.append(str(path))
    if missing or mismatched:
        raise ValueError(f"Stage 8B object reconciliation failed: missing={len(missing)} mismatched={len(mismatched)}")
    parent = archive / "KDA01_V2_PARENT_EPISODE_TAPE.parquet"
    event = archive / "KDA01_V2_EVENT_TAPE.parquet"
    definitions = archive / "KDA01_LEVEL3_DEFINITION_REGISTER.csv"
    rules = json.loads((archive / "KDA01_LEVEL3_DECISION_RULES.json").read_text())
    if sha256_file(parent) != SOURCE_PARENT_SHA256 or sha256_file(event) != SOURCE_EVENT_SHA256:
        raise ValueError("Stage 8B tape authority mismatch")
    if rules.get("level3_contract_hash") != SOURCE_LEVEL3_HASH:
        raise ValueError("Stage 8B Level-3 authority mismatch")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "parent_path": parent,
        "event_path": event,
        "definition_path": definitions,
        "archive_objects": len(manifest["files"]),
        "cache_objects": len(manifest["local_cache_files"]),
    }


def load_timestamp_only_bars(rows: Sequence[c01.AuthorityRow], symbol: str) -> tuple[pd.DatetimeIndex, str]:
    selected = [row for row in rows if row.dataset == "historical_trade_candles_5m" and row.symbol == symbol]
    if not selected:
        raise ValueError(f"missing timestamp authority for {symbol}")
    arrays: list[np.ndarray] = []
    bar_rows: list[c01.AuthorityRow] = []
    for row in selected:
        parquet = pq.ParquetFile(row.parquet_path)
        if not set(TIMESTAMP_ONLY_COLUMNS).issubset(parquet.schema_arrow.names):
            continue
        raw = parquet.read(columns=list(TIMESTAMP_ONLY_COLUMNS)).to_pandas()
        if (
            not raw.venue_symbol.eq(symbol).all()
            or not raw.resolution.eq("5m").all()
            or not raw.rankable_pre_holdout.map(c01._as_bool).all()
            or raw.contains_protected_period.map(c01._as_bool).any()
        ):
            raise ValueError(f"timestamp authority identity mismatch: {row.parquet_path}")
        arrays.append(raw.time.to_numpy())
        bar_rows.append(row)
    if not bar_rows:
        raise ValueError(f"no rankable timestamp bar-shaped authority rows for {symbol}")
    values = np.concatenate(arrays) if arrays else np.array([], dtype="int64")
    times = pd.to_datetime(values, unit="ms", utc=True)
    times = normalized_bar_times(times[(times >= TRAIN_START) & (times < PROTECTED_START)])
    return times, stable_hash([row.reference_id for row in bar_rows])


def cluster_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_type, cluster_column in (
        ("market_day_primary", "market_day_cluster_id"),
        ("market_6h_sensitivity", "market_6h_cluster_id"),
    ):
        work = events.copy()
        work["year"] = pd.to_datetime(work.parent_onset_ts, utc=True).dt.year
        for (attempt, year), group in work.groupby(["attempt", "year"], sort=True):
            grouped = group.groupby(cluster_column).agg(event_count=("event_id", "size"), symbol_count=("symbol", "nunique"))
            row: dict[str, Any] = {
                "cluster_type": cluster_type,
                "attempt": attempt,
                "year": int(year),
                "cluster_count": len(grouped),
                "total_events": len(group),
                "largest_cluster_event_share": float(grouped.event_count.max() / len(group)),
            }
            for prefix, series in (("events_per_cluster", grouped.event_count), ("symbols_per_cluster", grouped.symbol_count)):
                row.update({
                    f"{prefix}_min": int(series.min()),
                    f"{prefix}_median": float(series.median()),
                    f"{prefix}_p90": float(series.quantile(0.90)),
                    f"{prefix}_p99": float(series.quantile(0.99)),
                    f"{prefix}_max": int(series.max()),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def amended_definitions(source: pd.DataFrame) -> pd.DataFrame:
    if len(source) != 16 or set(source.timeout_hours) != {1, 6}:
        raise ValueError("Stage 8B definition register authority mismatch")
    rows = []
    for record in source.sort_values("definition_id", kind="mergesort").to_dict("records"):
        old_hash = record.pop("definition_contract_hash")
        amended = {
            **record,
            "source_definition_contract_hash": old_hash,
            "expected_entry_grid": "first_5m_grid_timestamp_strictly_after_decision",
            "maximum_entry_delay_minutes": 10,
            "exit_target": "actual_entry_ts_plus_frozen_timeout",
            "maximum_exit_delay_minutes": 10,
            "primary_inference_cluster": "market_day_cluster_id",
            "sensitivity_clusters": "parent_episode_id|market_6h_cluster_id",
        }
        amended["definition_contract_hash"] = stable_hash(amended)
        rows.append(amended)
    result = pd.DataFrame(rows)
    if result.definition_contract_hash.duplicated().any():
        raise ValueError("duplicate amended definition contract hash")
    return result


def availability_outputs(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = records.copy()
    work["final_status"] = np.where(work.accepted, "accepted", work.status)
    counts = work.groupby(
        ["definition_id", "branch_id", "year", "symbol", "final_status"], sort=True
    ).size().rename("event_count").reset_index()
    rejected = work.loc[~work.accepted, [
        "definition_id", "definition_contract_hash", "branch_id", "event_id", "economic_address",
        "symbol", "decision_ts", "year", "expected_entry_ts", "entry_ts", "entry_delay_minutes",
        "exit_target_ts", "exit_ts", "exit_delay_minutes", "status", "prior_event_id", "prior_exit_ts",
    ]].sort_values(["definition_id", "symbol", "decision_ts", "event_id"], kind="mergesort")
    return counts, rejected


def primary_feasibility(records: pd.DataFrame, definitions: pd.DataFrame) -> list[dict[str, Any]]:
    primary = definitions.loc[~definitions.robustness_only.map(c01._as_bool)]
    results = []
    for definition in primary.itertuples():
        accepted = records.loc[records.definition_id.eq(definition.definition_id) & records.accepted]
        by_year = accepted.groupby("year").size().to_dict()
        by_symbol = accepted.groupby("symbol").size()
        total = len(accepted)
        result = {
            "definition_id": definition.definition_id,
            "branch_id": definition.branch_id,
            "timeout_hours": int(definition.timeout_hours),
            "accepted_events": total,
            "events_2023": int(by_year.get(2023, 0)),
            "events_2024": int(by_year.get(2024, 0)),
            "events_2025": int(by_year.get(2025, 0)),
            "symbol_count": int(accepted.symbol.nunique()),
            "max_symbol_share": float(by_symbol.max() / total) if total else 1.0,
            "duplicate_definition_events": int(accepted.event_id.duplicated().sum()),
            "protected_rows": int((pd.to_datetime(accepted.exit_ts, utc=True) >= PROTECTED_START).sum()),
        }
        result["mechanically_feasible"] = bool(
            total >= 100
            and all(result[f"events_{year}"] >= 20 for year in (2023, 2024, 2025))
            and result["symbol_count"] >= 20
            and result["max_symbol_share"] <= 0.25
            and result["duplicate_definition_events"] == 0
            and result["protected_rows"] == 0
        )
        results.append(result)
    return results


def build_artifact_manifest(
    root: Path, source: dict[str, Any], market_manifest: Path, contract_hash: str,
    source_refs: dict[str, str],
) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in {"ARTIFACT_MANIFEST.json", "TRANSFER_MANIFEST.json"}):
        if "handoff" in path.relative_to(root).parts:
            continue
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "drive_eligible": path.suffix != ".parquet",
        })
    repository_paths = (
        "tools/qlmg_kda01_v2.py",
        "tools/build_kda01_v2_prerun_freeze.py",
        "unit_tests/test_kda01_v2_prerun_freeze.py",
        "tools/qlmg_kda01_contract_closure.py",
        "tools/build_kda01_contract_closure.py",
        "unit_tests/test_kda01_contract_closure.py",
    )
    repository_files = []
    for relative in repository_paths:
        path = REPOSITORY_ROOT / relative
        repository_files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    payload = {
        "task_id": TASK_ID,
        "starting_commit": STARTING_COMMIT,
        "stage8b_source_commit": STARTING_COMMIT,
        "contract_version": CONTRACT_VERSION,
        "level3_contract_hash": contract_hash,
        "market_manifest_path": str(market_manifest),
        "market_manifest_sha256": sha256_file(market_manifest),
        "timestamp_authority_refs": source_refs,
        "stage8b_source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
        "stage8b_source_manifest_content_hash": SOURCE_MANIFEST_CONTENT_HASH,
        "stage8b_source_manifest": source["manifest"],
        "repository_files": repository_files,
        "files": files,
        "protected_rows_opened": 0,
        "economic_outputs_computed": 0,
    }
    payload["manifest_content_hash"] = stable_hash(payload)
    write_json(root / "ARTIFACT_MANIFEST.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage8b-archive", type=Path, default=Path("/opt/testerdonch-stage8b-20260719/docs/agent/task_archive/20260719_donch_bt_stage_8b_kda01_prerun_freeze_20260719_v1"))
    parser.add_argument("--market-manifest", type=Path, default=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"))
    parser.add_argument("--tg-token", default="")
    parser.add_argument("--tg-chat-id", default="")
    parser.add_argument("--tg-auto-chat", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    notifier = TelegramNotifier.from_args(args, run_label="Stage 8B1 KDA01 contract closure")
    notifier.send("started", "economic_run=no; prices=no; outcomes=no")
    source = verify_stage8b(args.stage8b_archive)
    source_snapshot = args.output / "source_stage8b" / "ARTIFACT_MANIFEST.json"
    source_snapshot.parent.mkdir(parents=True, exist_ok=True)
    source_snapshot.write_bytes(source["manifest_path"].read_bytes())
    events = pq.ParquetFile(source["event_path"]).read().to_pandas()
    episodes = pq.ParquetFile(source["parent_path"]).read().to_pandas()
    forbidden = {"open", "high", "low", "close", "price", "return", "pnl", "mae", "mfe", "funding"}
    if forbidden & {column.lower() for column in set(events.columns) | set(episodes.columns)}:
        raise ValueError("economic or price column entered Stage 8B1")
    clustered = attach_market_cluster_identity(events, episodes)
    clustered.to_parquet(args.output / "KDA01_V2_EVENT_CLUSTER_IDENTITY.parquet", index=False, compression="zstd")
    summary = cluster_summary(clustered)
    write_csv(args.output / "KDA01_MARKET_CLUSTER_SUMMARY.csv", summary)
    definitions = amended_definitions(pd.read_csv(source["definition_path"]))
    write_csv(args.output / "KDA01_LEVEL3_DEFINITION_REGISTER_V2.csv", definitions)
    authorities = c01.load_safe_manifest(args.market_manifest)
    bars_by_symbol: dict[str, pd.DatetimeIndex] = {}
    source_refs: dict[str, str] = {}
    symbols = sorted(clustered.symbol.unique())
    for number, symbol in enumerate(symbols, 1):
        bars_by_symbol[symbol], source_refs[symbol] = load_timestamp_only_bars(authorities, symbol)
        if number % 10 == 0 or number == len(symbols):
            notifier.send("progress", f"timestamp_symbols={number}/{len(symbols)}")
    records = execution_records(clustered, definitions, bars_by_symbol)
    counts, rejections = availability_outputs(records)
    write_csv(args.output / "KDA01_LEVEL3_EXECUTION_AVAILABILITY_COUNTS.csv", counts)
    write_csv(args.output / "KDA01_LEVEL3_EXECUTION_REJECTIONS.csv", rejections)
    feasibility = primary_feasibility(records, definitions)
    all_primary_pass = all(row["mechanically_feasible"] for row in feasibility) and len(feasibility) == 8
    contract = {
        "translation_id": TRANSLATION_ID,
        "contract_version": CONTRACT_VERSION,
        "source_level3_contract_hash": SOURCE_LEVEL3_HASH,
        "feature_extension_hash": FEATURE_EXTENSION_HASH,
        "generator_hash": GENERATOR_HASH,
        "source_event_tape_sha256": SOURCE_EVENT_SHA256,
        "source_parent_tape_sha256": SOURCE_PARENT_SHA256,
        "definitions": definitions.to_dict("records"),
        "execution": {
            "entry": "first PF 5m trade-bar timestamp strictly after decision",
            "expected_next_open": "first five-minute UTC grid timestamp strictly after decision",
            "maximum_entry_delay_minutes": 10,
            "exit_target": "actual entry timestamp plus frozen 1h or 6h timeout",
            "exit": "first PF 5m trade-bar timestamp at or after exit target",
            "maximum_exit_delay_minutes": 10,
            "non_overlap": "definition-local and symbol-local using actual eligible exit_ts",
            "prices_read_in_closure": False,
        },
        "inference": {
            "primary_bootstrap_unit": "market_day_cluster_id",
            "sensitivity_units": ["parent_episode_id", "market_6h_cluster_id"],
            "resamples": 10000,
            "seed": 20260719,
            "no_pooling": "definitions|branches|horizons|directions|attempts",
        },
        "costs": {"base_round_trip_bps": 14, "stress_round_trip_bps": 32},
        "funding": "separate exact/mixed/imputed/zero partitions excluded from Level-3 gates",
        "position": "fixed_notional",
        "controls": list(CONTROL_CLASSES),
        "controls_executed": False,
        "primary_level3_gates": {
            "executed_trades_min": 100,
            "trades_per_year_min": 20,
            "base_mean_bps_gt": 0,
            "base_median_bps_gt": 0,
            "market_day_bootstrap_95pct_lower_bound_bps_min": -5,
            "max_positive_market_day_contribution": 0.10,
            "max_positive_symbol_contribution": 0.25,
            "max_positive_year_contribution": 0.70,
            "stress_mean_bps_min": -10,
        },
        "robustness_can_rescue_primary": False,
        "economic_run_authorized": False,
        "protected_rows_opened": 0,
        "economic_outputs_computed": 0,
    }
    contract_hash = frozen_contract_hash(contract)
    contract["level3_contract_hash"] = contract_hash
    write_json(args.output / "KDA01_LEVEL3_DECISION_RULES_V2.json", contract)
    status = "ready_for_human_KDA01_Level3_run_approval" if all_primary_pass else "KDA01_mechanically_unavailable_after_contract_closure"
    (args.output / "STAGE8B_HANDOFF_GAP_REPORT.md").write_text(
        "# Stage 8B Handoff Gap Report\n\n"
        f"The omitted source manifest was recovered and fully verified: `{source['archive_objects']}` archive objects and `{source['cache_objects']}` cache objects, with zero missing or mismatched files. "
        f"The reported `{SOURCE_MANIFEST_CONTENT_HASH}` value is its deterministic embedded content hash; the JSON file SHA-256 is `{SOURCE_MANIFEST_FILE_SHA256}`. "
        "The replacement handoff includes the exact source object at `source_stage8b/ARTIFACT_MANIFEST.json` and the complete Stage 8B1 manifest, which embeds the verified Stage 8B source manifest, all object hashes, and repository code identities.\n",
        encoding="utf-8",
    )
    (args.output / "KDA01_FINAL_LEVEL3_ECONOMIC_CONTRACT_V2.md").write_text(
        "# KDA01 Frozen Level-3 Economic Contract v2\n\n"
        f"Contract hash: `{contract_hash}`. Definitions remain `16`; controls remain seven and unexecuted. "
        "The primary inference unit is the cross-symbol, cross-direction `market_day_cluster_id` within attempt. "
        "Entry and exit availability use timestamp-only PF five-minute bars with exact ten-minute delay caps. "
        "Parent episodes and six-hour clusters are sensitivities only. No outcome was opened and no economic execution is authorized.\n",
        encoding="utf-8",
    )
    (args.output / "KDA01_PRERUN_APPROVAL_PACKET_V2.md").write_text(
        "# KDA01 Pre-Run Approval Packet v2\n\n"
        f"Status: `{status}`. Stage 8B event generation is unchanged. All eight primary timeout definitions pass the original mechanical gates after timestamp availability and definition-local non-overlap: `{all_primary_pass}`. "
        f"Frozen Level-3 contract hash: `{contract_hash}`. Human approval remains required before any economic runner implementation or execution.\n",
        encoding="utf-8",
    )
    completion = {
        "status": status,
        "source_manifest_archive_objects": source["archive_objects"],
        "source_manifest_cache_objects": source["cache_objects"],
        "event_count": len(clustered),
        "definition_count": len(definitions),
        "execution_record_count": len(records),
        "accepted_record_count": int(records.accepted.sum()),
        "rejection_count": int((~records.accepted).sum()),
        "primary_definition_feasibility": feasibility,
        "level3_contract_hash": contract_hash,
        "controls_frozen": len(CONTROL_CLASSES),
        "protected_rows_opened": 0,
        "economic_outputs_computed": 0,
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
    }
    write_json(args.output / "completion_summary.json", completion)
    (args.output / "COMPLETION.md").write_text(
        f"# Completion\n\nDecision: `{status}`. Stage 8B event generation and all economic semantics other than the explicitly amended inference cluster and availability caps remain unchanged. No economic output was computed.\n",
        encoding="utf-8",
    )
    (args.output / "NEXT_ACTION.md").write_text(
        "# Next Action\n\nHuman review and explicit approval of the complete KDA01 Level-3 v2 contract. If approved, a separate task may implement and execute only the 16 frozen definitions and seven frozen controls.\n",
        encoding="utf-8",
    )
    build_artifact_manifest(args.output, source, args.market_manifest, contract_hash, source_refs)
    notifier.send("complete", f"status={status}; accepted={int(records.accepted.sum())}; rejected={int((~records.accepted).sum())}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        args = parse_args()
        TelegramNotifier.from_args(args, run_label="Stage 8B1 KDA01 contract closure").send(
            "failed", f"{type(exc).__name__}: {exc}"
        )
        raise
