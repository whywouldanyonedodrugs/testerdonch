#!/usr/bin/env python3
"""Build and mechanically adjudicate the outcome-free KDA02 v2 contract."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_kraken_c01_foundation as foundation
from tools.build_kda01_contract_closure import load_timestamp_only_bars
from tools.qlmg_kda01_timestamp_repair import repaired_execution_records
from tools.qlmg_kda02_v2 import (
    ATTEMPTS,
    FEATURE_EXTENSION_CONTRACT,
    FEATURE_EXTENSION_HASH,
    GENERATOR_CONTRACT,
    GENERATOR_HASH,
    INACTIVE_LINEAGE_ID,
    TRANSLATION_ID,
    extend_causal_features,
    generate_parent_episodes_and_events,
)
from tools.qlmg_kraken_derivatives_state import (
    PROTECTED_START,
    TRAIN_START,
    assert_no_outcomes,
    sha256_file,
    stable_hash,
    validate_rankable_times,
)


TASK_ID = "donch_bt_stage_9_kda02_purge_adjudication_conditional_level3_20260719_v1"
STARTING_COMMIT = "baaa10c224807e1dc7e32bfee7227711cb0c1279"
STAGE8A_COMMIT = "41b64b52a9146669eb26dcf25a86523a35219b8d"
STAGE8A_KDA02_SHA256 = "c4d553267e2107beeb042bc22f3280013c41a962d1674b4942d2b7f0de5e2b43"
SEMANTIC_HASH = "289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60"
STAGE8A_FEATURE_HASH = "4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4"
STAGE8A_GENERATOR_HASH = "c285af918e317722e91e3c1df5e44eb48abf27c97fedfe251b0881e3aa8f0017"
ANALYTICS_MANIFEST_HASH = "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d"
COHORT_HASH = "5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636"
EXPECTED_STAGE8A_COUNTS = {
    "kda02_primary_active_purge": 21241,
    "kda02_primary_completed": 43946,
    "kda02_primary_oi_vacuum": 1176354,
    "kda02_robust_active_purge": 3089,
    "kda02_robust_completed": 7602,
    "kda02_robust_oi_vacuum": 0,
}
PRIMARY_BRANCHES = (
    "primary_negative_active_purge_continuation",
    "primary_positive_active_purge_continuation",
    "primary_negative_completed_purge_reversal",
    "primary_positive_completed_purge_reversal",
)
CONTROL_CLASSES = (
    "price_mark_displacement_and_structural_break_without_liquidation_or_oi",
    "liquidation_extreme_without_material_oi_reset",
    "material_oi_reset_without_liquidation_extreme",
    "ordinary_high_volume_shock_with_matched_displacement",
    "structural_continuation_or_reversal_without_analytics_confirmation",
    "btc_eth_market_wide_stress_day",
    "stage8a_kda02_overlap_ablation",
    INACTIVE_LINEAGE_ID,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def verify_stage8a(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.stage8a_kda02_tape) != STAGE8A_KDA02_SHA256:
        raise ValueError("Stage 8A KDA02 event-tape hash mismatch")
    summary = json.loads((args.stage8a_archive / "completion_summary.json").read_text())
    expected = {
        "semantic_contract_hash": SEMANTIC_HASH,
        "analytics_data_manifest_hash": ANALYTICS_MANIFEST_HASH,
        "cohort_hash": COHORT_HASH,
        "feature_contract_hash": STAGE8A_FEATURE_HASH,
        "generator_contract_hash": STAGE8A_GENERATOR_HASH,
        "protected_rows_opened": 0,
        "economic_outputs_computed": 0,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage 8A completion authority mismatch")
    register = pd.read_csv(args.stage8a_archive / "KDA_FAMILY_AND_ATTEMPT_REGISTER.csv")
    observed = register.loc[register.family_id.eq("KDA02")].set_index("attempt_id").event_count.astype(int).to_dict()
    if observed != EXPECTED_STAGE8A_COUNTS:
        raise ValueError(f"Stage 8A KDA02 count reconciliation failed: {observed}")
    manifest = json.loads((args.stage8a_archive / "KDA_FEATURE_CACHE_MANIFEST.json").read_text())
    if (
        manifest.get("feature_contract_hash") != STAGE8A_FEATURE_HASH
        or manifest.get("analytics_manifest_hash") != ANALYTICS_MANIFEST_HASH
        or manifest.get("cohort_hash") != COHORT_HASH
        or manifest.get("protected_rows_opened") != 0
    ):
        raise ValueError("Stage 8A feature-cache authority mismatch")
    return manifest


def load_ohlc(
    rows: Sequence[foundation.AuthorityRow], symbol: str, dataset: str, prefix: str
) -> tuple[pd.DataFrame, str]:
    selected = [row for row in rows if row.symbol == symbol and row.dataset == dataset]
    if not selected:
        raise ValueError(f"missing OHLC authority: {dataset}:{symbol}")
    columns = [
        "time", "open", "high", "low", "close", "venue_symbol", "resolution",
        "rankable_pre_holdout", "contains_protected_period",
    ]
    parts: list[pd.DataFrame] = []
    accepted_rows: list[foundation.AuthorityRow] = []
    for row in selected:
        parquet = pq.ParquetFile(row.parquet_path)
        if not set(columns).issubset(parquet.schema_arrow.names):
            continue
        raw = parquet.read(columns=columns).to_pandas()
        if (
            not raw.venue_symbol.eq(symbol).all()
            or not raw.resolution.eq("5m").all()
            or not raw.rankable_pre_holdout.map(foundation._as_bool).all()
            or raw.contains_protected_period.map(foundation._as_bool).any()
        ):
            raise ValueError(f"unsafe OHLC authority: {row.parquet_path}")
        raw["timestamp_utc"] = pd.to_datetime(pd.to_numeric(raw.time, errors="raise"), unit="ms", utc=True)
        for column in ("open", "high", "low", "close"):
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        parts.append(raw[["timestamp_utc", "open", "high", "low", "close"]])
        accepted_rows.append(row)
    if not parts:
        raise ValueError(f"no rankable bar-shaped OHLC authority: {dataset}:{symbol}")
    frame = pd.concat(parts, ignore_index=True).sort_values("timestamp_utc", kind="mergesort")
    frame = frame[(frame.timestamp_utc >= TRAIN_START) & (frame.timestamp_utc < PROTECTED_START)]
    duplicate = frame.duplicated("timestamp_utc", keep=False)
    if duplicate.any() and frame.loc[duplicate].groupby("timestamp_utc")[["open", "high", "low", "close"]].nunique().gt(1).any().any():
        raise ValueError("conflicting duplicate OHLC bars")
    frame = frame.drop_duplicates("timestamp_utc", keep="first").reset_index(drop=True)
    validate_rankable_times(frame.timestamp_utc)
    return frame.rename(columns={name: f"{prefix}_{name}" for name in ("open", "high", "low", "close")}), stable_hash([row.reference_id for row in accepted_rows])


def add_market_clusters(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    onset = pd.to_datetime(out.parent_onset_ts, utc=True, errors="raise")
    day = onset.dt.floor("D")
    block = day + pd.to_timedelta((onset.dt.hour // 6) * 6, unit="h")
    out["market_day_cluster_id"] = [
        "kda02v2_day_" + stable_hash({"translation_id": TRANSLATION_ID, "attempt": attempt, "date": date.isoformat()})[:32]
        for attempt, date in zip(out.attempt, day)
    ]
    out["market_6h_cluster_id"] = [
        "kda02v2_6h_" + stable_hash({"translation_id": TRANSLATION_ID, "attempt": attempt, "block": value.isoformat()})[:32]
        for attempt, value in zip(out.attempt, block)
    ]
    return out


def attempt_register() -> pd.DataFrame:
    rows = []
    for attempt in ATTEMPTS:
        for direction in ("negative", "positive"):
            for mechanism in ("active_purge_continuation", "completed_purge_reversal"):
                rows.append({
                    "multiplicity_family": "KDA02A",
                    "translation_id": TRANSLATION_ID,
                    "attempt_id": f"{attempt}_{direction}_{mechanism}",
                    "attempt": attempt,
                    "parent_direction": direction,
                    "mechanism": mechanism,
                    "registered_before_counts": True,
                    "robustness_only": attempt == "robustness",
                    "can_rescue_primary": False,
                    "feature_extension_hash": FEATURE_EXTENSION_HASH,
                    "generator_hash": GENERATOR_HASH,
                    "event_count": 0,
                })
    rows.append({
        "multiplicity_family": "KDA02B",
        "translation_id": INACTIVE_LINEAGE_ID,
        "attempt_id": "kda02b_oi_vacuum_inactive",
        "attempt": "inactive",
        "parent_direction": "both",
        "mechanism": "oi_vacuum_without_liquidation",
        "registered_before_counts": True,
        "robustness_only": False,
        "can_rescue_primary": False,
        "feature_extension_hash": "",
        "generator_hash": "",
        "event_count": 0,
    })
    return pd.DataFrame(rows)


def provisional_definitions() -> pd.DataFrame:
    rows = []
    for attempt in ATTEMPTS:
        for direction in ("negative", "positive"):
            for mechanism in ("active_purge_continuation", "completed_purge_reversal"):
                branch = f"{attempt}_{direction}_{mechanism}"
                for timeout in (1, 6):
                    row = {
                        "definition_id": f"kda02_v2_{branch}_timeout_{timeout}h",
                        "branch_id": branch,
                        "attempt": attempt,
                        "parent_direction": direction,
                        "mechanism": mechanism,
                        "timeout_hours": timeout,
                        "decision": "completed_trade_and_mark_confirmation_bar_availability",
                        "entry": "first_authorized_PF_5m_trade_bar_open_at_or_after_decision_ts",
                        "maximum_entry_delay_minutes": 10,
                        "exit": f"first_authorized_PF_5m_trade_bar_open_at_or_after_actual_entry_plus_{timeout}h",
                        "maximum_exit_delay_minutes": 10,
                        "position": "fixed_notional",
                        "base_round_trip_bps": 14,
                        "stress_round_trip_bps": 32,
                        "funding": "diagnostic_partitions_excluded_from_gates",
                        "primary_inference_cluster": "market_day_cluster_id",
                        "sensitivity_clusters": "market_6h_cluster_id|parent_episode_id",
                        "robustness_only": attempt == "robustness",
                        "can_rescue_primary": False,
                    }
                    row["definition_contract_hash"] = stable_hash(row)
                    rows.append(row)
    return pd.DataFrame(rows)


def schedule_gates(records: pd.DataFrame, definitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = records.loc[records.accepted].copy()
    matrix = accepted.groupby(
        ["branch_id", "definition_id", "year", "symbol"], sort=True
    ).size().rename("event_count").reset_index()
    rows = []
    primary = definitions.loc[definitions.attempt.eq("primary")]
    for definition in primary.itertuples(index=False):
        subset = accepted.loc[accepted.definition_id.eq(definition.definition_id)]
        years = subset.year.value_counts().to_dict()
        symbols = subset.symbol.value_counts()
        total = len(subset)
        checks = {
            "candidate_events_ge_100": total >= 100,
            "each_year_ge_20": all(int(years.get(year, 0)) >= 20 for year in (2023, 2024, 2025)),
            "symbols_ge_20": int(subset.symbol.nunique()) >= 20,
            "maximum_symbol_share_le_25pct": total > 0 and float(symbols.max() / total) <= .25,
            "duplicate_event_ids_zero": not subset.duplicated("event_id").any(),
            "duplicate_economic_addresses_zero": not subset.duplicated("economic_address").any(),
            "protected_rows_zero": not (pd.to_datetime(subset.exit_ts, utc=True) >= PROTECTED_START).any(),
        }
        rows.append({
            "branch_id": definition.branch_id,
            "definition_id": definition.definition_id,
            "timeout_hours": definition.timeout_hours,
            "accepted_events": total,
            "events_2023": int(years.get(2023, 0)),
            "events_2024": int(years.get(2024, 0)),
            "events_2025": int(years.get(2025, 0)),
            "symbol_count": int(subset.symbol.nunique()),
            "maximum_symbol_share": float(symbols.max() / total) if total else float("nan"),
            **checks,
            "definition_mechanically_feasible": all(checks.values()),
        })
    gates = pd.DataFrame(rows)
    branch_pass = gates.groupby("branch_id").definition_mechanically_feasible.all().to_dict()
    gates["branch_mechanically_feasible"] = gates.branch_id.map(branch_pass)
    return matrix, gates


def final_definitions(provisional: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    feasible = set(gates.loc[gates.branch_mechanically_feasible, "branch_id"])
    allowed = feasible | {"robustness_" + branch.removeprefix("primary_") for branch in feasible}
    result = provisional.loc[provisional.branch_id.isin(allowed)].copy().reset_index(drop=True)
    if result.definition_id.duplicated().any() or result.definition_contract_hash.duplicated().any():
        raise ValueError("duplicate final KDA02 definition identity")
    return result


def cluster_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_type, column in (("market_day_primary", "market_day_cluster_id"), ("market_6h_sensitivity", "market_6h_cluster_id")):
        work = events.copy()
        work["year"] = pd.to_datetime(work.parent_onset_ts, utc=True).dt.year
        for (attempt, year), group in work.groupby(["attempt", "year"], sort=True):
            clustered = group.groupby(column).agg(event_count=("event_id", "size"), symbol_count=("symbol", "nunique"))
            rows.append({
                "cluster_type": cluster_type,
                "attempt": attempt,
                "year": int(year),
                "cluster_count": len(clustered),
                "total_events": len(group),
                "largest_cluster_event_share": float(clustered.event_count.max() / len(group)),
                "events_per_cluster_median": float(clustered.event_count.median()),
                "events_per_cluster_max": int(clustered.event_count.max()),
                "symbols_per_cluster_median": float(clustered.symbol_count.median()),
                "symbols_per_cluster_max": int(clustered.symbol_count.max()),
            })
    return pd.DataFrame(rows)


def frozen_contract(definitions: pd.DataFrame, gates: pd.DataFrame) -> dict[str, Any]:
    feasible = sorted(gates.loc[gates.branch_mechanically_feasible, "branch_id"].unique())
    code_paths = (
        "tools/qlmg_kda02_v2.py",
        "tools/build_kda02_v2_prerun_freeze.py",
        "tools/qlmg_kda02_level3.py",
        "tools/run_kda02_level3.py",
        "unit_tests/test_kda02_v2.py",
        "unit_tests/test_kda02_level3.py",
    )
    contract = {
        "contract_version": "kda02_v2_level3_contract_v1_20260719",
        "translation_id": TRANSLATION_ID,
        "feature_extension_hash": FEATURE_EXTENSION_HASH,
        "generator_hash": GENERATOR_HASH,
        "semantic_contract_hash": SEMANTIC_HASH,
        "analytics_manifest_hash": ANALYTICS_MANIFEST_HASH,
        "cohort_hash": COHORT_HASH,
        "code_provenance": {
            "starting_commit": STARTING_COMMIT,
            "task_files_sha256": {path: sha256_file(ROOT / path) for path in code_paths},
        },
        "venue": "Kraken linear PF derivatives in manifest-authorized capped cohort",
        "rankable_interval": {"start_inclusive": TRAIN_START.isoformat(), "end_exclusive": PROTECTED_START.isoformat()},
        "eligible_primary_branches": feasible,
        "definitions": definitions.to_dict("records"),
        "execution": {
            "decision_ts": "completed signal-bar start plus five minutes",
            "entry": "first authorized PF 5m trade-bar open at or after decision_ts",
            "maximum_entry_delay_minutes": 10,
            "exit_target": "actual entry plus frozen timeout",
            "exit": "first authorized PF 5m trade-bar open at or after exit target",
            "maximum_exit_delay_minutes": 10,
            "non_overlap": "definition-local symbol-local using actual executable exit",
        },
        "position": "fixed_notional",
        "timeouts_hours": [1, 6],
        "costs_bps": {"base_all_in_round_trip": 14, "stress_all_in_round_trip": 32},
        "funding": "diagnostic exact/mixed/imputed/zero partitions; excluded from gates",
        "bootstrap": {"unit": "equal_weight_market_day", "resamples": 10000, "seed": 20260719},
        "primary_inference": "equal-weight trades within market_day_cluster_id then equal-weight market days",
        "sensitivity_clusters": ["market_6h_cluster_id", "parent_episode_id"],
        "gates": {
            "executed_trades_min": 100,
            "trades_each_2023_2024_2025_min": 20,
            "equal_market_day_base_mean_gt_bps": 0,
            "equal_market_day_base_median_gt_bps": 0,
            "market_day_bootstrap_95pct_lower_bound_min_bps": -5,
            "maximum_positive_market_day_contribution": .10,
            "maximum_positive_symbol_contribution": .25,
            "maximum_positive_year_contribution": .70,
            "equal_market_day_stress_mean_min_bps": -10,
        },
        "controls": list(CONTROL_CLASSES),
        "controls_executed": False,
        "kda02b_outcomes_executed": False,
        "economic_run_authorization": "conditional_once_after_matching_independent_preoutcome_review",
        "terminal_decisions": [
            "KDA02_level3_primary_pass_controls_required",
            "KDA02_level3_no_primary_pass_stop",
            "KDA02_v2_mechanically_unavailable",
        ],
    }
    contract["level3_contract_hash"] = stable_hash(contract)
    return contract


def artifact_manifest(root: Path, cache: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "ARTIFACT_MANIFEST.json"):
        if "handoff" in path.relative_to(root).parts:
            continue
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "drive_eligible": path.suffix not in {".parquet", ".duckdb"},
        })
    cache_files = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(cache.rglob("*"))
        if path.is_file() and path.suffix not in {".duckdb", ".tmp"}
    ]
    payload = {"task_id": TASK_ID, "files": files, "local_cache_files": cache_files}
    payload["manifest_content_hash"] = stable_hash(payload)
    write_json(root / "ARTIFACT_MANIFEST.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/opt/parquet/kraken_derivatives/analytics/stage9_kda02_v2_prerun_v1"))
    parser.add_argument("--stage8a-archive", type=Path, default=Path("docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1"))
    parser.add_argument("--stage8a-kda02-tape", type=Path, default=Path("/opt/testerdonch-stage8a-20260719/docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1/KDA02_EVENT_TAPE.parquet"))
    parser.add_argument("--market-manifest", type=Path, default=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"))
    parser.add_argument("--symbol-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    if (args.output / "KDA02_V2_EVENT_TAPE.parquet").exists():
        raise ValueError("fresh KDA02 v2 final tape path required")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    stage8a = verify_stage8a(args)
    authorities = foundation.load_safe_manifest(args.market_manifest)
    partitions = stage8a["partitions"][: args.symbol_limit or None]
    parent_paths: list[Path] = []
    event_paths: list[Path] = []
    total_episodes = total_events = 0
    selected_feature_columns = [
        "timestamp_utc", "trade_return_15m", "mark_return_15m",
        "liquidation_base_units_15m", "liquidation_to_lagged_oi_15m",
        "liquidation_intensity_15m_robust_z", "liquidation_intensity_15m_percentile",
        "liquidation_intensity_15m_normalization_valid",
        "oi_log_change_15m", "oi_change_15m_robust_z", "oi_change_15m_percentile",
        "oi_change_15m_normalization_valid", "price_displacement_15m",
        "price_displacement_15m_robust_z", "price_displacement_15m_percentile",
        "price_displacement_15m_normalization_valid", "exact_contiguous_15m_valid",
        "pre_window_oi_close_base_units",
    ]
    for number, partition in enumerate(partitions, 1):
        symbol = partition["symbol"]
        source = Path(partition["path"])
        if sha256_file(source) != partition["sha256"]:
            raise ValueError(f"Stage 8A feature hash mismatch: {symbol}")
        shard = args.cache / f"symbol={symbol}"
        manifest_path = shard / "manifest.json"
        if shard.exists():
            if not manifest_path.is_file():
                raise ValueError(f"incomplete KDA02 v2 shard: {symbol}")
            manifest = json.loads(manifest_path.read_text())
            expected = {
                "feature_extension_hash": FEATURE_EXTENSION_HASH,
                "generator_hash": GENERATOR_HASH,
                "stage8a_feature_sha256": partition["sha256"],
                "status": "complete",
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"stale KDA02 v2 shard: {symbol}")
            for key in ("features", "episodes", "events"):
                path = shard / f"{key}.parquet"
                if sha256_file(path) != manifest[f"{key}_sha256"]:
                    raise ValueError(f"KDA02 v2 shard hash mismatch: {symbol}:{key}")
        else:
            features = pq.ParquetFile(source).read().to_pandas()
            features["timestamp_utc"] = pd.to_datetime(features.timestamp_utc, utc=True, errors="raise")
            assert_no_outcomes(features.columns)
            trade, trade_ref = load_ohlc(authorities, symbol, "historical_trade_candles_5m", "trade")
            mark, mark_ref = load_ohlc(authorities, symbol, "historical_mark_candles_5m", "mark")
            features = features.merge(trade, on=["timestamp_utc", "trade_close"], how="inner", validate="one_to_one")
            features = features.merge(mark, on=["timestamp_utc", "mark_close"], how="inner", validate="one_to_one")
            if len(features) != int(partition["rows"]):
                raise ValueError(f"unexplained Stage 8A-to-OHLC attrition: {symbol}")
            features = extend_causal_features(features)
            refs = stable_hash({"stage8a": partition["sha256"], "trade": trade_ref, "mark": mark_ref})
            episodes, events = generate_parent_episodes_and_events(
                features,
                symbol=symbol,
                semantic_hash=SEMANTIC_HASH,
                analytics_manifest_hash=ANALYTICS_MANIFEST_HASH,
                cohort_hash=COHORT_HASH,
                source_refs=refs,
            )
            temp = shard.with_name(f".{shard.name}.tmp")
            if temp.exists():
                raise ValueError(f"stale temporary KDA02 v2 shard: {symbol}")
            temp.mkdir(parents=True)
            features[selected_feature_columns].to_parquet(temp / "features.parquet", index=False, compression="zstd")
            episodes.to_parquet(temp / "episodes.parquet", index=False, compression="zstd")
            events.to_parquet(temp / "events.parquet", index=False, compression="zstd")
            manifest = {
                "symbol": symbol,
                "feature_extension_hash": FEATURE_EXTENSION_HASH,
                "generator_hash": GENERATOR_HASH,
                "stage8a_feature_sha256": partition["sha256"],
                "feature_rows": len(features),
                "episode_count": len(episodes),
                "event_count": len(events),
                "features_sha256": sha256_file(temp / "features.parquet"),
                "episodes_sha256": sha256_file(temp / "episodes.parquet"),
                "events_sha256": sha256_file(temp / "events.parquet"),
                "status": "complete",
            }
            write_json(temp / "manifest.json", manifest)
            os.replace(temp, shard)
        if int(manifest["episode_count"]):
            parent_paths.append(shard / "episodes.parquet")
        if int(manifest["event_count"]):
            event_paths.append(shard / "events.parquet")
        total_episodes += int(manifest["episode_count"])
        total_events += int(manifest["event_count"])
        write_json(args.output / "watch_status.json", {
            "stage": "outcome_free_generation", "symbols_completed": number,
            "symbols_total": len(partitions), "parent_episodes": total_episodes,
            "events": total_events, "elapsed_seconds": time.monotonic() - started,
            "peak_rss_gib": rss_gib(),
        })
        print(f"[{number}/{len(partitions)}] {symbol}: episodes={manifest['episode_count']} events={manifest['event_count']}", flush=True)
    if not parent_paths or not event_paths:
        raise ValueError("KDA02 v2 generated no parent/event shards")
    con = duckdb.connect(str(args.cache / "reducer.duckdb"))
    con.execute("SET memory_limit='1GB'")
    con.execute("SET threads=2")
    parent_source = "read_parquet([" + ",".join("'" + str(path).replace("'", "''") + "'" for path in parent_paths) + "], union_by_name=true)"
    event_source = "read_parquet([" + ",".join("'" + str(path).replace("'", "''") + "'" for path in event_paths) + "], union_by_name=true)"
    parent_out = args.output / "KDA02_V2_PARENT_EPISODE_TAPE.parquet"
    raw_event_out = args.cache / "KDA02_V2_EVENT_TAPE_RAW.parquet"
    con.execute(f"COPY (SELECT * FROM {parent_source} ORDER BY symbol,parent_onset_ts,attempt,parent_direction) TO '{parent_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    con.execute(f"COPY (SELECT * FROM {event_source} ORDER BY symbol,decision_ts,branch_id) TO '{raw_event_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    duplicate_episodes = con.execute(f"SELECT count(*) FROM (SELECT parent_episode_id FROM {parent_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    duplicate_events = con.execute(f"SELECT count(*) FROM (SELECT event_id FROM {event_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    duplicate_addresses = con.execute(f"SELECT count(*) FROM (SELECT economic_address FROM {event_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    protected = con.execute(f"SELECT (SELECT count(*) FROM {parent_source} WHERE parent_decision_ts >= TIMESTAMPTZ '2026-01-01') + (SELECT count(*) FROM {event_source} WHERE decision_ts >= TIMESTAMPTZ '2026-01-01')").fetchone()[0]
    con.close()
    if duplicate_episodes or duplicate_events or duplicate_addresses or protected:
        raise ValueError(f"identity/protected hard gate failed: {duplicate_episodes}/{duplicate_events}/{duplicate_addresses}/{protected}")
    events = add_market_clusters(pd.read_parquet(raw_event_out))
    event_out = args.output / "KDA02_V2_EVENT_TAPE.parquet"
    events.sort_values(["symbol", "decision_ts", "branch_id"], kind="mergesort").to_parquet(event_out, index=False, compression="zstd")
    provisional = provisional_definitions()
    bars: dict[str, pd.DatetimeIndex] = {}
    timestamp_refs: dict[str, str] = {}
    for symbol in sorted(events.symbol.unique()):
        bars[symbol], timestamp_refs[symbol] = load_timestamp_only_bars(authorities, symbol)
    schedules = repaired_execution_records(events, provisional, bars)
    schedule_out = args.cache / "KDA02_V2_TIMESTAMP_ELIGIBILITY.parquet"
    schedules.to_parquet(schedule_out, index=False, compression="zstd")
    count_matrix, gates = schedule_gates(schedules, provisional)
    definitions = final_definitions(provisional, gates)
    attempts = attempt_register()
    attempts["event_count"] = attempts.attempt_id.map(events.branch_id.value_counts()).fillna(0).astype(int)
    write_csv(args.output / "KDA02_V2_COUNT_MATRIX.csv", count_matrix)
    write_csv(args.output / "KDA02_V2_FEASIBILITY_GATES.csv", gates)
    write_csv(args.output / "KDA02_V2_ATTEMPT_REGISTER.csv", attempts)
    write_csv(args.output / "KDA02_V2_MARKET_CLUSTER_SUMMARY.csv", cluster_summary(events))
    write_csv(args.output / "KDA02_LEVEL3_DEFINITION_REGISTER.csv", definitions)
    contract = frozen_contract(definitions, gates)
    contract["timestamp_authority_hash"] = stable_hash(timestamp_refs)
    contract["level3_contract_hash"] = stable_hash({key: value for key, value in contract.items() if key != "level3_contract_hash"})
    write_json(args.output / "KDA02_FINAL_LEVEL3_CONTRACT.json", contract)
    (args.output / "KDA02_STAGE8A_ADJUDICATION.md").write_text(
        "# KDA02 Stage 8A Adjudication\n\n"
        "Stage 8A is preserved as valid outcome-free feasibility provenance, not a code defect. "
        "Its primary active purge required liquidation z>=2 plus any negative one-hour OI change; "
        "primary completed purge used prior-bar liquidation extremity, current OI decline, and aligned "
        "five-minute trade/one-hour mark direction; primary OI vacuum required any OI decline and modest "
        "one-hour price displacement without liquidation. Reconciled counts are 21,241 / 43,946 / "
        "1,176,354 primary and 3,089 / 7,602 robust active/completed; robust OI vacuum remains a killed "
        "semantic duplicate with zero tape. No Stage 8A artifact or attempt identity was changed.\n",
        encoding="utf-8",
    )
    (args.output / "KDA02_V2_FEATURE_EXTENSION_CONTRACT.md").write_text(
        "# KDA02 v2 Feature Extension Contract\n\n"
        f"Feature contract hash: `{FEATURE_EXTENSION_HASH}`. Generator hash: `{GENERATOR_HASH}`.\n\n"
        "Exact three-bar completed windows form trade/mark returns, summed liquidation, lagged-OI intensity, "
        "and OI change. Absolute 15-minute trade return is the reviewed price-displacement interpretation. "
        "All scores use prior UTC days only; liquidation uses daily maxima while OI and displacement use daily "
        "medians. Median/MAD scale must be finite and nonzero. The liquidation/OI base unit cancels only under "
        "the preserved inferred common-base-unit semantics; native liquidation side is unavailable.\n",
        encoding="utf-8",
    )
    (args.output / "KDA02B_OI_VACUUM_LINEAGE_DECISION.md").write_text(
        "# KDA02B OI-Vacuum Lineage Decision\n\n"
        f"`{INACTIVE_LINEAGE_ID}` is separated from KDA02A. It has no economic contract, definition, event "
        "generation, outcome access, or rescue role in Stage 9. Stage 8A OI-vacuum provenance remains unchanged.\n",
        encoding="utf-8",
    )
    (args.output / "KDA02_LEVEL4_CONTROL_CONTRACT.md").write_text(
        "# KDA02 Frozen Level-4 Control Contract\n\nControls are registered and frozen but not executed:\n\n"
        + "\n".join(f"{index}. `{control}`" for index, control in enumerate(CONTROL_CLASSES, 1))
        + "\n\nNo control may alter or rescue a Level-3 primary result.\n",
        encoding="utf-8",
    )
    feasible_branches = int(gates.groupby("branch_id").branch_mechanically_feasible.first().sum())
    mechanical_status = "preoutcome_review_required" if feasible_branches >= 2 else "KDA02_v2_mechanically_unavailable"
    (args.output / "KDA02_PRERUN_REVIEW.md").write_text(
        "# KDA02 Independent Pre-Run Review\n\nStatus: `pending_independent_review`.\n\n"
        f"Mechanical status: `{mechanical_status}`. Feasible primary branches: `{feasible_branches}`. "
        f"Frozen contract hash: `{contract['level3_contract_hash']}`. No price outcome column has been opened.\n",
        encoding="utf-8",
    )
    (args.output / "VALIDATION.md").write_text(
        "# Validation\n\n"
        f"Outcome-free generation completed for `{len(partitions)}` symbols, `{total_episodes}` parent episodes, "
        f"and `{total_events}` candidates. Duplicate episodes/events/economic addresses and protected rows are "
        f"`0/0/0/0`. Feasible primary branches: `{feasible_branches}`. Economic outputs: `0`.\n",
        encoding="utf-8",
    )
    write_json(args.output / "preoutcome_summary.json", {
        "task_id": TASK_ID,
        "starting_commit": STARTING_COMMIT,
        "stage8a_commit": STAGE8A_COMMIT,
        "stage8a_kda02_sha256": STAGE8A_KDA02_SHA256,
        "feature_extension_hash": FEATURE_EXTENSION_HASH,
        "generator_hash": GENERATOR_HASH,
        "level3_contract_hash": contract["level3_contract_hash"],
        "parent_episode_count": total_episodes,
        "event_count": total_events,
        "feasible_primary_branches": feasible_branches,
        "definition_count": len(definitions),
        "mechanical_status": mechanical_status,
        "protected_rows_opened": 0,
        "economic_outputs_computed": 0,
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_gib": rss_gib(),
    })
    artifact_manifest(args.output, args.cache)
    print(json.dumps({"status": mechanical_status, "episodes": total_episodes, "events": total_events, "feasible_primary_branches": feasible_branches, "definitions": len(definitions)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
