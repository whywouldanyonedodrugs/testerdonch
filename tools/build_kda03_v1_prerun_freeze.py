#!/usr/bin/env python3
"""Build and mechanically adjudicate the outcome-free KDA03 v1 contract."""

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
from tools.qlmg_kda03_v1 import (
    ATTEMPTS,
    FEATURE_EXTENSION_CONTRACT,
    FEATURE_EXTENSION_HASH,
    GENERATOR_CONTRACT,
    GENERATOR_HASH,
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


TASK_ID = "donch_bt_stage_11_kda03_basis_shock_20260719_v2"
STARTING_COMMIT = "e841469984478f7436db824587eac46dcd454c6d"
STAGE8A_COMMIT = "41b64b52a9146669eb26dcf25a86523a35219b8d"
STAGE8A_KDA03_SHA256 = "b34f5744170c85d02b44d6a0bfc2b5e1d58e6a5d46784ee57488560200ab17df"
STAGE8A_CACHE_MANIFEST_SHA256 = "463e108569469309e064fd8168235064e8b242eb11668a1e186ec4ceb1cf5538"
SEMANTIC_HASH = "289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60"
STAGE8A_FEATURE_HASH = "4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4"
STAGE8A_GENERATOR_HASH = "c285af918e317722e91e3c1df5e44eb48abf27c97fedfe251b0881e3aa8f0017"
ANALYTICS_MANIFEST_HASH = "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d"
COHORT_HASH = "5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636"
EXPECTED_STAGE8A_COUNTS = {
    "kda03_basis_oi": 186265,
    "kda03_basis_no_price": 247169,
    "kda03_basis_liq_reset": 60836,
}
PRIMARY_BRANCHES = (
    "primary_negative_reference_led_catchup",
    "primary_positive_reference_led_catchup",
    "primary_negative_basis_impulse_continuation",
    "primary_positive_basis_impulse_continuation",
    "primary_negative_completed_basis_impulse_rejection",
    "primary_positive_completed_basis_impulse_rejection",
)
CONTROL_CLASSES = (
    "same_trade_mark_state_without_basis_shock",
    "kda03a_without_price_non_confirmation",
    "kda03a_without_stable_oi",
    "price_oi_impulse_without_basis_shock",
    "basis_price_impulse_without_oi_expansion",
    "price_only_structural_rejection",
    "basis_level_extreme_as_kda01_overlap_non_rescue",
    "basis_liquidation_oi_reset_as_kda02_overlap_non_rescue",
    "matched_btc_eth_basis_context",
    "timestamp_null",
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
    if sha256_file(args.stage8a_kda03_matrix) != STAGE8A_KDA03_SHA256:
        raise ValueError("Stage 8A KDA03 feasibility-matrix hash mismatch")
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
    matrix = pd.read_csv(args.stage8a_kda03_matrix)
    observed = matrix.groupby("definition_id").feasible_row_count.sum().astype(int).to_dict()
    if observed != EXPECTED_STAGE8A_COUNTS:
        raise ValueError(f"Stage 8A KDA03 count reconciliation failed: {observed}")
    manifest_path = args.stage8a_archive / "KDA_FEATURE_CACHE_MANIFEST.json"
    if sha256_file(manifest_path) != STAGE8A_CACHE_MANIFEST_SHA256:
        raise ValueError("Stage 8A feature-cache manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
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


def verified_trade_authority_hash(
    rows: Sequence[foundation.AuthorityRow], symbols: Sequence[str]
) -> str:
    """Hash-bind every official trade-bar shard used by the frozen schedule."""
    symbol_set = set(symbols)
    selected = sorted(
        (
            row for row in rows
            if row.dataset == "historical_trade_candles_5m" and row.symbol in symbol_set
        ),
        key=lambda row: (row.symbol, row.chunk_start, str(row.parquet_path)),
    )
    present = {row.symbol for row in selected}
    if present != symbol_set:
        raise ValueError(f"missing official trade-bar authority: {sorted(symbol_set - present)}")
    records = []
    for row in selected:
        actual = sha256_file(row.parquet_path)
        if actual != row.parquet_sha256:
            raise ValueError(f"official trade-bar payload hash mismatch: {row.parquet_path}")
        records.append({
            "dataset": row.dataset,
            "symbol": row.symbol,
            "chunk_start": row.chunk_start.isoformat(),
            "chunk_end": row.chunk_end.isoformat(),
            "parquet_path": str(row.parquet_path),
            "parquet_sha256": actual,
            "rows": row.rows,
        })
    return stable_hash(records)


def add_market_clusters(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    onset = pd.to_datetime(out.parent_onset_ts, utc=True, errors="raise")
    day = onset.dt.floor("D")
    block = day + pd.to_timedelta((onset.dt.hour // 6) * 6, unit="h")
    out["market_day_cluster_id"] = [
        "kda03_day_" + stable_hash({"translation_id": TRANSLATION_ID, "attempt": attempt, "date": date.isoformat()})[:32]
        for attempt, date in zip(out.attempt, day)
    ]
    out["market_6h_cluster_id"] = [
        "kda03_6h_" + stable_hash({"translation_id": TRANSLATION_ID, "attempt": attempt, "block": value.isoformat()})[:32]
        for attempt, value in zip(out.attempt, block)
    ]
    return out


def attempt_register() -> pd.DataFrame:
    rows = []
    for attempt in ATTEMPTS:
        for direction in ("negative", "positive"):
            for mechanism, family in (
                ("reference_led_catchup", "KDA03A"),
                ("basis_impulse_continuation", "KDA03B"),
                ("completed_basis_impulse_rejection", "KDA03C"),
            ):
                rows.append({
                    "multiplicity_family": family,
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
    return pd.DataFrame(rows)


def provisional_definitions() -> pd.DataFrame:
    rows = []
    for attempt in ATTEMPTS:
        for direction in ("negative", "positive"):
            for mechanism, archetype in (
                ("reference_led_catchup", "mean_reversion"),
                ("basis_impulse_continuation", "symmetric_directional"),
                ("completed_basis_impulse_rejection", "mean_reversion"),
            ):
                branch = f"{attempt}_{direction}_{mechanism}"
                for timeout in (1, 6):
                    row = {
                        "definition_id": f"kda03_v1_{branch}_timeout_{timeout}h",
                        "branch_id": branch,
                        "attempt": attempt,
                        "parent_direction": direction,
                        "mechanism": mechanism,
                        "timeout_hours": timeout,
                        "decision": "completed_parent_availability for A/B; completed basis+trade+mark rejection availability for C",
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
                        "payoff_archetype": archetype,
                        "intended_claim_scope": "directional_PF_futures_not_spread_or_arbitrage",
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
        total = len(subset)
        checks = {
            "candidate_events_ge_100": total >= 100,
            "market_day_clusters_ge_50": int(subset.market_day_cluster_id.nunique()) >= 50,
            "symbols_ge_10": int(subset.symbol.nunique()) >= 10,
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
            "market_day_cluster_count": int(subset.market_day_cluster_id.nunique()),
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
        raise ValueError("duplicate final KDA03 definition identity")
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
        "tools/qlmg_kda03_v1.py",
        "tools/build_kda03_v1_prerun_freeze.py",
        "tools/qlmg_kda03_level3.py",
        "tools/run_kda03_level3.py",
        "unit_tests/test_kda03_v1.py",
        "unit_tests/test_kda03_level3.py",
    )
    contract = {
        "contract_version": "kda03_level3_contract_v1_20260719",
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
        "material_dependence_rule": {
            "estimand": "material when either six-hour-cluster or parent-episode base mean has opposite positive/nonpositive sign from the market-day base mean",
            "context_bins": ["both_positive_BTC_ETH_basis_change", "both_negative_BTC_ETH_basis_change", "mixed_or_missing"],
            "context": "material when context bins with at least 20 market days contain both positive and nonpositive base means",
            "role": "routing diagnostic only; never modifies a definition",
        },
        "hard_gates": [
            "source_platform_purpose_interval_schema_and_hash_authority",
            "zero_protected_period_selection_leakage",
            "causal_feature_and_decision_availability",
            "deterministic_candidate_event_control_episode_and_run_identity",
            "platform_faithful_execution_fields_and_costs",
            "reproducible_arithmetic_manifests_tests_and_independent_review",
            "no_outcome_conditioned_contract_mutation",
            "no_same_sample_rescue_of_closed_translation",
        ],
        "routing_diagnostics": [
            "bootstrap_uncertainty",
            "market_day_contribution",
            "symbol_contribution",
            "positive_year_contribution",
            "per_year_event_count",
            "cluster_estimand_sensitivity",
            "execution_stress",
            "sample_and_episode_independence",
        ],
        "control_eligibility": {
            "requires_separate_task_authorization": True,
            "minimum_conditions": [
                "all_integrity_gates_pass",
                "declared_base_cost_mean_positive",
                "frozen_payoff_archetype_rule_satisfied",
                "independent_event_or_cluster_count_adequate_for_claim",
                "cluster_bootstrap_not_clearly_incompatible_with_claim",
                "no_single_event_or_technical_defect_explains_result",
            ],
            "controls_validate_candidate": False,
        },
        "independent_evidence_requirement": {
            "preoutcome_review": "required_hash_matching_approval_before_the_single_economic_run",
            "postrun_review": "required_independent_recomputation_and_evidence_review",
            "promotion": "requires_independent_or_prospective_evidence_beyond_this_same_sample_run",
        },
        "mechanical_gates": {"candidates_min": 100, "market_day_clusters_min": 50, "symbols_min": 10, "duplicate_event_and_economic_addresses": 0, "protected_rows": 0},
        "routing_policy": {
            "policy_id": "aggressive_conditional_alpha_gate_routing", "version": "1.0",
            "policy_sha256": "c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa",
            "base_market_day_mean_gt_bps": 0, "base_market_day_median_gt_bps": 0,
            "sample_limited_bootstrap_lower_lt_bps": -5,
            "execution_sensitive_stress_mean_lt_bps": -10,
            "narrow_sleeve_symbol_contribution_gt": .25,
            "conditional_context_year_contribution_gt": .70,
            "conditional_context_day_contribution_gt": .10,
        },
        "controls": list(CONTROL_CLASSES),
        "controls_executed": False,
        "economic_run_authorization": "conditional_once_after_matching_independent_preoutcome_review",
        "overall_status_vocabulary": ["KDA03_level3_routes_assigned", "KDA03_mechanically_unavailable", "blocked_with_exact_mechanical_remedy"],
        "route_vocabulary": ["translation_rejected", "sample_limited_prospective_candidate", "execution_sensitive_candidate", "narrow_sleeve_candidate", "conditional_context_candidate_unvalidated", "unconditional_control_candidate"],
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


def build_context_tape(feature_paths: Sequence[Path], output: Path) -> pd.DataFrame:
    """Build outcome-free breadth plus BTC/ETH basis context on the completed grid."""
    if not feature_paths:
        raise ValueError("no KDA03 feature shards for context")
    con = duckdb.connect()
    source = "read_parquet([" + ",".join("'" + str(path).replace("'", "''") + "'" for path in feature_paths) + "], union_by_name=true)"
    query = f"""
      SELECT timestamp_utc,
        count(*) FILTER (WHERE eligible AND basis_change_15m_normalization_valid AND exact_contiguous_15m_valid) AS eligible_denominator,
        count(*) FILTER (WHERE eligible AND basis_change_15m_normalization_valid AND exact_contiguous_15m_valid AND basis_change_15m_robust_z >= 2) AS primary_positive_shocks,
        count(*) FILTER (WHERE eligible AND basis_change_15m_normalization_valid AND exact_contiguous_15m_valid AND basis_change_15m_robust_z <= -2) AS primary_negative_shocks,
        count(*) FILTER (WHERE eligible AND basis_change_15m_normalization_valid AND exact_contiguous_15m_valid AND basis_change_15m_percentile >= .95) AS robustness_positive_shocks,
        count(*) FILTER (WHERE eligible AND basis_change_15m_normalization_valid AND exact_contiguous_15m_valid AND basis_change_15m_percentile <= .05) AS robustness_negative_shocks,
        max(CASE WHEN symbol='PF_XBTUSD' THEN basis_decimal END) AS btc_basis_decimal,
        max(CASE WHEN symbol='PF_XBTUSD' THEN basis_change_15m END) AS btc_basis_change_15m,
        max(CASE WHEN symbol='PF_ETHUSD' THEN basis_decimal END) AS eth_basis_decimal,
        max(CASE WHEN symbol='PF_ETHUSD' THEN basis_change_15m END) AS eth_basis_change_15m
      FROM {source}
      GROUP BY timestamp_utc ORDER BY timestamp_utc
    """
    context = con.execute(query).fetchdf()
    con.close()
    denominator = context.eligible_denominator.replace(0, np.nan)
    context["primary_signed_shock_breadth"] = (context.primary_positive_shocks - context.primary_negative_shocks) / denominator
    context["robustness_signed_shock_breadth"] = (context.robustness_positive_shocks - context.robustness_negative_shocks) / denominator
    context.to_parquet(output, index=False, compression="zstd")
    return context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/opt/parquet/kraken_derivatives/analytics/stage11_kda03_v1"))
    parser.add_argument("--stage8a-archive", type=Path, default=Path("docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1"))
    parser.add_argument("--stage8a-kda03-matrix", type=Path, default=Path("docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1/KDA03_FEASIBILITY_MATRIX.csv"))
    parser.add_argument("--market-manifest", type=Path, default=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"))
    parser.add_argument("--symbol-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    if (args.output / "KDA03_V1_EVENT_TAPE.parquet").exists():
        raise ValueError("fresh KDA03 v1 final tape path required")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    stage8a = verify_stage8a(args)
    authorities = foundation.load_safe_manifest(args.market_manifest)
    partitions = stage8a["partitions"][: args.symbol_limit or None]
    parent_paths: list[Path] = []
    event_paths: list[Path] = []
    feature_paths: list[Path] = []
    total_episodes = total_events = 0
    selected_feature_columns = [
        "symbol", "timestamp_utc", "basis_decimal", "basis_change_15m", "prior_basis_level",
        "basis_change_15m_robust_z", "basis_change_15m_percentile",
        "basis_change_15m_normalization_valid", "prior_basis_level_robust_z",
        "prior_basis_level_normalization_valid", "trade_return_15m", "mark_return_15m",
        "onset_trade_open", "onset_mark_open", "trade_abs_displacement_15m",
        "mark_abs_displacement_15m", "trade_displacement_15m_robust_z",
        "trade_displacement_15m_percentile", "trade_displacement_15m_normalization_valid",
        "mark_displacement_15m_robust_z", "mark_displacement_15m_percentile",
        "mark_displacement_15m_normalization_valid",
        "liquidation_base_units_15m", "liquidation_to_lagged_oi_15m",
        "liquidation_intensity_15m_robust_z", "liquidation_intensity_15m_percentile",
        "liquidation_intensity_15m_normalization_valid",
        "oi_log_change_15m", "oi_change_15m_robust_z", "oi_change_15m_percentile",
        "oi_change_15m_normalization_valid", "exact_contiguous_15m_valid", "eligible",
        "known_lifecycle_mask", "trade_coverage", "mark_coverage", "analytics_coverage",
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
                raise ValueError(f"incomplete KDA03 v1 shard: {symbol}")
            manifest = json.loads(manifest_path.read_text())
            expected = {
                "feature_extension_hash": FEATURE_EXTENSION_HASH,
                "generator_hash": GENERATOR_HASH,
                "stage8a_feature_sha256": partition["sha256"],
                "status": "complete",
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"stale KDA03 v1 shard: {symbol}")
            for key in ("features", "episodes", "events"):
                path = shard / f"{key}.parquet"
                if sha256_file(path) != manifest[f"{key}_sha256"]:
                    raise ValueError(f"KDA03 v1 shard hash mismatch: {symbol}:{key}")
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
            features["symbol"] = symbol
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
                raise ValueError(f"stale temporary KDA03 v1 shard: {symbol}")
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
        feature_paths.append(shard / "features.parquet")
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
        raise ValueError("KDA03 v1 generated no parent/event shards")
    con = duckdb.connect(str(args.cache / "reducer.duckdb"))
    con.execute("SET memory_limit='1GB'")
    con.execute("SET threads=2")
    parent_source = "read_parquet([" + ",".join("'" + str(path).replace("'", "''") + "'" for path in parent_paths) + "], union_by_name=true)"
    event_source = "read_parquet([" + ",".join("'" + str(path).replace("'", "''") + "'" for path in event_paths) + "], union_by_name=true)"
    parent_out = args.output / "KDA03_V1_PARENT_EPISODE_TAPE.parquet"
    raw_event_out = args.cache / "KDA03_V1_EVENT_TAPE_RAW.parquet"
    con.execute(f"COPY (SELECT * FROM {parent_source} ORDER BY symbol,parent_onset_ts,attempt,parent_direction) TO '{parent_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    con.execute(f"COPY (SELECT * FROM {event_source} ORDER BY symbol,decision_ts,branch_id) TO '{raw_event_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    duplicate_episodes = con.execute(f"SELECT count(*) FROM (SELECT parent_episode_id FROM {parent_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    duplicate_events = con.execute(f"SELECT count(*) FROM (SELECT event_id FROM {event_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    duplicate_addresses = con.execute(f"SELECT count(*) FROM (SELECT economic_address FROM {event_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    protected = con.execute(f"SELECT (SELECT count(*) FROM {parent_source} WHERE parent_decision_ts >= TIMESTAMPTZ '2026-01-01') + (SELECT count(*) FROM {event_source} WHERE decision_ts >= TIMESTAMPTZ '2026-01-01')").fetchone()[0]
    con.close()
    if duplicate_episodes or duplicate_events or duplicate_addresses or protected:
        raise ValueError(f"identity/protected hard gate failed: {duplicate_episodes}/{duplicate_events}/{duplicate_addresses}/{protected}")
    context_path = args.cache / "KDA03_V1_CONTEXT_TAPE.parquet"
    context = build_context_tape(feature_paths, context_path)
    events = add_market_clusters(pd.read_parquet(raw_event_out))
    events = events.merge(context, left_on="state_ts", right_on="timestamp_utc", how="left", validate="many_to_one").drop(columns="timestamp_utc")
    if events.eligible_denominator.isna().any():
        raise ValueError("missing KDA03 breadth/context at event timestamp")
    event_out = args.output / "KDA03_V1_EVENT_TAPE.parquet"
    events.sort_values(["symbol", "decision_ts", "branch_id"], kind="mergesort").to_parquet(event_out, index=False, compression="zstd")
    provisional = provisional_definitions()
    bars: dict[str, pd.DatetimeIndex] = {}
    timestamp_refs: dict[str, str] = {}
    event_symbols = sorted(events.symbol.unique())
    official_trade_authority_hash = verified_trade_authority_hash(authorities, event_symbols)
    for symbol in event_symbols:
        bars[symbol], timestamp_refs[symbol] = load_timestamp_only_bars(authorities, symbol)
    schedules = repaired_execution_records(events, provisional, bars)
    schedules = schedules.merge(events[["event_id", "market_day_cluster_id"]], on="event_id", how="left", validate="many_to_one")
    schedule_out = args.cache / "KDA03_V1_TIMESTAMP_ELIGIBILITY.parquet"
    schedules.to_parquet(schedule_out, index=False, compression="zstd")
    count_matrix, gates = schedule_gates(schedules, provisional)
    definitions = final_definitions(provisional, gates)
    attempts = attempt_register()
    attempts["event_count"] = attempts.attempt_id.map(events.branch_id.value_counts()).fillna(0).astype(int)
    write_csv(args.output / "KDA03_V1_COUNT_MATRIX.csv", count_matrix)
    write_csv(args.output / "KDA03_V1_FEASIBILITY_GATES.csv", gates)
    write_csv(args.output / "KDA03_V1_ATTEMPT_REGISTER.csv", attempts)
    write_csv(args.output / "KDA03_V1_MARKET_CLUSTER_SUMMARY.csv", cluster_summary(events))
    context_summary = events.groupby(["attempt", "branch_id"], sort=True).agg(
        events=("event_id", "size"), eligible_denominator_mean=("eligible_denominator", "mean"),
        primary_signed_shock_breadth_mean=("primary_signed_shock_breadth", "mean"),
        robustness_signed_shock_breadth_mean=("robustness_signed_shock_breadth", "mean"),
        btc_basis_change_mean=("btc_basis_change_15m", "mean"), eth_basis_change_mean=("eth_basis_change_15m", "mean"),
    ).reset_index()
    write_csv(args.output / "KDA03_V1_BREADTH_CONTEXT_SUMMARY.csv", context_summary)
    write_csv(args.output / "KDA03_LEVEL3_DEFINITION_REGISTER.csv", definitions)
    contract = frozen_contract(definitions, gates)
    contract["market_manifest_sha256"] = sha256_file(args.market_manifest)
    contract["official_trade_bar_authority_hash"] = official_trade_authority_hash
    contract["timestamp_authority_hash"] = stable_hash(timestamp_refs)
    contract["level3_contract_hash"] = stable_hash({key: value for key, value in contract.items() if key != "level3_contract_hash"})
    write_json(args.output / "KDA03_FINAL_LEVEL3_CONTRACT.json", contract)
    (args.output / "KDA03_STAGE8A_ADJUDICATION.md").write_text(
        "# KDA03 Stage 8A Adjudication\n\n"
        "Stage 8A is preserved as valid outcome-free feasibility provenance, not an economic event tape. "
        "The 494,270 broad row masks reconcile exactly: basis change plus OI 186,265; basis change without "
        "price confirmation 247,169; and extreme basis plus liquidation/OI reset 60,836. The last state is "
        "KDA02 overlap evidence and receives no KDA03 event or economics. No Stage 8A artifact was changed.\n",
        encoding="utf-8",
    )
    (args.output / "KDA03_V1_FEATURE_EXTENSION_CONTRACT.md").write_text(
        "# KDA03 v1 Feature Extension Contract\n\n"
        f"Feature contract hash: `{FEATURE_EXTENSION_HASH}`. Generator hash: `{GENERATOR_HASH}`.\n\n"
        "Exact three-bar completed windows form signed basis change, trade/mark open-to-close displacement, "
        "OI change, and summed liquidation divided by pre-window OI. The pre-shock basis is the immediately "
        "preceding completed state; onset opens are the first opens in the shock window. All scores use prior "
        "UTC days only; basis change uses all valid prior-day observations, liquidation uses prior daily "
        "maxima, and prior basis, price displacement, and OI use prior daily medians. Basis remains the "
        "signed inferred-authoritative decimal state. KDA03A is a directional PF-futures proxy, not arbitrage.\n",
        encoding="utf-8",
    )
    (args.output / "KDA03_SEMANTIC_AND_CLAIM_BOUNDARY.md").write_text(
        "# KDA03 Semantic and Claim Boundary\n\n"
        "Basis is `inferred_authoritative_v1`, positive when futures are above the spot/reference state. "
        "KDA03A is reference-led only as a proxy because a complete executable reference panel is unavailable. "
        "All KDA03 branches are directional PF-futures definitions, not spreads or arbitrage. The Stage 8A "
        "basis-plus-liquidation/OI-reset mask remains KDA02 overlap and is excluded from KDA03 economics.\n",
        encoding="utf-8",
    )
    (args.output / "KDA03_LEVEL4_CONTROL_CONTRACT.md").write_text(
        "# KDA03 Frozen Level-4 Control Contract\n\nControls are registered and frozen but not executed:\n\n"
        + "\n".join(f"{index}. `{control}`" for index, control in enumerate(CONTROL_CLASSES, 1))
        + "\n\nNo control may alter or rescue a Level-3 primary result.\n",
        encoding="utf-8",
    )
    feasible_branches = int(gates.groupby("branch_id").branch_mechanically_feasible.first().sum())
    mechanical_status = "preoutcome_review_required" if feasible_branches >= 1 else "KDA03_mechanically_unavailable"
    (args.output / "KDA03_PRERUN_REVIEW.md").write_text(
        "# KDA03 Independent Pre-Run Review\n\nStatus: `pending_independent_review`.\n\n"
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
        "stage8a_kda03_sha256": STAGE8A_KDA03_SHA256,
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
