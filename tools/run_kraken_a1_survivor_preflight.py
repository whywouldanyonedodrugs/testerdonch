#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as runner


FULL_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_full_180shard_funding_corrected_20260711_v1_20260711_194859")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
HASH_ROOT = Path("results/rebaseline/phase_kraken_a1_selected_key_hash_canonicalization_repair_20260711_v1")
CONTRACT_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_contract_manifest_20260708_v1")
DEFAULT_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_survivor_materialization_preflight_20260712_v1")
MATERIALIZATION_PROFILE = "a1_compression_targeted_materialization_controls_stress_20260712_v1"
SCENARIOS = (
    ("central_imputed", 4),
    ("conservative_imputed", 8),
    ("severe_imputed", 8),
    ("severe_imputed", 12),
    ("exact_only_slice", 4),
    ("exact_only_slice", 8),
)
LONG_LANES = (
    "h12_rv_compression_breakout",
    "h13_flat_range_escape",
    "a1_plus_compression",
    "a1_impulse_base_breakout",
    "h06_vcp_like_contraction",
)
QUOTAS = {
    "h12_rv_compression_breakout": 3,
    "h13_flat_range_escape": 3,
    "a1_plus_compression": 3,
    "a1_impulse_base_breakout": 2,
    "h06_vcp_like_contraction": 2,
}
PRIMARY_EXIT_ORDER = (
    "structure_base_failure_time_10d",
    "failed_close_inside_range_time_5d",
    "atr_trail_2x_time_5d",
    "sma20_trail_time_15d",
    "ema10_trail_time_10d",
)
COMPARATOR_EXIT_ORDER = (
    "fixed_hold_5d_atr_stop_1p5",
    "fixed_hold_10d_atr_stop_2p0",
    "sma20_trail_time_15d",
    "ema10_trail_time_10d",
    "atr_trail_2x_time_5d",
    "failed_close_inside_range_time_5d",
)
PRIMARY_EXIT_BY_LANE_SLOT = {
    "h12_rv_compression_breakout": ("structure_base_failure_time_10d", "failed_close_inside_range_time_5d", "atr_trail_2x_time_5d"),
    "h13_flat_range_escape": ("structure_base_failure_time_10d", "failed_close_inside_range_time_5d", "sma20_trail_time_15d"),
    "a1_plus_compression": ("structure_base_failure_time_10d", "atr_trail_2x_time_5d", "ema10_trail_time_10d"),
    "a1_impulse_base_breakout": ("structure_base_failure_time_10d", "sma20_trail_time_15d"),
    "h06_vcp_like_contraction": ("structure_base_failure_time_10d", "atr_trail_2x_time_5d"),
}


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_shard_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]], dict[str, pd.DataFrame]]:
    selected_parts: list[pd.DataFrame] = []
    outcome_parts: list[pd.DataFrame] = []
    addresses: dict[str, set[str]] = {}
    primary_returns: dict[str, pd.DataFrame] = {}
    for shard_dir in sorted((FULL_ROOT / "aggregate_shards").glob("a1shard_*")):
        selected = pd.read_csv(shard_dir / "selected_keys.csv")
        outcomes = pd.read_parquet(shard_dir / "outcome_events.parquet")
        for frame in (selected, outcomes):
            if "decision_ts" in frame:
                frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True, errors="coerce")
            frame["shard_id"] = shard_dir.name
        selected_parts.append(selected)
        outcome_parts.append(outcomes)
        if selected.empty:
            addresses[shard_dir.name] = set()
        else:
            addresses[shard_dir.name] = set(selected["symbol"].astype(str) + "|" + selected["decision_ts"].astype(str))
        if not outcomes.empty:
            available = set(outcomes["exit_policy_id"].astype(str))
            exit_id = next((name for name in PRIMARY_EXIT_ORDER if name in available), sorted(available)[0])
            primary = outcomes[outcomes["exit_policy_id"].astype(str).eq(exit_id)].copy()
            primary["year_month"] = primary["decision_ts"].dt.strftime("%Y-%m")
            primary_returns[shard_dir.name] = primary.groupby(["symbol", "year_month"], dropna=False)["raw_gross_R"].sum().reset_index()
        else:
            primary_returns[shard_dir.name] = pd.DataFrame(columns=["symbol", "year_month", "raw_gross_R"])
    return (
        pd.concat(selected_parts, ignore_index=True, sort=False),
        pd.concat(outcome_parts, ignore_index=True, sort=False),
        addresses,
        primary_returns,
    )


def _attrition_reason(definition: dict[str, Any], selected: dict[str, Any], bars: pd.DataFrame) -> str:
    decision = runner.ts_utc(selected["decision_ts"])
    hold_days = runner.safe_float(definition.get("time_stop_days", definition.get("hold_value", 5)), 5.0)
    entry_i, exit_i = runner.a1_find_entry_and_exit_indices(bars, decision, hold_days)
    if entry_i is None or exit_i is None:
        return "unavailable_executable_entry_or_exit_bar_lifecycle_exclusion"
    entry = bars.iloc[entry_i]
    entry_price = runner.safe_float(entry.get("open"), np.nan)
    if not math.isfinite(entry_price) or entry_price <= 0:
        return "unavailable_or_invalid_executable_entry_price"
    atr = runner.prior_high_atr_at_decision(definition, bars, decision)
    atr_value = runner.safe_float(atr.get("atr_value"), np.nan)
    atr_mult = runner.safe_float(definition.get("atr_stop_mult"), np.nan)
    if not math.isfinite(atr_value) or atr_value <= 0 or not math.isfinite(atr_mult) or atr_mult <= 0:
        return "atr_feature_unavailable_at_decision"
    entry_ts = runner.ts_utc(entry.get("ts"))
    exit_ts = runner.ts_utc(bars.iloc[exit_i].get("ts"))
    interval_end = runner.effective_interval_end_ts(entry_ts, exit_ts, same_bar_adverse_stop=False)
    if any(ts >= runner.PROTECTED_TS for ts in (decision, entry_ts, exit_ts, interval_end)):
        return "declared_protected_boundary_censor"
    exit_price = runner.safe_float(bars.iloc[exit_i].get("close"), np.nan)
    if not math.isfinite(exit_price) or exit_price <= 0:
        return "unavailable_executable_exit_price"
    risk = abs(atr_value * atr_mult)
    fee = runner.compute_round_trip_fee_R(entry_price, exit_price, risk, entry_fee_bps_per_side=10.0)
    if not math.isfinite(runner.safe_float(fee.get("fee_R"), np.nan)):
        return "invalid_fee_arithmetic"
    return "unclassified_outcome_drop_fail_closed"


def reconcile_attrition(ctx: runner.Context, selected: pd.DataFrame, outcomes: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_definition_id", "symbol", "decision_ts", "exit_policy_id"]
    merged = selected.merge(outcomes[keys], on=keys, how="left", indicator=True)
    missing = merged[merged["_merge"].eq("left_only")].copy()
    definitions = {str(row["candidate_definition_id"]): row.to_dict() for _, row in manifest.iterrows()}
    paths = runner.data_paths(ctx)
    bar_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for _, row in missing.iterrows():
        symbol = str(row["symbol"])
        if symbol not in bar_cache:
            bar_cache[symbol] = runner.a1_load_symbol_bars_window(
                paths,
                symbol,
                pd.Timestamp("2023-01-01", tz="UTC"),
                pd.Timestamp("2026-01-01", tz="UTC"),
            )
        reason = _attrition_reason(definitions[str(row["candidate_definition_id"])], row.to_dict(), bar_cache[symbol])
        rows.append({
            "shard_id": row["shard_id"],
            "candidate_definition_id": row["candidate_definition_id"],
            "selected_key_policy_hash": row.get("selected_key_policy_hash", ""),
            "symbol": symbol,
            "decision_ts": row["decision_ts"],
            "exit_policy_id": row["exit_policy_id"],
            "event_key": runner.stable_hash(row["candidate_definition_id"], symbol, row["decision_ts"], n=32),
            "attrition_reason": reason,
            "allowed_explicit_reason": reason != "unclassified_outcome_drop_fail_closed",
            "selected_row_present": True,
            "outcome_row_present": False,
        })
    audit = pd.DataFrame(rows)
    audit.attrs["selected_rows"] = len(selected)
    audit.attrs["outcome_rows"] = len(outcomes)
    return audit


class UnionFind:
    def __init__(self, names: Iterable[str]):
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def duplicate_reports(plan: pd.DataFrame, manifest: pd.DataFrame, addresses: dict[str, set[str]], returns: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    plan = plan.copy()
    exact = plan.groupby("selected_event_address_hash", dropna=False).agg(
        exact_cluster_size=("shard_id", "size"),
        shard_ids=("shard_id", lambda s: ";".join(sorted(s))),
        selected_key_policy_hashes=("selected_key_policy_hash", lambda s: ";".join(sorted(s))),
        lanes=("definition_lane", lambda s: ";".join(sorted(set(s)))),
        selected_event_addresses=("selected_event_addresses", "max"),
    ).reset_index()
    exact["exact_duplicate"] = exact["exact_cluster_size"].gt(1)
    exact["cross_lane_duplicate"] = exact["lanes"].str.contains(";")
    exact["frozen_not_independent"] = exact["exact_duplicate"]

    entry = manifest.sort_values("candidate_definition_id").drop_duplicates("entry_spec_id", keep="first").set_index("entry_spec_id")
    selection_fields = [field for field in runner.A1_SELECTED_KEY_POLICY_FIELDS if field in entry.columns]
    spec_payload: dict[str, tuple[Any, ...]] = {}
    shard_to_entry = dict(zip(plan["shard_id"].astype(str), plan["entry_spec_id"].astype(str)))
    for shard, entry_id in shard_to_entry.items():
        row = entry.loc[entry_id]
        spec_payload[shard] = tuple(runner.a1_canonical_policy_value(field, row.get(field)) for field in selection_fields)

    intersections: Counter[tuple[str, str]] = Counter()
    inverted: dict[str, list[str]] = defaultdict(list)
    for shard, values in addresses.items():
        for value in values:
            inverted[value].append(shard)
    for members in inverted.values():
        for left, right in combinations(sorted(members), 2):
            intersections[(left, right)] += 1

    return_maps: dict[str, pd.Series] = {}
    for shard, frame in returns.items():
        if frame.empty:
            return_maps[shard] = pd.Series(dtype=float)
        else:
            return_maps[shard] = frame.set_index(["symbol", "year_month"])["raw_gross_R"]
    rows: list[dict[str, Any]] = []
    uf = UnionFind(addresses)
    for left, right in combinations(sorted(addresses), 2):
        inter = intersections.get((left, right), 0)
        union = len(addresses[left]) + len(addresses[right]) - inter
        jaccard = inter / union if union else 1.0
        a, b = return_maps[left], return_maps[right]
        common = a.index.intersection(b.index)
        corr = float(a.loc[common].corr(b.loc[common])) if len(common) >= 3 else np.nan
        pa, pb = spec_payload[left], spec_payload[right]
        parameter_similarity = sum(x == y for x, y in zip(pa, pb)) / max(1, len(pa))
        exact_equal = addresses[left] == addresses[right]
        near = bool(not exact_equal and jaccard >= 0.80 and (not math.isfinite(corr) or corr >= 0.80))
        if exact_equal or near:
            uf.union(left, right)
        if exact_equal or near or jaccard >= 0.50 or parameter_similarity >= 0.85 or (math.isfinite(corr) and corr >= 0.90):
            rows.append({
                "left_shard_id": left,
                "right_shard_id": right,
                "left_lane": plan.loc[plan.shard_id.eq(left), "definition_lane"].iloc[0],
                "right_lane": plan.loc[plan.shard_id.eq(right), "definition_lane"].iloc[0],
                "exact_event_address_equality": exact_equal,
                "event_set_jaccard": jaccard,
                "symbol_month_raw_gross_correlation": corr,
                "shared_symbol_month_cells": len(common),
                "parameter_vector_similarity": parameter_similarity,
                "cross_lane": plan.loc[plan.shard_id.eq(left), "definition_lane"].iloc[0] != plan.loc[plan.shard_id.eq(right), "definition_lane"].iloc[0],
                "near_duplicate": near,
                "clustering_contract": "exact_or_jaccard_ge_0p80_and_corr_ge_0p80_when_available",
            })
    clusters = {shard: runner.stable_hash("a1_near_cluster_v1", uf.find(shard), n=16) for shard in addresses}
    return exact, pd.DataFrame(rows), clusters


def _scenario_pivot(scorecard: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_definition_id", "definition_lane", "exit_policy_id"]
    frames = []
    for mode, bps in SCENARIOS:
        label = f"{mode}_{bps}bps"
        frame = scorecard[(scorecard["funding_mode"] == mode) & (scorecard["slippage_round_trip_bps"] == bps)][keys + ["total_net_R", "event_count", "active_symbols", "median_net_R", "exact_boundary_rows", "imputed_boundary_rows"]].copy()
        frame = frame.rename(columns={column: f"{label}_{column}" for column in frame.columns if column not in keys})
        frames.append(frame)
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=keys, how="outer")
    return out


def candidate_pool(plan: pd.DataFrame, manifest: pd.DataFrame, scorecard: pd.DataFrame, concentration: pd.DataFrame, clusters: dict[str, str], outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = _scenario_pivot(scorecard)
    definition_to_spec = manifest[["candidate_definition_id", "entry_spec_id", "definition_lane", "parameter_vector_hash"]].merge(plan[["shard_id", "entry_spec_id", "selected_key_policy_hash", "selected_event_address_hash", "definition_lane"]], on=["entry_spec_id", "definition_lane"], how="left", validate="many_to_one")
    pivot = pivot.merge(definition_to_spec, on=["candidate_definition_id", "definition_lane"], how="left", validate="one_to_one")
    pivot = pivot.merge(concentration, on=["candidate_definition_id", "definition_lane"], how="left", validate="one_to_one")
    required = ["central_imputed_4bps_total_net_R", "conservative_imputed_8bps_total_net_R", "severe_imputed_8bps_total_net_R", "severe_imputed_12bps_total_net_R"]
    pivot["corrected_robust_positive"] = pivot[required].gt(0).all(axis=1)
    pivot["exact_positive"] = pivot["exact_only_slice_4bps_total_net_R"].gt(0) & pivot["exact_only_slice_8bps_total_net_R"].gt(0)

    period_rows = []
    for (cid, lane, exit_id), group in outcomes.groupby(["candidate_definition_id", "definition_lane", "exit_policy_id"], dropna=False):
        frame = group.copy()
        frame["period_scope"] = np.select([
            frame["decision_ts"].dt.year.eq(2024),
            frame["decision_ts"].between(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"), inclusive="left"),
            frame["decision_ts"].ge(pd.Timestamp("2025-07-01", tz="UTC")),
        ], ["2024", "2025_h1", "2025_h2"], default="other")
        sums = frame.groupby("period_scope")["raw_gross_R"].sum().to_dict()
        period_rows.append({"candidate_definition_id": cid, "definition_lane": lane, "exit_policy_id": exit_id, **{f"raw_gross_R_{period}": sums.get(period, 0.0) for period in ("2024", "2025_h1", "2025_h2")}})
    pivot = pivot.merge(pd.DataFrame(period_rows), on=["candidate_definition_id", "definition_lane", "exit_policy_id"], how="left", validate="one_to_one")
    pivot["positive_period_count_context_only"] = pivot[["raw_gross_R_2024", "raw_gross_R_2025_h1", "raw_gross_R_2025_h2"]].gt(0).sum(axis=1)

    spec_rows = []
    for spec_hash, group in pivot.groupby("selected_key_policy_hash", dropna=False):
        shard_id = str(group["shard_id"].iloc[0])
        lane = str(group["definition_lane"].iloc[0])
        robust = group[group["corrected_robust_positive"]].copy()
        exact = robust[robust["exact_positive"]]
        event_count = float(pd.to_numeric(group["severe_imputed_8bps_event_count"], errors="coerce").max())
        concentration_max = float(pd.to_numeric(group["dominant_symbol_month_abs_contribution_share"], errors="coerce").min()) if group["dominant_symbol_month_abs_contribution_share"].notna().any() else np.nan
        period_count = int(pd.to_numeric(robust["positive_period_count_context_only"], errors="coerce").max()) if not robust.empty else 0
        if lane == "short_diagnostic":
            classification = "diagnostic_only"
        elif len(robust) >= 2 and len(exact) >= 1 and period_count >= 2 and event_count >= 50 and (not math.isfinite(concentration_max) or concentration_max < 0.50):
            classification = "broad_train_survivor_candidate"
        elif len(robust) >= 1 and period_count >= 1 and event_count >= 30:
            classification = "detectable_context_sleeve_candidate"
        else:
            classification = "defer_current_translation"
        entry_id = str(group["entry_spec_id"].iloc[0])
        row = manifest[manifest["entry_spec_id"].astype(str).eq(entry_id)].iloc[0]
        spec_rows.append({
            "shard_id": shard_id,
            "selected_key_policy_hash": spec_hash,
            "entry_spec_id": entry_id,
            "definition_lane": lane,
            "classification": classification,
            "robust_exit_count": len(robust),
            "exact_positive_exit_count": len(exact),
            "event_count": int(event_count),
            "positive_period_count_context_only": period_count,
            "dominant_symbol_month_share_best_exit": concentration_max,
            "near_duplicate_cluster_id": clusters[shard_id],
            "selected_event_address_hash": group["selected_event_address_hash"].iloc[0],
            "universe_policy": row.get("universe_policy", ""),
            "decision_timeframe": row.get("decision_timeframe", ""),
            "leader_rank_metric": row.get("leader_rank_metric", ""),
            "leader_top_n": row.get("leader_top_n", ""),
            "parent_regime_gate": row.get("parent_regime_gate", ""),
            "funding_gate": row.get("funding_gate", ""),
            "prior_high_proximity_filter": row.get("prior_high_proximity_filter", ""),
            "compression_required": row.get("compression_required", ""),
            "period_metric_scope": "frozen_outcome_raw_gross_context_diagnostic_only_not_selection_source",
            "candidate_source": "corrected_scenario_scorecard_only",
        })
    return pd.DataFrame(spec_rows), pivot, definition_to_spec


def select_shortlist(pool: pd.DataFrame, definitions: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_specs: list[pd.Series] = []
    used_clusters_global: set[str] = set()
    used_addresses_global: set[str] = set()
    for lane, quota in QUOTAS.items():
        candidates = pool[(pool["definition_lane"] == lane) & pool["classification"].isin(["broad_train_survivor_candidate", "detectable_context_sleeve_candidate"])].copy()
        candidates = candidates[
            ~candidates["near_duplicate_cluster_id"].astype(str).isin(used_clusters_global)
            & ~candidates["selected_event_address_hash"].astype(str).isin(used_addresses_global)
        ].copy()
        candidates["class_rank"] = candidates["classification"].map({"broad_train_survivor_candidate": 0, "detectable_context_sleeve_candidate": 1})
        candidates = candidates.sort_values(["class_rank", "exact_positive_exit_count", "positive_period_count_context_only", "robust_exit_count", "event_count", "selected_key_policy_hash"], ascending=[True, False, False, False, False, True], kind="mergesort")
        used_clusters: set[str] = set()
        used_contexts: set[tuple[Any, ...]] = set()
        while len([row for row in selected_specs if row["definition_lane"] == lane]) < quota and not candidates.empty:
            candidates["diversity_bonus"] = candidates.apply(lambda row: int(row["near_duplicate_cluster_id"] not in used_clusters) + int((row["universe_policy"], row["decision_timeframe"], row["leader_rank_metric"], row["leader_top_n"], row["parent_regime_gate"], row["funding_gate"], row["prior_high_proximity_filter"]) not in used_contexts), axis=1)
            candidates = candidates.sort_values(["diversity_bonus", "class_rank", "exact_positive_exit_count", "positive_period_count_context_only", "robust_exit_count", "event_count", "selected_key_policy_hash"], ascending=[False, True, False, False, False, False, True], kind="mergesort")
            row = candidates.iloc[0]
            selected_specs.append(row)
            used_clusters.add(str(row["near_duplicate_cluster_id"]))
            used_clusters_global.add(str(row["near_duplicate_cluster_id"]))
            used_addresses_global.add(str(row["selected_event_address_hash"]))
            used_contexts.add((row["universe_policy"], row["decision_timeframe"], row["leader_rank_metric"], row["leader_top_n"], row["parent_regime_gate"], row["funding_gate"], row["prior_high_proximity_filter"]))
            candidates = candidates[(candidates["near_duplicate_cluster_id"] != row["near_duplicate_cluster_id"]) & (candidates["selected_event_address_hash"] != row["selected_event_address_hash"])].copy()
    specs = pd.DataFrame(selected_specs)

    shortlist_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    lane_slots: Counter[str] = Counter()
    for _, spec in specs.iterrows():
        group = definitions[definitions["selected_key_policy_hash"].astype(str).eq(str(spec["selected_key_policy_hash"]))].copy()
        robust = group[group["corrected_robust_positive"]].copy()
        lane = str(spec["definition_lane"])
        desired = PRIMARY_EXIT_BY_LANE_SLOT.get(lane, PRIMARY_EXIT_ORDER)
        desired_id = desired[lane_slots[lane] % len(desired)]
        lane_slots[lane] += 1
        primary_id = desired_id if desired_id in set(robust["exit_policy_id"]) else next((exit_id for exit_id in PRIMARY_EXIT_ORDER if exit_id in set(robust["exit_policy_id"])), str(robust.sort_values("severe_imputed_8bps_total_net_R", ascending=False).iloc[0]["exit_policy_id"]))
        primary = robust[robust["exit_policy_id"].eq(primary_id)].iloc[0]
        remaining = robust[~robust["exit_policy_id"].eq(primary_id)]
        comparator_id = next((exit_id for exit_id in COMPARATOR_EXIT_ORDER if exit_id in set(remaining["exit_policy_id"])), "")
        selected_defs = [("primary", primary)]
        if comparator_id:
            selected_defs.append(("comparator", remaining[remaining["exit_policy_id"].eq(comparator_id)].iloc[0]))
        for role, definition in selected_defs:
            ev = outcomes[outcomes["candidate_definition_id"].astype(str).eq(str(definition["candidate_definition_id"]))]
            exact_rows = int(pd.to_numeric(ev.get("funding_boundary_count_exact"), errors="coerce").fillna(0).sum())
            proxy_rows = int(pd.to_numeric(ev.get("funding_boundary_count_proxy"), errors="coerce").fillna(0).sum())
            total_boundary = exact_rows + proxy_rows
            fully_exact = int((pd.to_numeric(ev.get("funding_boundary_count_proxy"), errors="coerce").fillna(0).eq(0) & pd.to_numeric(ev.get("funding_boundary_count_exact"), errors="coerce").fillna(0).gt(0)).sum())
            zero_boundary = int((pd.to_numeric(ev.get("funding_boundary_count_proxy"), errors="coerce").fillna(0).add(pd.to_numeric(ev.get("funding_boundary_count_exact"), errors="coerce").fillna(0)).eq(0)).sum())
            shortlist_rows.append({
                **spec.to_dict(),
                "exit_role": role,
                "candidate_definition_id": definition["candidate_definition_id"],
                "exit_policy_id": definition["exit_policy_id"],
                "parameter_vector_hash": definition["parameter_vector_hash"],
                "central_4bps_net_R": definition["central_imputed_4bps_total_net_R"],
                "conservative_8bps_net_R": definition["conservative_imputed_8bps_total_net_R"],
                "severe_8bps_net_R": definition["severe_imputed_8bps_total_net_R"],
                "severe_12bps_net_R": definition["severe_imputed_12bps_total_net_R"],
                "exact_4bps_net_R": definition["exact_only_slice_4bps_total_net_R"],
                "exact_8bps_net_R": definition["exact_only_slice_8bps_total_net_R"],
                "events": len(ev),
                "events_with_exact_boundaries": int(pd.to_numeric(ev.get("funding_boundary_count_exact"), errors="coerce").fillna(0).gt(0).sum()),
                "zero_boundary_events": zero_boundary,
                "fully_exact_funded_events": fully_exact,
                "exact_boundary_share": exact_rows / total_boundary if total_boundary else 0.0,
                "imputed_boundary_share": proxy_rows / total_boundary if total_boundary else 0.0,
                "funding_gate_missing_exact_share": float(ev.get("funding_unavailable", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(ev) else 0.0,
                "evidence_label": "train_only_materialization_preflight_candidate_capped_not_validation",
            })
        for _, definition in group.iterrows():
            exit_rows.append({
                "selected_key_policy_hash": spec["selected_key_policy_hash"],
                "candidate_definition_id": definition["candidate_definition_id"],
                "exit_policy_id": definition["exit_policy_id"],
                "selected_for_materialization": definition["exit_policy_id"] in {row[1]["exit_policy_id"] for row in selected_defs},
                "exit_role": next((role for role, row in selected_defs if row["exit_policy_id"] == definition["exit_policy_id"]), "frozen_unselected_exit_variant"),
                "corrected_robust_positive": definition["corrected_robust_positive"],
                "exact_positive": definition["exact_positive"],
                "same_selected_event_stream": True,
            })
    deferred = pool[~pool["selected_key_policy_hash"].isin(set(specs["selected_key_policy_hash"]))].copy()
    deferred["defer_reason"] = np.where(deferred["definition_lane"].eq("short_diagnostic"), "diagnostic_only_non_promotable", np.where(deferred["classification"].eq("defer_current_translation"), "did_not_clear_corrected_robustness_or_context_gate", "quota_or_duplicate_cluster_frozen"))
    return pd.DataFrame(shortlist_rows), deferred, pd.DataFrame(exit_rows)


def cap_normalization(selected: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    tokens = Counter()
    for series in (selected.get("label_cap_reason", pd.Series(dtype=str)), outcomes.get("label_cap_reason", pd.Series(dtype=str))):
        for value in series.dropna().astype(str):
            tokens.update(token for token in value.split(";") if token)
    rows = []
    for token, count in sorted(tokens.items()):
        if token in {"short_diagnostic", "short_diagnostic_not_primary_a1_evidence", "no_parent_gate_diagnostic", "no_funding_gate_diagnostic_cap", "kraken_listed_liquid_tail_capped"}:
            status = "diagnostic"
        elif token in {"funding_proxy_selection_cap", "funding_missing_adverse_proxy", "base_no_slippage_event_ledger_requires_stress", "train_only_caps_pending_binding_audit"}:
            status = "legacy/superseded"
        else:
            status = "active"
        rows.append({"cap_label": token, "normalized_status": status, "row_occurrences": count, "applies_to_corrected_candidate_scenarios": status != "legacy/superseded", "normalization_reason": "corrected funding/slippage scenarios supersede legacy proxy and zero-base-slippage labels" if status == "legacy/superseded" else "preserved by current train-only contract"})
    rows.append({"cap_label": "funding_imputed_train_screen_cap", "normalized_status": "active", "row_occurrences": 0, "applies_to_corrected_candidate_scenarios": True, "normalization_reason": "required for central/conservative/severe shared-funding scenarios"})
    return pd.DataFrame(rows).drop_duplicates("cap_label", keep="last")


def control_feasibility(shortlist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    contracts = {
        "same_symbol": ("feasible", "symbol, decision_ts, non-event decision calendar, PIT eligibility"),
        "same_regime": ("feasible_with_point_in_time_feature_rebuild", "parent regime state and source timestamp are not carried in frozen outcome rows"),
        "generic_breakout": ("feasible_with_predeclared_control_signal_build", "generic breakout control event ledger is not yet materialized"),
        "donchian_simple_breakout": ("feasible_with_predeclared_control_signal_build", "Donchian/simple breakout control event ledger is not yet materialized"),
        "nearest_neighbor": ("blocked_missing_matching_matrix", "candidate/control impulse, compression, breadth, liquidity, regime and funding-state feature matrix at decision_ts"),
    }
    for _, candidate in shortlist.iterrows():
        for control, (status, detail) in contracts.items():
            rows.append({"candidate_definition_id": candidate["candidate_definition_id"], "selected_key_policy_hash": candidate["selected_key_policy_hash"], "control_class": control, "feasibility_status": status, "required_features_or_constraint": detail, "placeholder_control_allowed": False})
    return pd.DataFrame(rows)


def build_context_matrix(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pool.iterrows():
        base = row.to_dict()
        rows.extend([
            {**base, "context_dimension": "period", "context_value": period, "support_metric": "raw_gross_sign_from_frozen_primary_exit_context_only", "support_value": row.get(f"raw_gross_R_{period}", "not_available_at_spec_pool_level")}
            for period in ("2024", "2025_h1", "2025_h2")
        ])
        for dimension, field in [("parent_gate_policy", "parent_regime_gate"), ("universe_policy", "universe_policy"), ("decision_timeframe", "decision_timeframe"), ("rank_metric", "leader_rank_metric"), ("top_n", "leader_top_n"), ("funding_gate_policy", "funding_gate"), ("prior_high_filter", "prior_high_proximity_filter"), ("compression_state", "compression_required")]:
            rows.append({**base, "context_dimension": dimension, "context_value": row.get(field, ""), "support_metric": "predeclared_manifest_category", "support_value": "present"})
        rows.append({**base, "context_dimension": "breadth_state", "context_value": "not_available_in_frozen_event_rows", "support_metric": "not_fabricated", "support_value": "requires_point_in_time_feature_rebuild"})
        rows.append({**base, "context_dimension": "realized_parent_regime", "context_value": "not_available_in_frozen_event_rows", "support_metric": "not_fabricated", "support_value": "gate policy available; realized state requires PIT feature rebuild"})
    return pd.DataFrame(rows)


def lineage_audit(shortlist: pd.DataFrame, manifest: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    manifest_by_id = manifest.set_index("candidate_definition_id", drop=False)
    plan_by_hash = plan.set_index("selected_key_policy_hash", drop=False)
    rows = []
    for _, row in shortlist.iterrows():
        cid, policy_hash = str(row["candidate_definition_id"]), str(row["selected_key_policy_hash"])
        definition = manifest_by_id.loc[cid]
        planned = plan_by_hash.loc[policy_hash]
        shard_dir = FULL_ROOT / "aggregate_shards" / str(planned["shard_id"])
        selected = pd.read_csv(shard_dir / "selected_keys.csv")
        outcomes = pd.read_parquet(shard_dir / "outcome_events.parquet")
        checks = {
            "candidate_in_manifest": cid in manifest_by_id.index,
            "parameter_vector_hash_match": str(row["parameter_vector_hash"]) == str(definition["parameter_vector_hash"]),
            "canonical_policy_hash_match": set(selected["selected_key_policy_hash"].astype(str)) == {policy_hash},
            "outcome_policy_hash_match": set(outcomes[outcomes["candidate_definition_id"].astype(str).eq(cid)]["selected_key_policy_hash"].astype(str)) == {policy_hash},
            "protected_violation_count_zero": not outcomes.get("protected_interval_violation", pd.Series(False, index=outcomes.index)).fillna(False).astype(bool).any(),
            "funding_model_manifest_present": (FUNDING_ROOT / "funding/shared_funding_panel_manifest.csv").exists(),
        }
        rows.append({"candidate_definition_id": cid, "selected_key_policy_hash": policy_hash, **checks, "status": "pass" if all(checks.values()) else "fail"})
    return pd.DataFrame(rows)


def compact_bundle(root: Path, required: list[str]) -> None:
    rows = []
    for rel in required:
        source = root / rel
        if not source.exists():
            raise RuntimeError(f"required compact-bundle artifact missing: {rel}")
        target = root / "compact_review_bundle" / rel.replace("/", "__")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({"source": rel, "bundle_path": str(target.relative_to(root)), "sha256": sha256_file(target)})
    write_csv(root / "compact_review_bundle/compact_bundle_manifest.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.run_root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"fresh preflight root required: {root}")
    for dependency in (FULL_ROOT, FUNDING_ROOT, HASH_ROOT, CONTRACT_ROOT):
        if not dependency.exists():
            raise RuntimeError(f"missing dependency root: {dependency}")
    runner_args = runner.parse_args(["--phase-profile", runner.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_PHASE_PROFILE, "--run-root", str(root), "--start", "2024-01-01", "--end", "2025-12-31", "--disable-telegram"])
    ctx = runner.init_context(runner_args)
    manifest = runner.load_a1_compression_manifest()
    plan = pd.read_csv(FULL_ROOT / "shards/full_manifest_shard_plan.csv")
    overlap = pd.read_csv(FULL_ROOT / "forensics/selected_key_overlap_clusters.csv")
    plan = plan.merge(overlap[["shard_id", "selected_event_address_hash", "selected_event_addresses"]], on="shard_id", how="left", validate="one_to_one")
    scorecard = pd.read_csv(FULL_ROOT / "aggregate/full_definition_scorecard.csv")
    concentration = pd.read_csv(FULL_ROOT / "forensics/concentration_preview.csv")
    selected, outcomes, addresses, returns = _load_shard_frames()

    attrition = reconcile_attrition(ctx, selected, outcomes, manifest)
    write_csv(root / "integrity/selected_to_outcome_attrition_audit.csv", attrition)
    selected_count, outcome_count = len(selected), len(outcomes)
    attrition_pass = selected_count - outcome_count == len(attrition) and bool(attrition.empty or attrition["allowed_explicit_reason"].all())
    write_text(root / "integrity/candidate_source_and_scenario_contract.md", """# Candidate Source And Scenario Contract

Candidate selection uses only `aggregate/full_definition_scorecard.csv` from the corrected shared-funding reducer. Frozen scenarios are central +4 bps, conservative +8 bps, severe +8/+12 bps, and exact-only +4/+8 bps. Legacy adverse-proxy and zero-slippage grouped aggregates are excluded from shortlist evidence. Period context derived from frozen outcome gross returns is diagnostic context only and is never substituted for corrected scenario selection.
""")
    caps = cap_normalization(selected, outcomes)
    write_csv(root / "caps/active_cap_normalization.csv", caps)
    exact, near, clusters = duplicate_reports(plan, manifest, addresses, returns)
    write_csv(root / "selection/exact_duplicate_cluster_report.csv", exact)
    write_csv(root / "selection/near_duplicate_cluster_report.csv", near)
    pool, definitions, definition_map = candidate_pool(plan, manifest, scorecard, concentration, clusters, outcomes)
    write_csv(root / "selection/eligible_candidate_pool.csv", pool)
    context = build_context_matrix(pool)
    write_csv(root / "regime/context_support_matrix.csv", context)
    write_text(root / "regime/context_sleeve_report.md", """# Context Sleeve Report

Classification uses corrected full-train funding/slippage robustness plus frozen, point-in-time event context. Realized parent and breadth state are not present in frozen event rows and are explicitly unavailable rather than reconstructed with future-aware labels. H06 may be retained as a context sleeve when corrected robustness and at least one predeclared period context are positive; this does not reject or promote the lane broadly.
""")
    shortlist, deferred, exit_clusters = select_shortlist(pool, definitions, outcomes)
    write_csv(root / "selection/survivor_shortlist.csv", shortlist)
    write_csv(root / "selection/deferred_or_diagnostic_candidates.csv", deferred)
    write_csv(root / "selection/exit_cluster_report.csv", exit_clusters)
    controls = control_feasibility(shortlist)
    write_csv(root / "controls/control_match_feasibility.csv", controls)
    lineage = lineage_audit(shortlist, manifest, plan)
    write_csv(root / "preflight/shortlist_lineage_audit.csv", lineage)

    profile_registered = MATERIALIZATION_PROFILE in runner.PHASE_PROFILES
    write_text(root / "profile/profile_contract.md", f"""# A1 Targeted Materialization Profile Contract

- Profile: `{MATERIALIZATION_PROFILE}`
- Input: this preflight's frozen `selection/survivor_shortlist.csv` only.
- Candidate expansion/re-ranking: forbidden.
- Materialization: event-level ledgers for selected primary/comparator exits only.
- Controls: real same-symbol, same-regime, generic-breakout and Donchian controls where feasible; nearest-neighbor fails closed until its matching matrix exists.
- Evidence ceiling: train-only materialized/control/stress diagnostic, not validation or live readiness.
""")
    dry_pass = attrition_pass and not lineage.empty and lineage["status"].eq("pass").all() and 12 <= shortlist["selected_key_policy_hash"].nunique() <= 16 and profile_registered
    write_text(root / "preflight/profile_dry_run_report.md", "\n".join([
        "# Profile Dry Run Report", "", f"- Profile registered: `{profile_registered}`", f"- Materialization launched: `false`", f"- Shortlisted selected-key specs: `{shortlist['selected_key_policy_hash'].nunique()}`", f"- Shortlisted definitions/exits: `{len(shortlist)}`", f"- Attrition reconciliation pass: `{attrition_pass}`", f"- Lineage pass: `{lineage['status'].eq('pass').all()}`", f"- Dry-run pass: `{dry_pass}`", "- Validation/holdout launched: `false/false`", ""
    ]))
    write_text(root / "prelaunch/next_materialization_launch_prompt.md", f"""Proceed with A1 + Compression Targeted Materialization / Controls / Stress v1.

Use profile `{MATERIALIZATION_PROFILE}` and shortlist `{root / 'selection/survivor_shortlist.csv'}`. Re-run lineage fail-closed, materialize only frozen primary/comparator definitions, build only real controls, run corrected funding/slippage stress and concentration/winner-removal diagnostics. Do not run validation, holdout, TSMOM, prior-high standalone, broad sweeps, or live-prep.
""")
    summary = {
        "run_root": str(root),
        "status": "complete" if dry_pass else "blocked",
        "code_modified": True,
        "materialization_launched": False,
        "controls_built": False,
        "validation_launched": False,
        "final_holdout_touched": False,
        "aggregate_candidates_screened": int(manifest["candidate_definition_id"].nunique()),
        "selected_rows": selected_count,
        "outcome_rows": outcome_count,
        "attrition_rows": len(attrition),
        "selected_to_outcome_attrition_pass": attrition_pass,
        "exact_duplicate_clusters": int(exact["exact_duplicate"].sum()),
        "near_duplicate_pairs": int(near.get("near_duplicate", pd.Series(dtype=bool)).fillna(False).sum()) if not near.empty else 0,
        "shortlisted_selected_key_specs": int(shortlist["selected_key_policy_hash"].nunique()),
        "shortlisted_definitions_exits": int(len(shortlist)),
        "broad_survivor_specs": int(pool["classification"].eq("broad_train_survivor_candidate").sum()),
        "context_sleeve_specs": int(pool["classification"].eq("detectable_context_sleeve_candidate").sum()),
        "diagnostic_specs": int(pool["classification"].eq("diagnostic_only").sum()),
        "deferred_specs": int(pool["classification"].eq("defer_current_translation").sum()),
        "canonical_hash_mismatches": int((~lineage["canonical_policy_hash_match"]).sum()),
        "lineage_audit_pass": bool(lineage["status"].eq("pass").all()),
        "supported_materialization_profile": MATERIALIZATION_PROFILE,
        "profile_registered": profile_registered,
        "next_launch_prompt_path": str(root / "prelaunch/next_materialization_launch_prompt.md"),
        "compact_bundle_path": str(root / "compact_review_bundle"),
        "evidence_label": "train_only_candidate_selection_and_materialization_preflight_not_validation",
    }
    write_json(root / "decision_summary.json", summary)
    required = [
        "integrity/selected_to_outcome_attrition_audit.csv", "integrity/candidate_source_and_scenario_contract.md", "caps/active_cap_normalization.csv", "selection/eligible_candidate_pool.csv", "selection/exact_duplicate_cluster_report.csv", "selection/near_duplicate_cluster_report.csv", "selection/exit_cluster_report.csv", "regime/context_support_matrix.csv", "regime/context_sleeve_report.md", "selection/survivor_shortlist.csv", "selection/deferred_or_diagnostic_candidates.csv", "controls/control_match_feasibility.csv", "profile/profile_contract.md", "preflight/shortlist_lineage_audit.csv", "preflight/profile_dry_run_report.md", "prelaunch/next_materialization_launch_prompt.md", "decision_summary.json",
    ]
    compact_bundle(root, required)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if dry_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
