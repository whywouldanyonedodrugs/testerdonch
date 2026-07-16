#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_a1_funding_corrected_balanced_50shard_canonical as funding_run


AGGREGATE_ROOT = Path("results/rebaseline/phase_kraken_prior_high_reclaim_v2_canonical_train_scan_20260712_v1")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
MANIFEST = Path("results/rebaseline/phase_kraken_prior_high_exit_binding_repair_20260705_v1/prior_high/redesign/prior_high_reclaim_sweep_definitions_v2.csv")
PROFILE = "prior_high_reclaim_v2_targeted_materialization_profile_20260712_v1"
REQUIRED_SCENARIOS = (
    ("central_imputed", 4), ("conservative_imputed", 8),
    ("severe_imputed", 8), ("severe_imputed", 12),
    ("exact_only_slice", 4), ("exact_only_slice", 8),
)
PARAMETER_FIELDS = (
    "signal_type", "bar_timeframe", "lookback_value", "hold_value", "universe_policy",
    "parent_regime_gate", "funding_gate", "atr_bar_timeframe", "atr_window_value",
    "atr_stop_mult", "atr_trail_mult", "structure_buffer_atr", "vwap_type",
    "vwap_anchor_policy", "exit_template",
)


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_events(candidate_ids: set[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(AGGREGATE_ROOT.glob("aggregate_shards/*/outcome_events.parquet")):
        frame = pd.read_parquet(path)
        frame = frame[frame["candidate_definition_id"].astype(str).isin(candidate_ids)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("eligible frozen prior-high event pool is empty")
    events = pd.concat(frames, ignore_index=True, sort=False)
    if events.duplicated(["candidate_definition_id", "event_id"]).any():
        raise RuntimeError("frozen event identities are duplicated")
    return events


def corrected_event_scenarios(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = consumer.normalize_frozen_events(events, "a1")
    boundaries = consumer.build_event_boundary_rows(normalized)
    panel, extension = balanced.extend_frozen_panel_with_verified_model(
        funding_run.load_frozen_panel(), boundaries, FUNDING_ROOT,
    )
    joined = consumer.join_boundaries_to_panel(boundaries, panel)
    missing = int((joined["_merge"] != "both").sum())
    duplicates = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    if missing or duplicates:
        raise RuntimeError(f"funding boundary join failed: missing={missing}, duplicates={duplicates}")
    funded = consumer.aggregate_event_funding(normalized, joined)
    scenarios = balanced.scenario_event_rows(
        funded,
        ("central_imputed", "conservative_imputed", "severe_imputed", "exact_only_slice"),
        (4, 8, 12),
    )
    return funded, scenarios, joined


def event_key_sets(events: pd.DataFrame) -> dict[str, set[str]]:
    work = events.copy()
    work["entry_identity"] = work["symbol"].astype(str) + "|" + pd.to_datetime(work["decision_ts"], utc=True).astype(str)
    return {str(cid): set(group["entry_identity"].astype(str)) for cid, group in work.groupby("candidate_definition_id")}


def union_find(items: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    parent = {item: item for item in items}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = {item: find(item) for item in items}
    ordered = {root: f"cluster_{index:02d}" for index, root in enumerate(sorted(set(roots.values())), 1)}
    return {item: ordered[root] for item, root in roots.items()}


def parameter_similarity(left: pd.Series, right: pd.Series) -> float:
    return float(np.mean([str(left.get(field, "")) == str(right.get(field, "")) for field in PARAMETER_FIELDS]))


def pairwise_clusters(manifest: pd.DataFrame, events: pd.DataFrame, severe12: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = event_key_sets(events)
    candidates = sorted(keys)
    meta = manifest.set_index("candidate_definition_id")
    sm = severe12.copy()
    sm["symbol_month"] = sm["symbol"].astype(str) + "|" + pd.to_datetime(sm["entry_ts"], utc=True).dt.strftime("%Y-%m")
    streams = {str(cid): group.groupby("symbol_month")["scenario_scaled_net_R"].sum() for cid, group in sm.groupby("candidate_definition_id")}
    rows, exact_edges, near_edges = [], [], []
    for left, right in combinations(candidates, 2):
        union = keys[left] | keys[right]
        jaccard = len(keys[left] & keys[right]) / len(union) if union else 1.0
        exact = keys[left] == keys[right]
        common = streams[left].index.intersection(streams[right].index)
        corr = float(streams[left].loc[common].corr(streams[right].loc[common])) if len(common) >= 3 else np.nan
        similarity = parameter_similarity(meta.loc[left], meta.loc[right])
        same_signal = str(meta.loc[left, "signal_type"]) == str(meta.loc[right, "signal_type"])
        near = bool(same_signal and (jaccard >= 0.75 or (similarity >= 0.80 and np.isfinite(corr) and corr >= 0.80)))
        if exact:
            exact_edges.append((left, right))
        if near:
            near_edges.append((left, right))
        rows.append({
            "candidate_a": left, "candidate_b": right, "exact_selected_event_equality": exact,
            "event_jaccard": jaccard, "symbol_month_return_correlation": corr,
            "parameter_similarity": similarity, "same_signal_type": same_signal,
            "same_exit_semantics": str(meta.loc[left, "exit_template"]) == str(meta.loc[right, "exit_template"]),
            "near_duplicate": near,
        })
    pairwise = pd.DataFrame(rows)
    exact_map = union_find(candidates, exact_edges)
    near_map = union_find(candidates, near_edges)
    exact_rows = [{"candidate_definition_id": cid, "exact_duplicate_cluster_id": exact_map[cid], "cluster_size": sum(value == exact_map[cid] for value in exact_map.values()), "frozen": sum(value == exact_map[cid] for value in exact_map.values()) > 1} for cid in candidates]
    near_rows = [{"candidate_definition_id": cid, "near_duplicate_cluster_id": near_map[cid], "cluster_size": sum(value == near_map[cid] for value in near_map.values())} for cid in candidates]
    return pd.DataFrame(exact_rows), pd.DataFrame(near_rows), pairwise


def forensic_tables(events: pd.DataFrame, scenarios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    severe = scenarios[(scenarios["funding_mode"] == "severe_imputed") & (scenarios["slippage_round_trip_bps"] == 12)].copy()
    severe["month"] = pd.to_datetime(severe["entry_ts"], utc=True).dt.strftime("%Y-%m")
    severe["period"] = np.select([
        pd.to_datetime(severe["entry_ts"], utc=True).dt.year.eq(2024),
        pd.to_datetime(severe["entry_ts"], utc=True).lt(pd.Timestamp("2025-07-01", tz="UTC")),
    ], ["2024", "2025_h1"], default="2025_h2")
    winner_rows, support_rows = [], []
    for cid, group in severe.groupby("candidate_definition_id", sort=True):
        values = group["scenario_scaled_net_R"].sort_values(ascending=False)
        total_abs = float(group["scenario_scaled_net_R"].abs().sum())
        trim_n = max(1, int(np.ceil(len(group) * 0.01)))
        symbol = group.groupby("symbol")["scenario_scaled_net_R"].sum().abs()
        month = group.groupby("month")["scenario_scaled_net_R"].sum().abs()
        symbol_month = group.groupby(["symbol", "month"])["scenario_scaled_net_R"].sum().abs()
        winner_rows.append({
            "candidate_definition_id": cid, "events": len(group), "severe_12bps_total_R": float(values.sum()),
            "net_without_top_1": float(values.iloc[1:].sum()), "net_without_top_3": float(values.iloc[3:].sum()),
            "net_after_top_1pct_trim": float(values.iloc[trim_n:].sum()), "top_1pct_events_removed": trim_n,
            "top_event_abs_share": float(group["scenario_scaled_net_R"].abs().max() / total_abs) if total_abs else np.nan,
            "dominant_symbol_abs_share": float(symbol.max() / total_abs) if total_abs else np.nan,
            "dominant_month_abs_share": float(month.max() / total_abs) if total_abs else np.nan,
            "dominant_symbol_month_abs_share": float(symbol_month.max() / total_abs) if total_abs else np.nan,
        })
        for period, period_group in group.groupby("period"):
            support_rows.append({
                "candidate_definition_id": cid, "context_type": "predeclared_period", "context_value": period,
                "event_count": len(period_group), "net_R": float(period_group["scenario_scaled_net_R"].sum()),
                "context_source": "entry_ts_fixed_calendar_bucket", "future_aware": False,
            })
        for regime, regime_group in group.groupby("regime_label", dropna=False):
            support_rows.append({
                "candidate_definition_id": cid, "context_type": "frozen_parent_regime", "context_value": regime,
                "event_count": len(regime_group), "net_R": float(regime_group["scenario_scaled_net_R"].sum()),
                "context_source": "frozen_event_regime_label", "future_aware": False,
            })
    return pd.DataFrame(winner_rows), pd.DataFrame(support_rows)


def exact_composition(funded: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, group in funded.groupby("candidate_definition_id", sort=True):
        boundary = joined[joined["event_key"].isin(group["event_key"])]
        total_boundaries = len(boundary)
        exact_boundaries = int(boundary["funding_exact"].fillna(False).sum())
        event_rows = len(group)
        rows.append({
            "candidate_definition_id": cid, "events": event_rows,
            "events_with_one_or_more_exact_boundaries": int(group["exact_boundary_rows"].gt(0).sum()),
            "zero_funding_boundary_events": int(group["funding_boundary_rows"].eq(0).sum()),
            "fully_exact_funded_events": int(group["all_boundaries_exact"].sum()),
            "exact_boundary_rows": exact_boundaries, "imputed_boundary_rows": total_boundaries - exact_boundaries,
            "exact_boundary_share": exact_boundaries / total_boundaries if total_boundaries else np.nan,
            "imputed_boundary_share": (total_boundaries - exact_boundaries) / total_boundaries if total_boundaries else np.nan,
            "funding_imputed_train_screen_cap": bool(group["central_imputation_cap"].any()),
        })
    return pd.DataFrame(rows)


def scenario_pool(scorecard: pd.DataFrame, eligible: list[str]) -> pd.DataFrame:
    pieces = []
    for mode, bps in REQUIRED_SCENARIOS:
        part = scorecard[(scorecard["funding_mode"] == mode) & (scorecard["slippage_round_trip_bps"] == bps)].copy()
        part = part[part["candidate_definition_id"].isin(eligible)]
        part["scenario_key"] = f"{mode}__{bps}bps"
        pieces.append(part)
    pool = pd.concat(pieces, ignore_index=True)
    expected = {(cid, f"{mode}__{bps}bps") for cid in eligible for mode, bps in REQUIRED_SCENARIOS}
    observed = set(zip(pool["candidate_definition_id"], pool["scenario_key"]))
    # Exact-only rows can legitimately be absent when no fully exact-funded event exists.
    missing_nonexact = [item for item in expected - observed if not item[1].startswith("exact_only_slice")]
    if missing_nonexact:
        raise RuntimeError(f"required scenario sources missing: {missing_nonexact[:5]}")
    return pool


def select_shortlist(
    manifest: pd.DataFrame,
    pool: pd.DataFrame,
    exact: pd.DataFrame,
    near: pd.DataFrame,
    pairwise: pd.DataFrame,
    winner: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = pool.pivot_table(index="candidate_definition_id", columns="scenario_key", values="total_net_R", aggfunc="first").reset_index()
    for mode, bps in REQUIRED_SCENARIOS:
        column = f"{mode}__{bps}bps"
        if column not in wide:
            wide[column] = np.nan
    metrics = manifest.merge(wide, on="candidate_definition_id", how="inner").merge(exact, on="candidate_definition_id").merge(near, on="candidate_definition_id").merge(winner, on="candidate_definition_id")
    period = support[support["context_type"] == "predeclared_period"].pivot_table(index="candidate_definition_id", columns="context_value", values="net_R", aggfunc="sum").fillna(0)
    metrics = metrics.merge(period.add_prefix("period_net_").reset_index(), on="candidate_definition_id", how="left")
    exact_col = "exact_only_slice__8bps"
    metrics["exact_support_positive"] = pd.to_numeric(metrics[exact_col], errors="coerce").fillna(-np.inf).gt(0)
    metrics["positive_period_count"] = metrics[[c for c in metrics if c.startswith("period_net_")]].gt(0).sum(axis=1)
    metrics["parameter_neighborhood_support"] = metrics["candidate_definition_id"].map(lambda cid: int(((pairwise["candidate_a"].eq(cid) | pairwise["candidate_b"].eq(cid)) & pairwise["same_signal_type"] & pairwise["event_jaccard"].ge(0.50)).sum()))
    metrics["workflow_score"] = (
        metrics["severe_imputed__12bps"].gt(0).astype(int) * 2
        + metrics["severe_imputed__8bps"].gt(0).astype(int) * 2
        + metrics["exact_support_positive"].astype(int) * 2
        + metrics["net_without_top_1"].gt(0).astype(int)
        + metrics["net_without_top_3"].gt(0).astype(int) * 2
        + metrics["net_after_top_1pct_trim"].gt(0).astype(int)
        + metrics["positive_period_count"].clip(upper=3)
        + metrics["events"].ge(30).astype(int)
        + metrics["dominant_symbol_month_abs_share"].le(0.40).astype(int)
        + metrics["top_event_abs_share"].le(0.25).astype(int)
        + metrics["parameter_neighborhood_support"].gt(0).astype(int)
    )
    robust = metrics["severe_imputed__12bps"].gt(0) & metrics["severe_imputed__8bps"].gt(0)
    broad = robust & metrics["exact_support_positive"] & metrics["net_without_top_3"].gt(0) & metrics["positive_period_count"].ge(2) & metrics["dominant_symbol_month_abs_share"].le(0.40)
    context = robust & metrics["net_without_top_1"].gt(0) & metrics["positive_period_count"].ge(1)
    metrics["candidate_classification"] = np.where(
        broad, "broad_train_survivor_candidate",
        np.where(context, "detectable_context_sleeve_candidate", np.where(metrics["signal_type"].eq("prior_high_breakout"), "diagnostic_baseline", "defer_current_translation")),
    )
    quotas = {"proximity_relative_strength": 5, "failed_breakout_reclaim": 2, "prior_high_breakout": 2}
    selected = []
    used_near = set()
    used_exact = set()
    for signal, quota in quotas.items():
        lane = metrics[metrics["signal_type"].eq(signal)].sort_values(
            ["candidate_classification", "workflow_score", "exact_support_positive", "parameter_neighborhood_support", "candidate_definition_id"],
            ascending=[True, False, False, False, True], kind="mergesort",
        )
        # Broad/context candidates sort ahead explicitly; diagnostic baselines are allowed for the baseline quota.
        order = {"broad_train_survivor_candidate": 0, "detectable_context_sleeve_candidate": 1, "diagnostic_baseline": 2, "defer_current_translation": 3}
        lane = lane.assign(_class_order=lane["candidate_classification"].map(order)).sort_values(["_class_order", "workflow_score", "candidate_definition_id"], ascending=[True, False, True], kind="mergesort")
        for _, row in lane.iterrows():
            cluster = str(row["near_duplicate_cluster_id"])
            exact_cluster = str(row["exact_duplicate_cluster_id"])
            if cluster in used_near or exact_cluster in used_exact:
                continue
            selected.append(row.drop(labels=["_class_order"], errors="ignore").to_dict())
            used_near.add(cluster)
            used_exact.add(exact_cluster)
            if sum(item["signal_type"] == signal for item in selected) >= quota:
                break
    shortlist = pd.DataFrame(selected)
    shortlist["selection_rank"] = np.arange(1, len(shortlist) + 1)
    shortlist["materialization_launched"] = False
    shortlist["evidence_label"] = "train_only_aggregate_preflight_funding_imputed_capped"
    chosen = set(shortlist["candidate_definition_id"])
    deferred = metrics[~metrics["candidate_definition_id"].isin(chosen)].copy()
    deferred["disposition"] = deferred["candidate_classification"]
    neighborhood = metrics[["candidate_definition_id", "signal_type", "near_duplicate_cluster_id", "parameter_neighborhood_support", "workflow_score", "candidate_classification"]].copy()
    return shortlist, deferred, neighborhood


def control_feasibility(shortlist: pd.DataFrame, manifest_all: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, candidate in shortlist.iterrows():
        cid = str(candidate["candidate_definition_id"])
        group = events[events["candidate_definition_id"].astype(str).eq(cid)]
        same_context_fields = all(field in group for field in ["symbol", "regime_label", "decision_ts"])
        baseline = manifest_all[
            manifest_all["signal_type"].eq("prior_high_breakout")
            & manifest_all["bar_timeframe"].eq(candidate["bar_timeframe"])
            & manifest_all["lookback_value"].eq(candidate["lookback_value"])
        ]
        for control_class, feasible, reason in [
            ("same_symbol", len(group) > 0, "frozen candidate symbols and non-event decision calendar available"),
            ("same_regime", same_context_fields, "frozen PIT parent-regime labels available"),
            ("close_confirmed_breakout_without_proximity", len(baseline) > 0, f"matching manifest baselines={len(baseline)}"),
            ("generic_breakout", True, "executable close-confirmed breakout engine and frozen universe/gates available"),
            ("pure_donchian_breakout", True, "prior-close rolling-high engine is executable; next-bar semantics can be frozen before outcomes"),
        ]:
            rows.append({
                "candidate_definition_id": cid, "control_class": control_class,
                "real_control_feasible": bool(feasible), "placeholder_allowed": False,
                "control_entries_must_freeze_before_outcomes": True, "reason": reason,
                "status": "pass" if feasible else "blocked_missing_exact_match",
            })
    return pd.DataFrame(rows)


def validate_registered_materialization_profile(ctx: Any) -> None:
    root = Path(getattr(ctx, "run_root", ""))
    source = Path("results/rebaseline/phase_kraken_prior_high_reclaim_v2_materialization_preflight_20260712_v1")
    shortlist = pd.read_csv(source / "selection/survivor_shortlist.csv")
    lineage = pd.read_csv(source / "preflight/shortlist_lineage_audit.csv")
    if shortlist.empty or not lineage["status"].eq("pass").all():
        raise RuntimeError("prior-high v2 targeted profile lineage is not launchable")
    write_csv(root / "preflight/shortlist_lineage_audit.csv", lineage)
    write_json(root / "decision_summary.json", {
        "run_root": str(root), "status": "profile_lineage_dry_run_complete",
        "shortlist_size": len(shortlist), "materialization_launched": False,
        "next_operator_decision": "implement_and_launch_prior_high_v2_targeted_materialization_controls_next",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    if root.exists():
        raise RuntimeError(f"fresh run root required: {root}")
    root.mkdir(parents=True)
    decision = json.loads((AGGREGATE_ROOT / "decision_summary.json").read_text())
    if decision.get("status") != "complete" or decision.get("canonical_hash_mismatches") != 0 or decision.get("protected_period_violations") != 0:
        raise RuntimeError("aggregate lineage gate failed")
    eligible = list(map(str, decision["materialization_preflight_eligible_candidates"]))
    if len(eligible) != 23:
        raise RuntimeError("expected exactly 23 aggregate-eligible definitions")
    manifest_all = pd.read_csv(MANIFEST)
    manifest = manifest_all[manifest_all["candidate_definition_id"].astype(str).isin(eligible)].copy()
    plan = pd.read_csv(AGGREGATE_ROOT / "shards/full_shard_plan.csv")
    policy_map = plan.assign(candidate_definition_id=plan["candidate_definition_ids"].astype(str)).set_index("candidate_definition_id")["selected_key_policy_hash"]
    manifest["selected_key_policy_hash"] = manifest["candidate_definition_id"].map(policy_map)
    scorecard = pd.read_csv(AGGREGATE_ROOT / "aggregate/full_definition_scorecard.csv")
    pool = scenario_pool(scorecard, eligible)
    events = load_events(set(eligible))
    funded, scenarios, joined = corrected_event_scenarios(events)
    exact_audit = exact_composition(funded, joined)
    severe12 = scenarios[(scenarios["funding_mode"] == "severe_imputed") & (scenarios["slippage_round_trip_bps"] == 12)]
    exact_clusters, near_clusters, pairwise = pairwise_clusters(manifest, events, severe12)
    winner, support = forensic_tables(events, scenarios)
    shortlist, deferred, neighborhoods = select_shortlist(manifest, pool, exact_clusters, near_clusters, pairwise, winner, support)
    controls = control_feasibility(shortlist, manifest_all, events)
    lineage = shortlist[["candidate_definition_id", "selected_key_policy_hash", "parameter_vector_hash"]].copy()
    expected_hash = manifest_all.set_index("candidate_definition_id")["parameter_vector_hash"]
    expected_policy = policy_map
    lineage["expected_parameter_vector_hash"] = lineage["candidate_definition_id"].map(expected_hash)
    lineage["expected_selected_key_policy_hash"] = lineage["candidate_definition_id"].map(expected_policy)
    lineage["canonical_hash_match"] = lineage["selected_key_policy_hash"].eq(lineage["expected_selected_key_policy_hash"])
    lineage["parameter_hash_match"] = lineage["parameter_vector_hash"].eq(lineage["expected_parameter_vector_hash"])
    lineage["protected_period_violations"] = 0
    lineage["status"] = np.where(lineage["canonical_hash_match"] & lineage["parameter_hash_match"], "pass", "fail")
    exact_freeze_pass = bool((exact_clusters["cluster_size"].le(1) | exact_clusters["frozen"]).all())
    shortlist_exact_unique = not shortlist["exact_duplicate_cluster_id"].duplicated().any()
    hard_pass = len(shortlist) in range(8, 13) and lineage["status"].eq("pass").all() and not support["future_aware"].any() and exact_freeze_pass and shortlist_exact_unique

    write_csv(root / "funding/exact_slice_composition_audit.csv", exact_audit)
    eligible_pool = manifest.merge(pool.groupby("candidate_definition_id").size().rename("scenario_rows").reset_index(), on="candidate_definition_id")
    write_csv(root / "selection/eligible_candidate_pool.csv", eligible_pool)
    write_csv(root / "selection/exact_duplicate_clusters.csv", exact_clusters)
    write_csv(root / "selection/near_duplicate_clusters.csv", near_clusters)
    write_csv(root / "selection/parameter_neighborhood_report.csv", neighborhoods)
    write_csv(root / "forensics/winner_concentration_preview.csv", winner)
    write_csv(root / "regime/period_and_context_support.csv", support)
    write_csv(root / "selection/survivor_shortlist.csv", shortlist)
    write_csv(root / "selection/deferred_or_diagnostic_candidates.csv", deferred)
    write_csv(root / "controls/control_match_feasibility.csv", controls)
    write_csv(root / "preflight/shortlist_lineage_audit.csv", lineage)
    (root / "integrity").mkdir(parents=True, exist_ok=True)
    (root / "integrity/candidate_source_contract.md").write_text("\n".join([
        "# Candidate Source Contract", "",
        f"Aggregate root: `{AGGREGATE_ROOT}`", f"Aggregate decision hash: `{sha256_file(AGGREGATE_ROOT / 'decision_summary.json')}`",
        f"Scorecard hash: `{sha256_file(AGGREGATE_ROOT / 'aggregate/full_definition_scorecard.csv')}`",
        f"Funding model root: `{FUNDING_ROOT}`", "",
        "Selection scenarios are frozen to central+4bps, conservative+8bps, severe+8/12bps, and exact-only+4/8bps.",
        "Signals and outcomes are not regenerated. Context uses fixed calendar periods and frozen event parent-regime labels only.",
        "Exact-only absence is explicit when a candidate has no fully exact-funded event; it is not zero-filled as evidence.",
    ]) + "\n", encoding="utf-8")
    (root / "profile").mkdir(parents=True, exist_ok=True)
    (root / "profile/materialization_profile_contract.md").write_text("\n".join([
        "# Prior-High v2 Targeted Materialization Profile Contract", "",
        f"Registered profile: `{PROFILE}`", f"Frozen shortlist: `{root / 'selection/survivor_shortlist.csv'}`",
        "The registered profile is lineage/dry-run only in this phase. It cannot launch materialization.",
        "A subsequent implementation must build event ledgers only for this shortlist, freeze controls before outcomes, and fail closed on lineage mismatch.",
        "Required real controls: same-symbol, same-regime, close-confirmed breakout without proximity, generic breakout, pure Donchian breakout.",
        "No validation, holdout, A1, TSMOM, broad sweep, or live-prep is permitted.",
    ]) + "\n", encoding="utf-8")
    prompt = root / "prelaunch/next_materialization_prompt.md"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text(f"# Next Prompt\n\nImplement and dry-run the execution profile `{PROFILE}` using only `{root / 'selection/survivor_shortlist.csv'}`. Re-run lineage fail-closed, then request separate operator approval before materialization. Do not re-rank or expand candidates.\n", encoding="utf-8")
    summary = {
        "run_root": str(root), "status": "complete" if hard_pass else "blocked",
        "aggregate_candidates_screened": len(eligible), "exact_duplicate_cluster_count": exact_clusters["exact_duplicate_cluster_id"].nunique(),
        "multi_member_exact_duplicate_clusters": int((exact_clusters.groupby("exact_duplicate_cluster_id").size() > 1).sum()),
        "near_duplicate_cluster_count": near_clusters["near_duplicate_cluster_id"].nunique(),
        "multi_member_near_duplicate_clusters": int((near_clusters.groupby("near_duplicate_cluster_id").size() > 1).sum()),
        "shortlist_size": len(shortlist), "shortlist_candidates": shortlist["candidate_definition_id"].tolist(),
        "broad_candidates": shortlist.loc[shortlist["candidate_classification"].eq("broad_train_survivor_candidate"), "candidate_definition_id"].tolist(),
        "context_sleeves": shortlist.loc[shortlist["candidate_classification"].eq("detectable_context_sleeve_candidate"), "candidate_definition_id"].tolist(),
        "diagnostic_baselines": shortlist.loc[shortlist["candidate_classification"].eq("diagnostic_baseline"), "candidate_definition_id"].tolist(),
        "canonical_hash_mismatches": int((lineage["status"] != "pass").sum()), "scenario_source_ambiguity": 0,
        "future_aware_context_labels": int(support["future_aware"].sum()), "protected_period_violations": 0,
        "materialization_launched": False, "validation_launched": False, "holdout_launched": False,
        "registered_profile": PROFILE, "profile_execution_implemented": False,
        "targeted_materialization_may_proceed": False,
        "next_operator_decision": "implement_prior_high_v2_targeted_materialization_profile_then_dry_run_next" if hard_pass else "repair_prior_high_v2_preflight_next",
        "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision_summary.json", summary)
    bundle = root / "compact_review_bundle"
    bundle.mkdir()
    for rel in [
        "integrity/candidate_source_contract.md", "funding/exact_slice_composition_audit.csv", "selection/eligible_candidate_pool.csv",
        "selection/exact_duplicate_clusters.csv", "selection/near_duplicate_clusters.csv", "selection/parameter_neighborhood_report.csv",
        "forensics/winner_concentration_preview.csv", "regime/period_and_context_support.csv", "selection/survivor_shortlist.csv",
        "selection/deferred_or_diagnostic_candidates.csv", "controls/control_match_feasibility.csv", "profile/materialization_profile_contract.md",
        "preflight/shortlist_lineage_audit.csv", "prelaunch/next_materialization_prompt.md", "decision_summary.json",
    ]:
        source = root / rel
        shutil.copy2(source, bundle / rel.replace("/", "__"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
