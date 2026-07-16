#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_prior_high_v2_materialization_profile as profile
from tools import run_kraken_prior_high_v2_full_targeted as full


BLOCKED_ROOT = Path("results/rebaseline/phase_kraken_prior_high_reclaim_v2_full_targeted_materialization_20260712_v1")
IMPLEMENTATION_ROOT = Path("results/rebaseline/phase_kraken_prior_high_reclaim_v2_materialization_profile_implementation_20260712_v1")
PAIR_GROUPS = ({"prior_high_v2_022", "prior_high_v2_038"}, {"prior_high_v2_019", "prior_high_v2_035"})


def csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def canonical_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def precheck(candidate: Mapping[str, Any], bars: pd.DataFrame, funding: pd.DataFrame, address: Any) -> tuple[bool, str]:
    decision = pd.Timestamp(address.decision_ts)
    if decision >= runner.PROTECTED_TS:
        return False, "protected_decision"
    if int(address.entry_idx) >= len(bars) or int(address.exit_idx) <= int(address.entry_idx) or int(address.exit_idx) >= len(bars):
        return False, "invalid_entry_or_hold_interval"
    if pd.Timestamp(bars.iloc[int(address.exit_idx)]["ts"]) >= runner.PROTECTED_TS:
        return False, "protected_planned_hold_interval"
    if not runner.candidate_event_universe_allowed(candidate, str(address.symbol), decision):
        return False, "pit_universe_not_eligible"
    parent = runner.evaluate_parent_regime_gate(candidate, bars, decision)
    if not bool(parent.get("allowed", True)):
        return False, str(parent.get("skip_reason") or "parent_gate_filter_skip")
    gate = runner.evaluate_funding_gate(candidate, funding, decision)
    if not bool(gate.get("allowed", True)):
        return False, str(gate.get("skip_reason") or "funding_gate_filter_skip")
    entry = runner.safe_float(bars.iloc[int(address.entry_idx)].get("open"), np.nan)
    if not np.isfinite(entry) or entry <= 0:
        return False, "invalid_next_bar_entry_price"
    atr = runner.prior_high_atr_at_decision(candidate, bars, decision)
    atr_value = runner.safe_float(atr.get("atr_value"), np.nan)
    stop_mult = runner.safe_float(candidate.get("atr_stop_mult", candidate.get("atr_stop_multiple")), np.nan)
    source_ts = pd.to_datetime(atr.get("atr_feature_source_ts"), utc=True, errors="coerce")
    if not np.isfinite(atr_value) or atr_value <= 0 or not np.isfinite(stop_mult) or stop_mult <= 0:
        return False, str(atr.get("reason") or "invalid_atr_r_denominator")
    if pd.isna(source_ts) or source_ts > decision:
        return False, "atr_feature_unavailable_at_decision"
    risk = stop_mult * atr_value
    if not np.isfinite(risk) or risk <= 0:
        return False, "invalid_r_denominator"
    vwap_type = str(candidate.get("vwap_type", candidate.get("vwap_mode", "")) or "").strip().lower()
    if vwap_type:
        value, value_ts = runner.prior_high_latest_vwap(bars, decision, vwap_type)
        close = runner.safe_float(bars.iloc[int(address.idx)].get("close"), np.nan)
        direction = runner.prior_high_side_direction(candidate.get("side", "long"))
        if not np.isfinite(value) or pd.to_datetime(value_ts, utc=True, errors="coerce") > decision:
            return False, "vwap_unavailable_at_decision"
        if (direction == "long" and close < value) or (direction == "short" and close > value):
            return False, "vwap_entry_filter_reject"
    return True, "eligible_decision_time_only"


def load_candidate_ledgers() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definitions = pd.read_csv(IMPLEMENTATION_ROOT / "selection/frozen_definition_variant_manifest.csv")
    clusters = pd.read_csv(IMPLEMENTATION_ROOT / "selection/frozen_entry_cluster_manifest.csv")
    frames = []
    for cid in definitions["candidate_definition_id"].astype(str):
        path = BLOCKED_ROOT / "materialized/event_ledgers" / f"{cid}.parquet"
        if not path.exists():
            raise RuntimeError(f"missing immutable candidate ledger: {path}")
        frame = pd.read_parquet(path)
        frame["candidate_ledger_content_hash"] = canonical_file_hash(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False), definitions, clusters


def build_eligible_keys(events: pd.DataFrame, definitions: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]]:
    paths = runner.data_paths(profile.make_context(root))
    meta = definitions.set_index("candidate_definition_id")
    bars_cache: dict[str, pd.DataFrame] = {}
    funding_cache: dict[str, pd.DataFrame] = {}
    address_cache: dict[tuple[str, str, str], list[Any]] = {}
    evaluation: dict[tuple[str, str], tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]] = {}
    pool_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    def loaded(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if symbol not in bars_cache:
            bars_cache[symbol] = runner.load_symbol_bars(paths, symbol, pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))
            funding_cache[symbol] = runner.load_funding(paths, symbol, pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))
        return bars_cache[symbol], funding_cache[symbol]

    cluster_members = definitions.groupby("entry_cluster_id")["candidate_definition_id"].apply(lambda values: sorted(values.astype(str))).to_dict()
    representative = {cluster: members[0] for cluster, members in cluster_members.items()}
    event_lookup = events.set_index(["candidate_definition_id", "symbol", "decision_ts"], drop=False)
    rep_events = events[events.apply(lambda row: str(row.candidate_definition_id) == representative[str(meta.loc[str(row.candidate_definition_id), "entry_cluster_id"])], axis=1)]

    def eligible_for_all(cluster: str, symbol: str, address: Any) -> tuple[bool, str, dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]]:
        local: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]] = {}
        reasons = []
        bars, funding = loaded(symbol)
        for cid in cluster_members[cluster]:
            candidate = profile.candidate_for_symbol(meta.loc[cid].to_dict() | {"candidate_definition_id": cid}, symbol)
            okay, reason = precheck(candidate, bars, funding, address)
            if not okay:
                reasons.append(f"{cid}:{reason}")
            local[cid] = (candidate, bars, funding)
        return not reasons, ";".join(reasons) if reasons else "eligible_decision_time_only", local

    for event in rep_events.sort_values(["candidate_definition_id", "decision_ts", "symbol"]).itertuples(index=False):
        rep_id, symbol, decision = str(event.candidate_definition_id), str(event.symbol), pd.Timestamp(event.decision_ts)
        cluster = str(meta.loc[rep_id, "entry_cluster_id"])
        base = profile.candidate_for_symbol(meta.loc[rep_id].to_dict() | {"candidate_definition_id": rep_id}, symbol)
        bars, _ = loaded(symbol)
        bar_ts = pd.to_datetime(bars["ts"], utc=True)
        pos = int(np.searchsorted(bar_ts.astype("int64").to_numpy(), decision.value, side="left"))
        cadence = 288 if str(base.get("bar_timeframe")) == "daily" else 48
        candidates_by_class: dict[str, list[Any]] = {}
        same_positions = sorted({pos + cadence * offset for offset in range(-30, 31) if offset and 1 <= pos + cadence * offset < len(bars) - 2}, key=lambda idx: abs(idx - pos))
        candidates_by_class["same_symbol"] = [address for idx in same_positions if (address := profile.address_from_position(base, bars, idx, 0)) is not None]
        same_regime: list[Any] = []
        alternatives = sorted((s for s in runner.pit_policy_symbols_for_decision(base, decision) if s != symbol), key=lambda s: runner.stable_hash(cluster, decision, s, n=20))
        for other_symbol in alternatives[:20]:
            other_bars, _ = loaded(other_symbol)
            other_ts = pd.to_datetime(other_bars["ts"], utc=True)
            other_pos = int(np.searchsorted(other_ts.astype("int64").to_numpy(), decision.value, side="left"))
            other_candidate = profile.candidate_for_symbol(meta.loc[rep_id].to_dict() | {"candidate_definition_id": rep_id}, other_symbol)
            address = profile.address_from_position(other_candidate, other_bars, other_pos, 0)
            if address is not None:
                same_regime.append(address)
        candidates_by_class["same_regime"] = same_regime
        for control_class, engine_id, signal_override in [
            ("close_confirmed_breakout_without_proximity", "prior_high_reclaim_engine", "prior_high_breakout"),
            ("generic_breakout", "liquid_continuation_breakout_engine", "generic_breakout"),
        ]:
            cache_key = (rep_id, symbol, control_class)
            if cache_key not in address_cache:
                control_candidate = dict(base); control_candidate["signal_type"] = signal_override
                address_cache[cache_key] = runner.ENGINES[engine_id].enumerate_valid_event_addresses(bars, control_candidate)
            candidates_by_class[control_class] = sorted(address_cache[cache_key], key=lambda address: (abs((pd.Timestamp(address.decision_ts) - decision).total_seconds()), int(address.idx)))
        cache_key = (rep_id, symbol, "pure_donchian_breakout")
        if cache_key not in address_cache:
            address_cache[cache_key] = profile.donchian_addresses(base, bars)
        candidates_by_class["pure_donchian_breakout"] = sorted(address_cache[cache_key], key=lambda address: (abs((pd.Timestamp(address.decision_ts) - decision).total_seconds()), int(address.idx)))

        for control_class, addresses in candidates_by_class.items():
            chosen = None; chosen_local = {}; rejected: dict[str, int] = {}
            for address in addresses:
                if abs((pd.Timestamp(address.decision_ts) - decision).total_seconds()) > 30 * 86400:
                    continue
                ok, reason, local = eligible_for_all(cluster, str(address.symbol), address)
                pool_rows.append({"entry_cluster_id": cluster, "candidate_event_id": event.event_id, "control_class": control_class, "control_symbol": address.symbol, "control_decision_ts": address.decision_ts, "eligible": ok, "reason": reason, "outcome_fields_read": False})
                if ok:
                    chosen, chosen_local = address, local
                    break
                rejected[reason] = rejected.get(reason, 0) + 1
            if chosen is None:
                pool_rows.append({"entry_cluster_id": cluster, "candidate_event_id": event.event_id, "control_class": control_class, "control_symbol": "", "control_decision_ts": "", "eligible": False, "reason": "no_eligible_control:" + json.dumps(rejected, sort_keys=True), "outcome_fields_read": False})
                continue
            for cid in cluster_members[cluster]:
                target = event_lookup.loc[(cid, symbol, decision)]
                if isinstance(target, pd.DataFrame): target = target.iloc[0]
                key = runner.stable_hash(cluster, target.event_id, control_class, chosen.symbol, chosen.decision_ts, n=24)
                key_rows.append({"control_key": key, "candidate_definition_id": cid, "candidate_event_id": target.event_id, "entry_cluster_id": cluster, "control_class": control_class, "symbol": chosen.symbol, "decision_ts": chosen.decision_ts, "entry_idx": chosen.entry_idx, "exit_idx": chosen.exit_idx, "address_idx": chosen.idx, "control_key_frozen": True, "matching_used_outcomes": False})
                evaluation[(cid, key)] = chosen_local[cid]
    return pd.DataFrame(key_rows).drop_duplicates("control_key"), pd.DataFrame(pool_rows), evaluation


def rebuild_frozen_keys_from_pool(events: pd.DataFrame, definitions: pd.DataFrame, root: Path, pools: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]]:
    """Rehydrate frozen decision-time matches without repeating pool search."""
    paths = runner.data_paths(profile.make_context(root))
    meta = definitions.set_index("candidate_definition_id")
    cluster_members = definitions.groupby("entry_cluster_id")["candidate_definition_id"].apply(lambda values: sorted(values.astype(str))).to_dict()
    event_by_id = events.set_index("event_id", drop=False)
    event_lookup = events.set_index(["candidate_definition_id", "symbol", "decision_ts"], drop=False)
    bars_cache: dict[str, pd.DataFrame] = {}; funding_cache: dict[str, pd.DataFrame] = {}
    def loaded(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if symbol not in bars_cache:
            bars_cache[symbol] = runner.load_symbol_bars(paths, symbol, pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))
            funding_cache[symbol] = runner.load_funding(paths, symbol, pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))
        return bars_cache[symbol], funding_cache[symbol]
    selected = pools[pools["eligible"].astype(bool)].drop_duplicates(["entry_cluster_id", "candidate_event_id", "control_class"], keep="first")
    rows = []; evaluation = {}
    for match in selected.itertuples(index=False):
        source = event_by_id.loc[str(match.candidate_event_id)]
        if isinstance(source, pd.DataFrame): source = source.iloc[0]
        symbol = str(match.control_symbol); decision = pd.Timestamp(match.control_decision_ts)
        bars, funding = loaded(symbol)
        bar_ts = pd.to_datetime(bars["ts"], utc=True)
        pos = int(np.searchsorted(bar_ts.astype("int64").to_numpy(), decision.value, side="left"))
        for cid in cluster_members[str(match.entry_cluster_id)]:
            candidate = profile.candidate_for_symbol(meta.loc[cid].to_dict() | {"candidate_definition_id": cid}, symbol)
            address = profile.address_from_position(candidate, bars, pos, 0)
            if address is None or pd.Timestamp(address.decision_ts) != decision:
                raise RuntimeError(f"cannot deterministically reconstruct frozen control address: {cid} {match.control_class} {decision}")
            target = event_lookup.loc[(cid, str(source.symbol), pd.Timestamp(source.decision_ts))]
            if isinstance(target, pd.DataFrame): target = target.iloc[0]
            key = runner.stable_hash(str(match.entry_cluster_id), target.event_id, match.control_class, symbol, decision, n=24)
            rows.append({"control_key": key, "candidate_definition_id": cid, "candidate_event_id": target.event_id, "entry_cluster_id": match.entry_cluster_id, "control_class": match.control_class, "symbol": symbol, "decision_ts": decision, "entry_idx": address.entry_idx, "exit_idx": address.exit_idx, "address_idx": address.idx, "control_key_frozen": True, "matching_used_outcomes": False})
            evaluation[(cid, key)] = (candidate, bars, funding)
    return pd.DataFrame(rows).drop_duplicates("control_key"), evaluation


def adjudicate(definitions: pd.DataFrame, symmetric: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_scope = symmetric[(symmetric["funding_mode"] == "severe_imputed") & (symmetric["slippage_round_trip_bps"] == 12) & (symmetric["funding_partition"] == "all") & (symmetric["period"] == "full_train")]
    structural = {"close_confirmed_breakout_without_proximity", "generic_breakout", "pure_donchian_breakout"}
    contextual = {"same_symbol", "same_regime"}
    for definition in definitions.itertuples(index=False):
        cid = str(definition.candidate_definition_id)
        adequate = set(coverage[(coverage.candidate_definition_id == cid) & coverage.adequate_control_class].control_class)
        robust = full_scope[(full_scope.candidate_definition_id == cid) & full_scope.control_class.isin(adequate)]
        robust = set(robust[(robust.paired_mean_uplift_R > 0) & (robust.paired_median_uplift_R > 0) & (robust.winsorized_mean_uplift_R > 0) & (robust.mean_after_top_3_difference_removal > 0) & (robust.block_bootstrap_mean_ci_low > 0)].control_class)
        adjudicable = len(adequate) >= 3 and bool(adequate & structural) and bool(adequate & contextual)
        if cid in set(profile.PAIRS.values()): decision = "diagnostic_only"
        elif adjudicable and len(robust) >= 3 and bool(robust & structural) and bool(robust & contextual): decision = "advance_to_train_stability_review"
        elif adjudicable: decision = "preserve_as_context_sleeve"
        else: decision = "defer_current_translation"
        rows.append({"candidate_definition_id": cid, "entry_cluster_id": definition.entry_cluster_id, "adequate_control_classes": ";".join(sorted(adequate)), "adequate_control_class_count": len(adequate), "robust_control_classes": ";".join(sorted(robust)), "robust_control_class_count": len(robust), "has_contextual_control": bool(adequate & contextual), "has_structural_control": bool(adequate & structural), "candidate_adjudicable": adjudicable, "candidate_decision": decision, "evidence_level": "train_only_level_3_controls_capped" if decision == "advance_to_train_stability_review" else "train_only_diagnostic_or_context_capped"})
    return pd.DataFrame(rows)


def run(root: Path, *, resume: bool = False) -> dict[str, Any]:
    if root.exists() and not resume: raise RuntimeError(f"fresh run root required: {root}")
    root.mkdir(parents=True, exist_ok=resume)
    events, definitions, clusters = load_candidate_ledgers()
    if len(definitions) != 11 or len(clusters) != 9: raise RuntimeError("frozen scope mismatch")
    hashes = events.groupby("candidate_definition_id")["candidate_ledger_content_hash"].first().reset_index()
    csv(root / "audit/candidate_ledger_reuse_audit.csv", hashes.assign(candidate_ledgers_reused=True, candidate_rows_regenerated=False))
    pool_path = root / "controls/eligible_control_pool_manifest.csv"
    if resume:
        if not pool_path.exists() or not (root / "controls/control_key_manifest.csv").exists():
            raise RuntimeError("resume requires finalized eligible pool and prior key manifest")
        pools = pd.read_csv(pool_path)
        if pools.empty or "outcome_fields_read" not in pools or pools["outcome_fields_read"].astype(bool).any():
            raise RuntimeError("resume pool manifest failed decision-time-only validation")
        keys, evaluation = rebuild_frozen_keys_from_pool(events, definitions, root, pools)
    else:
        keys, pools, evaluation = build_eligible_keys(events, definitions, root)
        csv(pool_path, pools)
    (root / "controls/control_matching_contract.md").write_text("# Control matching contract\n\nPools use PIT universe, parent/funding gates, entry/VWAP executability, ATR risk denominator, lifecycle, and protected-boundary inputs available at decision time. No return, exit price, PnL, MAE/MFE, or control outcome is read before key freeze. Paired variants share entry matches and apply their own economic policy after freeze.\n", encoding="utf-8")
    freeze_ts = pd.Timestamp.now(tz="UTC")
    key_hash = runner.canonical_frame_hash(keys, sort_keys=["candidate_definition_id", "candidate_event_id", "control_class", "control_key"])
    if resume:
        prior_manifest = pd.read_csv(root / "controls/control_key_manifest.csv")
        if len(prior_manifest) != 1 or str(prior_manifest.iloc[0]["control_key_hash"]) != key_hash or int(prior_manifest.iloc[0]["row_count"]) != len(keys):
            raise RuntimeError("reconstructed control keys do not match prior frozen key manifest")
    csv(root / "controls/control_keys.csv", keys)
    csv(root / "controls/control_key_manifest.csv", [{"control_key_hash": key_hash, "row_count": len(keys), "freeze_ts": freeze_ts, "matching_used_outcomes": False, "status": "pass"}])
    outcome_start = pd.Timestamp.now(tz="UTC")
    outcomes = profile.evaluate_control_outcomes(keys, evaluation)
    outcomes_path = root / "controls/control_outcomes.parquet"
    runner.parquet_safe_frame(outcomes).to_parquet(outcomes_path, index=False, compression="zstd")
    attrition = profile.diagnose_control_attrition(keys, outcomes, evaluation)
    csv(root / "controls/control_attrition_reasons.csv", attrition)
    funded, scenarios, joined = profile.funding_correct(outcomes)
    candidate_scenarios = []
    for path in sorted((BLOCKED_ROOT / "materialized/event_ledgers").glob("*.parquet")):
        funded_candidate = pd.read_parquet(path)
        candidate_scenarios.append(profile.balanced.scenario_event_rows(
            funded_candidate, ("central_imputed", "conservative_imputed", "severe_imputed", "exact_only_slice"), (4, 8, 12),
        ))
    candidate_scenarios = pd.concat(candidate_scenarios, ignore_index=True)
    candidate_counts = events.groupby("candidate_definition_id")["event_id"].nunique()
    key_counts = keys.groupby(["candidate_definition_id", "control_class"]).agg(control_key_count=("control_key", "nunique"), matched_candidate_events=("candidate_event_id", "nunique")).reset_index()
    outcome_counts = outcomes.groupby(["source_candidate_definition_id", "control_class"]).agg(completed_control_outcomes=("control_key", "nunique")).reset_index().rename(columns={"source_candidate_definition_id": "candidate_definition_id"})
    grid = pd.MultiIndex.from_product([definitions.candidate_definition_id.astype(str), profile.CONTROL_CLASSES], names=["candidate_definition_id", "control_class"]).to_frame(index=False)
    coverage = grid.merge(key_counts, how="left").merge(outcome_counts, how="left").fillna({"control_key_count": 0, "matched_candidate_events": 0, "completed_control_outcomes": 0})
    for col in ["control_key_count", "matched_candidate_events", "completed_control_outcomes"]: coverage[col] = coverage[col].astype(int)
    coverage["candidate_event_count"] = coverage.candidate_definition_id.map(candidate_counts)
    coverage["key_coverage"] = coverage.matched_candidate_events / coverage.candidate_event_count
    coverage["outcome_coverage"] = coverage.completed_control_outcomes / coverage.candidate_event_count
    coverage["adequate_control_class"] = (coverage.outcome_coverage >= .70) & (coverage.completed_control_outcomes >= 15)
    attr_counts = attrition.groupby(["candidate_definition_id", "control_class"]).agg(explained_attrition=("explained", "sum"), unexplained_attrition=("explained", lambda x: int((~x.astype(bool)).sum())), remaining_unavailable_reasons=("attrition_reason", lambda x: ";".join(sorted(set(x.astype(str)))))).reset_index() if len(attrition) else pd.DataFrame(columns=["candidate_definition_id", "control_class", "explained_attrition", "unexplained_attrition", "remaining_unavailable_reasons"])
    coverage = coverage.merge(attr_counts, how="left").fillna({"explained_attrition": 0, "unexplained_attrition": 0, "remaining_unavailable_reasons": ""})
    zero_reason = pools[~pools.eligible & pools.reason.astype(str).str.startswith("no_eligible_control")].groupby(["entry_cluster_id", "control_class"])["reason"].first().to_dict()
    coverage["zero_control_explicit_reason"] = coverage.apply(lambda row: zero_reason.get((definitions.set_index("candidate_definition_id").loc[row.candidate_definition_id, "entry_cluster_id"], row.control_class), "") if row.control_key_count == 0 else "", axis=1)
    csv(root / "controls/control_match_coverage.csv", coverage)
    old = pd.read_csv(BLOCKED_ROOT / "controls/control_match_coverage.csv")
    comparison = old.merge(coverage, on=["candidate_definition_id", "control_class"], suffixes=("_old", "_new"))
    csv(root / "controls/old_new_coverage_comparison.csv", comparison)
    paired = full.build_paired(candidate_scenarios, scenarios)
    symmetric, leave = full.symmetric_forensics(paired)
    csv(root / "controls/symmetric_control_forensics.csv", symmetric)
    csv(root / "controls/symmetric_control_leave_one.csv", leave)
    csv(root / "controls/matched_unmatched_bias.csv", full.matched_unmatched(candidate_scenarios, outcomes))
    decisions = adjudicate(definitions, symmetric, coverage)
    csv(root / "decision/candidate_control_adjudication.csv", decisions)
    csv(root / "candidate_library/prior_high_candidate_library_update.csv", decisions.assign(candidate_library_status=decisions.candidate_decision, train_only=True))
    missing = int((joined._merge != "both").sum()); duplicate = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    unexplained = int(coverage.unexplained_attrition.sum())
    unexplained_zero = int(((coverage.control_key_count == 0) & coverage.zero_control_explicit_reason.eq("")).sum())
    protected = sum(int((pd.to_datetime(outcomes[col], utc=True, errors="coerce") >= runner.PROTECTED_TS).sum()) for col in ["decision_ts", "entry_ts", "exit_interval_end_ts"])
    hard = len(definitions) == 11 and len(clusters) == 9 and not pools.outcome_fields_read.any() and unexplained == 0 and unexplained_zero == 0 and missing == duplicate == protected == 0 and outcome_start >= freeze_ts
    summary = {"run_root": str(root), "status": "complete" if hard else "blocked", "candidate_ledgers_reused": True, "definitions": 11, "entry_clusters": 9, "candidate_event_rows": int(events.event_key.nunique()), "control_key_rows": len(keys), "control_outcome_rows": int(outcomes.control_key.nunique()), "outcome_informed_matching": 0, "unexplained_control_attrition": unexplained, "remaining_zero_control_cells": int((coverage.completed_control_outcomes == 0).sum()), "zero_control_cells_without_reason": unexplained_zero, "missing_funding_joins": missing, "duplicate_funding_joins": duplicate, "protected_period_violations": protected, "control_keys_frozen_before_outcomes": outcome_start >= freeze_ts, "placeholder_controls": 0, "validation_launched": False, "holdout_launched": False, "candidates_with_three_adequate_controls": decisions.loc[decisions.adequate_control_class_count >= 3, "candidate_definition_id"].tolist(), "candidates_robustly_beating_controls": decisions.loc[decisions.candidate_decision == "advance_to_train_stability_review", "candidate_definition_id"].tolist(), "advanced_candidates": decisions.loc[decisions.candidate_decision == "advance_to_train_stability_review", "candidate_definition_id"].tolist(), "context_candidates": decisions.loc[decisions.candidate_decision == "preserve_as_context_sleeve", "candidate_definition_id"].tolist(), "deferred_candidates": decisions.loc[decisions.candidate_decision == "defer_current_translation", "candidate_definition_id"].tolist(), "diagnostic_candidates": decisions.loc[decisions.candidate_decision == "diagnostic_only", "candidate_definition_id"].tolist(), "train_only_stability_review_authorized": bool((decisions.candidate_decision == "advance_to_train_stability_review").any()), "compact_bundle_path": str(root / "compact_review_bundle")}
    (root / "decision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    bundle = root / "compact_review_bundle"; bundle.mkdir()
    for path in [root / "controls/eligible_control_pool_manifest.csv", root / "controls/control_matching_contract.md", root / "controls/old_new_coverage_comparison.csv", root / "controls/control_key_manifest.csv", root / "controls/control_attrition_reasons.csv", root / "controls/control_match_coverage.csv", root / "controls/symmetric_control_forensics.csv", root / "controls/matched_unmatched_bias.csv", root / "decision/candidate_control_adjudication.csv", root / "candidate_library/prior_high_candidate_library_update.csv", root / "decision_summary.json"]: shutil.copy2(path, bundle / path.name)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    result = run(Path(args.run_root), resume=args.resume); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"] == "complete" else 2


if __name__ == "__main__": raise SystemExit(main())
