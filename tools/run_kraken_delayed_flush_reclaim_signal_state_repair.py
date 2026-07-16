#!/usr/bin/env python3
"""Repair DFRL signal state and rerun the frozen train-only 24-definition screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from tools import qlmg_evidence_contracts as evidence
from tools import run_kraken_backside_blowoff_short_screen as reports
from tools import run_kraken_delayed_flush_reclaim_long_screen as base
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs


START = base.START
END = base.END
PROTECTED = base.PROTECTED
EVALUATION_WINDOWS = base.EVALUATION_WINDOWS
SOURCE_ROOT = Path("results/rebaseline/phase_kraken_delayed_flush_reclaim_long_screen_20260715_v1")
CONTRACT_VERSION = "kraken_delayed_flush_reclaim_signal_state_repair_v1_20260715"
KNOWN_REGRESSIONS = (
    ("PF_RENDERUSD", pd.Timestamp("2025-02-05T12:00:00Z"), "moderate_12pct_3d_1.5atr", 3),
    ("PF_ENAUSD", pd.Timestamp("2025-10-15T12:00:00Z"), "strong_20pct_5d_2.5atr", 3),
)


def raw_policy_hash(flush_profile: str, stabilization_bars: int) -> str:
    return base.stable_hash({
        "flush_profile": flush_profile,
        "stabilization_bars": int(stabilization_bars),
        "signal_timeframe": "4h_completed",
        "execution_timeframe": "5m_next_open",
        "protected_boundary": PROTECTED.isoformat(),
        "sequence_contract": "lower_low_reset_exact_stabilization_completion_anchored_vwap",
        "contract_version": CONTRACT_VERSION,
    })


def raw_signal_address(row: Mapping[str, Any]) -> str:
    return base.stable_hash({
        "raw_policy_hash": row["raw_policy_hash"],
        "symbol": row["symbol"],
        "decision_ts": row["decision_ts"],
        "entry_ts": row["entry_ts"],
        "entry_price": row["entry_price"],
        "initial_stop": row["initial_stop"],
        "risk_denominator": row["risk_denominator"],
    })


def raw_specs() -> list[dict[str, Any]]:
    rows = []
    for flush in ("moderate_12pct_3d_1.5atr", "strong_20pct_5d_2.5atr"):
        for stabilization in (1, 3):
            rows.append({
                "flush_profile": flush,
                "stabilization_bars": stabilization,
                "raw_policy_hash": raw_policy_hash(flush, stabilization),
            })
    return rows


def enumerate_raw_signals(
    ctx: Any,
    panel: pd.DataFrame,
    symbol: str,
    bars: pd.DataFrame,
    prepared: tuple | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, dict]:
    """Emit all PIT/mechanically valid signals without parent or position filtering."""
    frame, _, _, sequences = prepared or base.prepare_symbol(bars)
    rows: list[dict[str, Any]] = []
    boundary_drops: list[dict[str, Any]] = []
    panel_row = panel[panel.symbol.eq(symbol)]
    if panel_row.empty or str(panel_row.iloc[0].status) != "available":
        return rows, boundary_drops, frame, sequences
    listed = pd.Timestamp(panel_row.iloc[0].start_ts)
    for spec in raw_specs():
        for sequence in sequences[(spec["flush_profile"], spec["stabilization_bars"])]["confirmed"]:
            decision = frame.iloc[sequence["decision_index"]]
            decision_ts = pd.Timestamp(decision.decision_ts)
            if decision_ts < START or decision_ts >= PROTECTED:
                boundary_drops.append({"symbol": symbol, "decision_ts": decision_ts, "reason": "raw_decision_outside_train_window", **spec})
                continue
            if decision_ts < listed + pd.Timedelta(days=30) or not base.pit_allowed(ctx, panel, decision_ts, symbol):
                continue
            if pd.isna(decision.feature_available_ts) or pd.Timestamp(decision.feature_available_ts) > decision_ts:
                continue
            entries = bars[bars.ts >= decision_ts]
            if entries.empty:
                boundary_drops.append({"symbol": symbol, "decision_ts": decision_ts, "reason": "next_executable_bar_unavailable", **spec})
                continue
            entry = entries.iloc[0]
            entry_ts = pd.Timestamp(entry.ts)
            if entry_ts >= PROTECTED:
                boundary_drops.append({"symbol": symbol, "decision_ts": decision_ts, "entry_ts": entry_ts, "reason": "next_entry_crosses_protected_boundary", **spec})
                continue
            daily_atr = float(sequence["daily_atr"])
            if not np.isfinite(daily_atr) or daily_atr <= 0:
                continue
            stop = float(sequence["sequence_low"])
            risk = float(entry.open) - stop
            risk_atr = risk / daily_atr
            if risk_atr < .25 or risk_atr > 1.5:
                continue
            period, window_start, window_end = base.evaluation_window(entry_ts)
            row = {
                "raw_policy_hash": spec["raw_policy_hash"],
                "symbol": symbol,
                "flush_profile": spec["flush_profile"],
                "stabilization_bars": spec["stabilization_bars"],
                "parent_state": decision.parent_state,
                "parent_source_ts": decision.parent_source_ts,
                "decision_ts": decision_ts,
                "feature_available_ts": decision.feature_available_ts,
                "entry_ts": entry_ts,
                "entry_price": float(entry.open),
                "initial_stop": stop,
                "risk_denominator": risk,
                "risk_to_daily_atr": risk_atr,
                "daily_atr": daily_atr,
                "sequence_low": stop,
                "flush_anchor_ts": sequence["anchor_ts"],
                "flush_anchored_vwap": sequence["flush_anchored_vwap"],
                "pre_flush_high": sequence["pre_flush_high"],
                "evaluation_period": period,
                "evaluation_window_start": window_start,
                "evaluation_window_end": window_end,
                "bar_based_forced_flow_proxy_only": True,
                "imputed_funding_gate_activated": False,
            }
            row["raw_signal_address_hash"] = raw_signal_address(row)
            row["raw_signal_id"] = "DFRLRAW_" + row["raw_signal_address_hash"][:24]
            rows.append(row)
    return rows, boundary_drops, frame, sequences


def project_parent_policies(raw: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    """Project parent policies from one immutable raw tape; no overlap state lives here."""
    rows = []
    policies = definitions.drop_duplicates("selected_key_policy_hash")
    for policy in policies.itertuples(index=False):
        selected = raw[
            raw.flush_profile.eq(policy.flush_profile)
            & raw.stabilization_bars.eq(policy.stabilization_bars)
        ]
        selected = selected[selected.parent_state.map(lambda state: base.parent_allowed(policy.parent_policy, str(state)))]
        for source in selected.to_dict("records"):
            row = {**source, "parent_policy": policy.parent_policy, "selected_key_policy_hash": policy.selected_key_policy_hash}
            row["candidate_key"] = "DFRLK_" + base.stable_hash({
                "selected_key_policy_hash": policy.selected_key_policy_hash,
                "raw_signal_address_hash": source["raw_signal_address_hash"],
            })[:24]
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["selected_key_policy_hash", "symbol", "entry_ts", "candidate_key"]).drop_duplicates("candidate_key")
    return result


def raw_nesting_audit(raw: pd.DataFrame, candidates: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policies = definitions.drop_duplicates("selected_key_policy_hash")
    for (flush, stabilization), group in policies.groupby(["flush_profile", "stabilization_bars"]):
        raw_set = set(raw[(raw.flush_profile == flush) & (raw.stabilization_bars == stabilization)].raw_signal_address_hash)
        strict_hash = group[group.parent_policy.eq("stress_both_down")].selected_key_policy_hash.iloc[0]
        broad_hash = group[group.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        strict_set = set(candidates[candidates.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash)
        broad_set = set(candidates[candidates.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash)
        expected_broad = set(raw[(raw.flush_profile == flush) & (raw.stabilization_bars == stabilization) & raw.parent_state.ne("unknown")].raw_signal_address_hash)
        rows.append({
            "flush_profile": flush,
            "stabilization_bars": stabilization,
            "raw_signals": len(raw_set),
            "strict_signals": len(strict_set),
            "all_regime_signals": len(broad_set),
            "strict_not_in_all_regime": len(strict_set - broad_set),
            "all_regime_missing_known_raw": len(expected_broad - broad_set),
            "pass": not (strict_set - broad_set) and broad_set == expected_broad,
        })
    return pd.DataFrame(rows)


def simulate_definition(
    candidates: pd.DataFrame,
    definition: Mapping[str, Any],
    execute_fn: Callable[[Mapping[str, Any], str], tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply no-overlap only to this definition using its actual executable exits."""
    accepted: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    open_trade: dict[str, dict[str, Any]] = {}
    ordered = candidates.sort_values(["entry_ts", "symbol", "candidate_key"])
    for key in ordered.to_dict("records"):
        prior = open_trade.get(key["symbol"])
        if prior is not None and pd.Timestamp(key["entry_ts"]) < pd.Timestamp(prior["exit_ts"]):
            skips.append({
                "definition_id": definition["definition_id"],
                "candidate_key": key["candidate_key"],
                "symbol": key["symbol"],
                "entry_ts": key["entry_ts"],
                "prior_trade_id": prior["event_id"],
                "prior_entry_ts": prior["entry_ts"],
                "prior_actual_exit_ts": prior["exit_ts"],
                "skip_reason": "same_symbol_same_definition_position_actually_open",
            })
            continue
        event, excluded = execute_fn(key, definition["exit_policy"])
        if excluded is not None:
            exclusions.append({**excluded, "definition_id": definition["definition_id"]})
            continue
        assert event is not None
        event["definition_id"] = definition["definition_id"]
        event["parameter_vector_hash"] = definition["parameter_vector_hash"]
        event["event_id"] = "DFRLE_" + base.stable_hash({"candidate": key["candidate_key"], "definition": definition["definition_id"]})[:24]
        event["candidate_economic_address_hash"] = base.economic_address(event)
        accepted.append(event)
        open_trade[key["symbol"]] = event
    return accepted, skips, exclusions


def simulate_all_definitions(
    candidates: pd.DataFrame,
    definitions: pd.DataFrame,
    bars_cache: dict[str, pd.DataFrame],
    feature_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for definition in definitions.to_dict("records"):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition["selected_key_policy_hash"])]

        def execute(key: Mapping[str, Any], exit_policy: str):
            return base.execute_event(key, exit_policy, bars_cache[key["symbol"]], feature_cache[key["symbol"]])

        accepted, definition_skips, definition_exclusions = simulate_definition(selected, definition, execute)
        outcomes.extend(accepted)
        skips.extend(definition_skips)
        exclusions.extend(definition_exclusions)
    return pd.DataFrame(outcomes), pd.DataFrame(skips), pd.DataFrame(exclusions)


def deterministic_outcome_hash(outcomes: pd.DataFrame) -> str:
    fields = ["definition_id", "candidate_key", "entry_ts", "exit_ts", "exit_price", "exit_reason", "gross_R"]
    rows = outcomes[fields].sort_values(["definition_id", "candidate_key"]).astype(str).to_dict("records")
    return base.stable_hash(rows)


def deterministic_frame_hash(frame: pd.DataFrame, sort_fields: list[str]) -> str:
    if frame.empty:
        return base.stable_hash([])
    fields = sorted(frame.columns)
    rows = frame.sort_values(sort_fields)[fields].astype(str).to_dict("records")
    return base.stable_hash(rows)


def directory_content_hash(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"relative_path": str(path.relative_to(root)), "sha256": base.file_hash(path), "bytes": path.stat().st_size})
    return base.stable_hash(rows)


def sentinel_audit(
    candidates: pd.DataFrame,
    definitions: pd.DataFrame,
    bars_cache: dict[str, pd.DataFrame],
    feature_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    representatives = definitions.sort_values("definition_id").groupby("selected_key_policy_hash", as_index=False).first()
    for definition in representatives.to_dict("records"):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition["selected_key_policy_hash"])].sort_values(["symbol", "entry_ts"]).head(3)
        first = []
        second = []
        for key in selected.to_dict("records"):
            a, _ = base.execute_event(key, definition["exit_policy"], bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            b, _ = base.execute_event(key, definition["exit_policy"], bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            for target, event in ((first, a), (second, b)):
                if event:
                    target.append(base.stable_hash({field: event[field] for field in ("candidate_key", "exit_ts", "exit_price", "exit_reason", "gross_R")}))
        rows.append({
            "selected_key_policy_hash": definition["selected_key_policy_hash"],
            "definition_id": definition["definition_id"],
            "sampled_signals": len(selected),
            "first_outcomes": len(first),
            "second_outcomes": len(second),
            "mismatch_count": len(set(first).symmetric_difference(second)),
            "pass": first == second,
            "profitability_used_for_continuation": False,
        })
    return pd.DataFrame(rows)


def known_regression_audit(raw: pd.DataFrame, candidates: pd.DataFrame, old_candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    affected = []
    for symbol, decision_ts, flush, stabilization in KNOWN_REGRESSIONS:
        raw_match = raw[(raw.symbol == symbol) & (pd.to_datetime(raw.decision_ts, utc=True) == decision_ts) & (raw.flush_profile == flush) & (raw.stabilization_bars == stabilization)]
        policy_rows = candidates[(candidates.symbol == symbol) & (pd.to_datetime(candidates.decision_ts, utc=True) == decision_ts) & (candidates.flush_profile == flush) & (candidates.stabilization_bars == stabilization)]
        parent_set = set(policy_rows.parent_policy)
        old_match = old_candidates[(old_candidates.symbol == symbol) & (pd.to_datetime(old_candidates.decision_ts, utc=True) == decision_ts)]
        row = {
            "symbol": symbol,
            "decision_ts": decision_ts,
            "flush_profile": flush,
            "stabilization_bars": stabilization,
            "raw_signal_present": len(raw_match) == 1,
            "strict_parent_present": "stress_both_down" in parent_set,
            "all_regime_present": "all_regime_comparator" in parent_set,
            "old_policy_rows": len(old_match),
        }
        row["pass"] = row["raw_signal_present"] and row["strict_parent_present"] and row["all_regime_present"]
        rows.append(row)
        if len(raw_match):
            for item in policy_rows.to_dict("records"):
                affected.append({"change": "repaired_parent_projection", **{field: item[field] for field in ("symbol", "decision_ts", "entry_ts", "parent_state", "parent_policy", "candidate_key", "raw_signal_address_hash")}})
        for item in old_match.to_dict("records"):
            affected.append({"change": "blocked_source_selected_row", **{field: item.get(field) for field in ("symbol", "decision_ts", "entry_ts", "parent_state", "parent_policy", "candidate_key")}})
    return pd.DataFrame(rows), pd.DataFrame(affected)


def old_new_comparison(root: Path, raw: pd.DataFrame, candidates: pd.DataFrame, outcomes: pd.DataFrame, definitions: pd.DataFrame) -> None:
    old_keys = pd.read_csv(SOURCE_ROOT / "keys/candidate_key_manifest.csv")
    old_events = pd.read_csv(SOURCE_ROOT / "materialized/event_ledger.csv")
    old_summary = pd.read_csv(SOURCE_ROOT / "economics/definition_summary.csv")
    new_summary, _, _ = reports.summarize_economics(outcomes, definitions)
    counts = definitions[["definition_id"]].copy()
    counts = counts.merge(old_events.groupby("definition_id").size().rename("old_accepted_trades"), on="definition_id", how="left")
    counts = counts.merge(outcomes.groupby("definition_id").size().rename("repaired_accepted_trades"), on="definition_id", how="left").fillna(0)
    counts["accepted_trade_change"] = counts.repaired_accepted_trades - counts.old_accepted_trades
    base.write_csv(root / "comparison/accepted_trade_count_comparison.csv", counts)
    rows = []
    for definition in definitions.itertuples(index=False):
        old_set = set(old_events[old_events.definition_id.eq(definition.definition_id)].candidate_economic_address_hash)
        new_set = set(outcomes[outcomes.definition_id.eq(definition.definition_id)].candidate_economic_address_hash)
        rows.append({"definition_id": definition.definition_id, "old_addresses": len(old_set), "repaired_addresses": len(new_set), "newly_admitted": len(new_set-old_set), "removed": len(old_set-new_set), "unchanged": len(old_set&new_set)})
    base.write_csv(root / "comparison/economic_address_change_audit.csv", rows)
    economics = old_summary.merge(new_summary, on=["definition_id", "cost_mode"], suffixes=("_old", "_repaired"))
    economics["mean_R_change"] = economics.mean_R_repaired - economics.mean_R_old
    economics["event_count_change"] = economics.events_repaired - economics.events_old
    base.write_csv(root / "comparison/economics_old_vs_repaired.csv", economics)
    base.write_csv(root / "comparison/raw_and_parent_signal_count_comparison.csv", [
        {"scope": "blocked_source_parent_filtered_preblocked", "rows": len(old_keys), "unique_addresses": old_keys.candidate_key.nunique()},
        {"scope": "repaired_parent_neutral_raw", "rows": len(raw), "unique_addresses": raw.raw_signal_address_hash.nunique()},
        {"scope": "repaired_parent_filtered_unblocked", "rows": len(candidates), "unique_addresses": candidates.candidate_key.nunique()},
    ])


def compact_bundle(root: Path) -> Path:
    files = (
        "decision_summary.json", "contract/signal_state_repair_contract.md", "signals/raw_signal_manifest.csv",
        "signals/parent_filtered_signal_manifest.csv", "audit/hard_gate_audit.csv", "audit/raw_signal_nesting_audit.csv",
        "audit/known_signal_regression_audit.csv", "audit/deterministic_rerun_parity.csv",
        "audit/boundary_reconciliation.csv", "audit/non_overlap_skip_ledger.csv",
        "comparison/raw_and_parent_signal_count_comparison.csv", "comparison/accepted_trade_count_comparison.csv",
        "comparison/economic_address_change_audit.csv", "comparison/economics_old_vs_repaired.csv",
        "economics/definition_summary.csv", "economics/period_summary.csv", "controls/control_summary.csv",
        "forensics/concentration_and_removal.csv", "forensics/exact_vs_imputed_funding.csv",
        "decision/candidate_decisions.csv", "candidate_library/delayed_flush_reclaim_candidate_library_update.csv",
        "reproducibility/run_manifest.json",
    )
    temp = root / ".compact_review_bundle.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    inventory = []
    for relative in files:
        source = root / relative
        target = temp / relative.replace("/", "__")
        shutil.copy2(source, target)
        inventory.append({"bundle_file": target.name, "source_relative_path": relative, "bytes": source.stat().st_size, "sha256": base.file_hash(source)})
    base.write_csv(temp / "bundle_manifest.csv", inventory)
    os.replace(temp, root / "compact_review_bundle")
    return root / "compact_review_bundle"


def run(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"fresh root required: {root}")
    source_decision_hash_before = base.file_hash(SOURCE_ROOT / "decision_summary.json")
    source_tree_hash_before = directory_content_hash(SOURCE_ROOT)
    root.mkdir(parents=True)
    started = time.monotonic()
    definitions = base.frozen_manifest()
    base.write_csv(root / "manifest/delayed_flush_reclaim_definitions.csv", definitions)
    contract = """# Delayed Flush Reclaim Signal-State Repair Contract

The blocked source root remains diagnostic-only. The frozen 24 definitions, feature formulas, PIT universe, entries, stops, exits, 0.25-1.5 ATR risk band, costs, shared funding model and five control definitions are unchanged. Four parent-neutral raw tapes emit every mechanically valid completed reclaim without a maximum-hold pre-block. The two parent policies are PIT projections of those immutable raw tapes. Non-overlap belongs only to each definition and is applied chronologically using that definition's actual executable exit timestamp. Evaluation-boundary crossings are excluded, never force-exited. Control keys are rebuilt from repaired accepted identities and frozen before outcomes. No profitability selects signals or definitions.
"""
    contract_path = root / "contract/signal_state_repair_contract.md"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(contract, encoding="utf-8")
    ctx = base.context(root)
    panel = runner.full_panel_for_launch_gate(ctx)
    base.write_csv(root / "manifest/pit_panel.csv", panel)
    paths = runner.data_paths(ctx)
    raw_rows: list[dict[str, Any]] = []
    raw_boundary_drops: list[dict[str, Any]] = []
    feature_cache: dict[str, pd.DataFrame] = {}
    sequence_cache: dict[str, dict] = {}
    bars_cache: dict[str, pd.DataFrame] = {}
    peak_rss = 0
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        bars = runner.load_symbol_bars(paths, symbol, START-pd.Timedelta(days=120), END)
        if bars.empty:
            continue
        bars = bars[["ts", "open", "high", "low", "close", "volume"]].copy()
        prepared = base.prepare_symbol(bars)
        rows, drops, frame, sequences = enumerate_raw_signals(ctx, panel, symbol, bars, prepared)
        raw_rows.extend(rows)
        raw_boundary_drops.extend(drops)
        if rows:
            feature_cache[symbol] = frame
            sequence_cache[symbol] = sequences
            bars_cache[symbol] = bars
        peak_rss = max(peak_rss, runner.current_rss_bytes())
        base.write_json(root / "watch_status.json", {"status": "running", "stage": "raw_signal_build", "symbols_completed": number, "symbols_planned": len(panel), "raw_signals": len(raw_rows), "rss_bytes": peak_rss, "elapsed_seconds": time.monotonic()-started, "updated_ts": runner.utc_now()})
    raw = pd.DataFrame(raw_rows).sort_values(["raw_policy_hash", "symbol", "entry_ts", "raw_signal_address_hash"])
    raw_duplicate_count = int(raw.duplicated(["raw_policy_hash", "raw_signal_address_hash"]).sum())
    raw_freeze_hash = base.stable_hash(raw.raw_signal_address_hash.tolist())
    raw["raw_signal_freeze_hash"] = raw_freeze_hash
    base.write_csv(root / "signals/raw_signal_manifest.csv", raw)
    base.write_csv(root / "audit/raw_boundary_drop_audit.csv", raw_boundary_drops)
    candidates = project_parent_policies(raw, definitions)
    parent_freeze_hash = base.stable_hash(candidates.candidate_key.tolist())
    candidates["selected_key_freeze_hash"] = parent_freeze_hash
    base.write_csv(root / "signals/parent_filtered_signal_manifest.csv", candidates)
    nesting = raw_nesting_audit(raw, candidates, definitions)
    base.write_csv(root / "audit/raw_signal_nesting_audit.csv", nesting)
    old_candidates = pd.read_csv(SOURCE_ROOT / "keys/candidate_key_manifest.csv")
    known, affected = known_regression_audit(raw, candidates, old_candidates)
    base.write_csv(root / "audit/known_signal_regression_audit.csv", known)
    base.write_csv(root / "audit/known_signal_before_after.csv", affected)
    sentinel = sentinel_audit(candidates, definitions, bars_cache, feature_cache)
    base.write_csv(root / "audit/exactness_sentinel.csv", sentinel)
    if len(sentinel) != 8 or sentinel.selected_key_policy_hash.nunique() != 8 or not sentinel["pass"].all():
        raise RuntimeError("repaired selected-key exactness sentinel failed")
    base.write_json(root / "watch_status.json", {"status": "running", "stage": "definition_outcome_simulation", "raw_signals": len(raw), "parent_filtered_signals": len(candidates), "rss_bytes": runner.current_rss_bytes(), "elapsed_seconds": time.monotonic()-started, "updated_ts": runner.utc_now()})
    outcomes, skips, exclusions = simulate_all_definitions(candidates, definitions, bars_cache, feature_cache)
    base.write_json(root / "watch_status.json", {"status": "running", "stage": "deterministic_rerun_parity", "accepted_trade_rows": len(outcomes), "overlap_skips": len(skips), "definition_outcome_exclusions": len(exclusions), "rss_bytes": runner.current_rss_bytes(), "elapsed_seconds": time.monotonic()-started, "updated_ts": runner.utc_now()})
    parity_outcomes, parity_skips, parity_exclusions = simulate_all_definitions(candidates, definitions, bars_cache, feature_cache)
    parity = {
        "first_outcome_hash": deterministic_outcome_hash(outcomes),
        "second_outcome_hash": deterministic_outcome_hash(parity_outcomes),
        "first_skip_hash": deterministic_frame_hash(skips, ["definition_id", "candidate_key"]),
        "second_skip_hash": deterministic_frame_hash(parity_skips, ["definition_id", "candidate_key"]),
        "first_exclusion_hash": deterministic_frame_hash(exclusions, ["definition_id", "candidate_key"]),
        "second_exclusion_hash": deterministic_frame_hash(parity_exclusions, ["definition_id", "candidate_key"]),
        "first_outcomes": len(outcomes), "second_outcomes": len(parity_outcomes),
        "first_skips": len(skips), "second_skips": len(parity_skips),
        "first_exclusions": len(exclusions), "second_exclusions": len(parity_exclusions),
    }
    parity["mismatch_count"] = int(
        parity["first_outcome_hash"] != parity["second_outcome_hash"]
        or parity["first_skip_hash"] != parity["second_skip_hash"]
        or parity["first_exclusion_hash"] != parity["second_exclusion_hash"]
    )
    base.write_csv(root / "audit/deterministic_rerun_parity.csv", [parity])
    base.write_csv(root / "audit/non_overlap_skip_ledger.csv", skips)
    base.write_csv(root / "audit/definition_outcome_exclusions.csv", exclusions)
    funding = lfbs.funding_panel()
    outcomes, boundaries = base.attach_long_costs(outcomes, funding, "event_id")
    base.write_csv(root / "materialized/event_ledger.csv", outcomes)
    accepted_candidates = candidates[candidates.candidate_key.isin(set(outcomes.candidate_key))].copy()
    base.write_json(root / "watch_status.json", {"status": "running", "stage": "control_key_construction", "accepted_trade_rows": len(outcomes), "rss_bytes": runner.current_rss_bytes(), "elapsed_seconds": time.monotonic()-started, "updated_ts": runner.utc_now()})
    control_keys, unavailable = base.build_control_keys(accepted_candidates, outcomes, feature_cache, sequence_cache, bars_cache, panel, ctx)
    control_freeze_hash = base.stable_hash(sorted(control_keys.control_key)) if len(control_keys) else base.stable_hash([])
    if len(control_keys):
        control_keys["control_key_freeze_hash"] = control_freeze_hash
    base.write_csv(root / "controls/control_key_manifest.csv", control_keys)
    base.write_csv(root / "controls/control_unavailable_reasons.csv", unavailable)
    base.write_json(root / "watch_status.json", {"status": "running", "stage": "control_outcome_simulation", "control_keys": len(control_keys), "rss_bytes": runner.current_rss_bytes(), "elapsed_seconds": time.monotonic()-started, "updated_ts": runner.utc_now()})
    control_outcomes: list[dict[str, Any]] = []
    control_exclusions: list[dict[str, Any]] = []
    for control in control_keys.to_dict("records"):
        event, excluded = base.execute_event(control, control["exit_policy"], bars_cache[control["symbol"]], feature_cache[control["symbol"]])
        if excluded:
            control_exclusions.append({**excluded, "control_key": control["control_key"]})
            continue
        assert event is not None
        event.update({"control_event_id": control["control_key"], "candidate_key": control["candidate_key"], "definition_id": control["definition_id"], "control_class": control["control_class"], "control_economic_address_hash": control["control_economic_address_hash"]})
        control_outcomes.append(event)
    controls = pd.DataFrame(control_outcomes)
    if len(controls):
        controls, control_boundaries = base.attach_long_costs(controls, funding, "control_event_id")
    else:
        control_boundaries = pd.DataFrame()
    base.write_csv(root / "controls/control_event_ledger.csv", controls)
    base.write_csv(root / "audit/control_outcome_exclusions.csv", control_exclusions)
    address_audit, control_summary = base.control_report(outcomes, controls)
    base.write_csv(root / "controls/control_address_audit.csv", address_audit)
    base.write_csv(root / "controls/control_summary.csv", control_summary)
    base.write_csv(root / "controls/risk_stable_control_diagnostics.csv", base.risk_stable_control_diagnostics(controls))
    summary, attribution, period = reports.summarize_economics(outcomes, definitions)
    base.write_csv(root / "economics/definition_summary.csv", summary)
    base.write_csv(root / "economics/cost_funding_attribution.csv", attribution)
    base.write_csv(root / "economics/period_summary.csv", period)
    concentration = lfbs.concentration_forensics(outcomes)
    base.write_csv(root / "forensics/concentration_and_removal.csv", concentration)
    neighborhood = outcomes.groupby(["flush_profile", "stabilization_bars", "parent_policy", "exit_policy"]).agg(events=("event_id", "size"), symbols=("symbol", "nunique"), months=("entry_ts", lambda x: pd.to_datetime(x, utc=True).dt.strftime("%Y-%m").nunique()), base_mean_R=("net_base_R", "mean"), conservative_mean_R=("net_conservative_R", "mean"), severe_mean_R=("net_severe_R", "mean")).reset_index()
    base.write_csv(root / "forensics/parameter_neighborhood.csv", neighborhood)
    base.write_csv(root / "forensics/exact_vs_imputed_funding.csv", reports.funding_partition_report(outcomes))
    base.write_csv(root / "forensics/horizon_path_behavior.csv", base.path_report(outcomes, bars_cache))
    decisions = base.decision_table(summary, concentration, control_summary, period, definitions)
    base.write_csv(root / "decision/candidate_decisions.csv", decisions)
    pairwise, _ = base.overlap_audits(candidates, outcomes, definitions)
    base.write_csv(root / "audit/pairwise_definition_overlap.csv", pairwise)
    interval_violations = []
    for label, (window_start, window_end) in EVALUATION_WINDOWS.items():
        interval_violations.extend(evidence.validate_evaluation_window_intervals(outcomes[outcomes.evaluation_period.eq(label)], window_start=window_start, window_end=window_end).violations)
    raw_drop_count = len(raw_boundary_drops)
    outcome_exclusion_count = len(exclusions)
    artificial_count = int(outcomes.artificial_horizon_exit.sum())
    boundary_rows = [
        {"scope": "raw_signal_boundary_drops", "count": raw_drop_count},
        {"scope": "definition_outcome_exclusions", "count": outcome_exclusion_count},
        {"scope": "accepted_natural_intervals_crossing_boundary", "count": len(interval_violations)},
        {"scope": "artificial_endpoint_exits", "count": artificial_count},
    ]
    base.write_csv(root / "audit/boundary_reconciliation.csv", boundary_rows)
    eligible_definition_rows = sum(len(candidates[candidates.selected_key_policy_hash.eq(row.selected_key_policy_hash)]) for row in definitions.itertuples(index=False))
    attrition_failure = int(eligible_definition_rows != len(outcomes) + len(skips) + len(exclusions))
    hard = {
        "definitions_evaluated": int(summary.definition_id.nunique()),
        "raw_policy_hashes": int(raw.raw_policy_hash.nunique()),
        "selected_key_policy_hashes": int(candidates.selected_key_policy_hash.nunique()),
        "raw_signal_duplicates": raw_duplicate_count,
        "strict_raw_nesting_failures": int((~nesting["pass"]).sum()),
        "known_regression_failures": int((~known["pass"]).sum()),
        "deterministic_rerun_mismatches": parity["mismatch_count"],
        "candidate_duplicate_addresses": int(outcomes.duplicated(["definition_id", "candidate_economic_address_hash"]).sum()),
        "unexplained_attrition": attrition_failure,
        "artificial_boundary_exits": artificial_count,
        "evaluation_interval_contract_violations": len(interval_violations),
        "funding_join_missing": int(boundaries.missing.sum()) if len(boundaries) else 0,
        "funding_join_duplicates": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0,
        "control_funding_join_missing": int(control_boundaries.missing.sum()) if len(control_boundaries) else 0,
        "control_funding_join_duplicates": int(control_boundaries.duplicated(["control_event_id", "boundary_ts"]).sum()) if len(control_boundaries) else 0,
        "decision_input_leaks": int((raw.feature_available_ts > raw.decision_ts).sum()) + (int((control_keys.feature_available_ts > control_keys.decision_ts).sum()) if len(control_keys) else 0),
        "protected_period_violations": int(outcomes.protected_violation.sum()),
        "placeholder_controls": int(control_keys.placeholder_control.sum()) if len(control_keys) else 0,
        "duplicate_controls_counted_independently": int(address_audit.duplicated_addresses_counted_independently.sum()) if len(address_audit) else 0,
        "risk_band_control_violations": int(((control_keys.risk_to_daily_atr < .25) | (control_keys.risk_to_daily_atr > 1.5) | (control_keys.risk_match_distance > base.RISK_MATCH_TOLERANCE_ATR+1e-12)).sum()) if len(control_keys) else 0,
        "actual_complement_accounting_failures": int(((control_summary.matched_count + control_summary.unmatched_count) != control_summary.full_count).sum()),
        "control_outcomes_accessed_before_freeze": int(control_keys.outcome_accessed_before_freeze.sum()) if len(control_keys) else 0,
        "imputed_funding_signal_activations": int(raw.imputed_funding_gate_activated.sum()),
        "frozen_manifest_hash_mismatches": int(base.file_hash(root / "manifest/delayed_flush_reclaim_definitions.csv") != base.file_hash(SOURCE_ROOT / "manifest/delayed_flush_reclaim_definitions.csv")),
        "source_root_content_mutations": int(directory_content_hash(SOURCE_ROOT) != source_tree_hash_before),
    }
    expected = {"definitions_evaluated": 24, "raw_policy_hashes": 4, "selected_key_policy_hashes": 8}
    base.write_csv(root / "audit/hard_gate_audit.csv", [{"gate": key, "value": value, "pass": value == expected.get(key, 0)} for key, value in hard.items()])
    gate_pass = all(value == expected.get(key, 0) for key, value in hard.items())
    old_new_comparison(root, raw, candidates, outcomes, definitions)
    final = "focused_mechanical_repair_required" if not gate_pass else "materialization_candidate" if (decisions.decision == "materialization_candidate").any() else "fragile_context_sleeve" if (decisions.decision == "fragile_context_sleeve").any() else "current_translation_weak"
    library = []
    for definition in definitions.itertuples(index=False):
        row = decisions[decisions.definition_id.eq(definition.definition_id)].iloc[0]
        library.append({
            "candidate_id": definition.definition_id, "candidate_definition_id": definition.definition_id, "definition_id": definition.definition_id,
            "hypothesis_id": "delayed_flush_reclaim_long", "family_engine_id": "kraken_dfrl_v1_repaired",
            "parameter_vector_hash": definition.parameter_vector_hash, "selected_key_policy_hash": definition.selected_key_policy_hash,
            "candidate_library_state": row.decision if gate_pass else "focused_mechanical_repair_required", "candidate_decision": row.decision if gate_pass else "focused_mechanical_repair_required",
            "evidence_level": "level_2_train_only_bounded_screen_capped", "evidence_level_contract": "train_only_not_validation_not_holdout_not_live",
            "clean_evidence_allowed": False, "evidence_cap_reason": "bar_based_forced_flow_proxy_no_liquidation_or_oi_confirmation_shared_funding_imputation_ohlcv_no_depth",
            "family_rejected": False, "train_only": True, "validation_run": False, "holdout_touched": False, "live_ready": False,
            "event_rows": row.events, "symbols": row.symbols, "base_mean_R": row.base_mean_R, "conservative_mean_R": row.conservative_mean_R, "severe_mean_R": row.severe_mean_R,
            "source_run_root": str(root), "contract_version": CONTRACT_VERSION,
        })
    base.write_csv(root / "candidate_library/delayed_flush_reclaim_candidate_library_update.csv", library)
    data_manifest = Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv")
    funding_manifest = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1/funding/shared_funding_panel_manifest.csv")
    repro = {
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_path": str(Path(__file__)), "code_hash": base.file_hash(Path(__file__)),
        "config_hash": base.file_hash(root / "manifest/delayed_flush_reclaim_definitions.csv"), "contract_hash": base.file_hash(contract_path),
        "data_snapshot_manifest_hash": base.file_hash(data_manifest), "pit_universe_manifest_hash": base.file_hash(root / "manifest/pit_panel.csv"),
        "funding_manifest_hash": base.file_hash(funding_manifest), "blocked_source_decision_hash": source_decision_hash_before,
        "blocked_source_tree_hash": source_tree_hash_before,
        "protected_boundary": PROTECTED.isoformat(), "seed_values": [], "contract_type": "Kraken PF perpetual instruments; R-normalized OHLCV screen",
    }
    base.write_json(root / "reproducibility/run_manifest.json", repro)
    source_decision_hash_after = base.file_hash(SOURCE_ROOT / "decision_summary.json")
    source_tree_hash_after = directory_content_hash(SOURCE_ROOT)
    decision = {
        "run_root": str(root), "status": "complete" if gate_pass else "blocked_by_protocol_issue", "final_decision": final, **hard,
        "raw_signals": len(raw), "parent_filtered_signals": len(candidates), "accepted_trade_rows": len(outcomes), "overlap_skips": len(skips),
        "definition_outcome_exclusions": len(exclusions), "control_event_rows": len(controls), "raw_signal_freeze_hash": raw_freeze_hash,
        "selected_key_freeze_hash": parent_freeze_hash, "control_key_freeze_hash": control_freeze_hash,
        "source_root_preserved_unchanged": source_decision_hash_before == source_decision_hash_after and source_tree_hash_before == source_tree_hash_after,
        "source_root_status": "blocked_by_protocol_issue", "source_root_evidence": "diagnostic_only_provenance",
        "peak_rss_bytes": peak_rss, "runtime_seconds": time.monotonic()-started,
        "raw_signal_boundary_drops": raw_drop_count, "accepted_natural_intervals_crossing_boundary": len(interval_violations),
        "artificial_endpoint_exits": artificial_count,
        "materialization_candidates": decisions[decisions.decision.eq("materialization_candidate")].definition_id.tolist() if gate_pass else [],
        "context_sleeves": decisions[decisions.decision.eq("fragile_context_sleeve")].definition_id.tolist() if gate_pass else [],
        "validation_launched": False, "cpcv_launched": False, "holdout_launched": False, "portfolio_construction_launched": False, "live_work_launched": False,
        "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    base.write_json(root / "decision_summary.json", decision)
    compact_bundle(root)
    base.write_json(root / "watch_status.json", {**decision, "stage": "complete", "updated_ts": runner.utc_now()})
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    result = run(Path(args.run_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
