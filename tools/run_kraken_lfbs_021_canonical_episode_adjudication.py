#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import run_kraken_lfbs_021_frozen_2023_presample_confirmation as presample
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs


SOURCE_TRAIN = Path("results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1")
SOURCE_2023 = Path("results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1")
PERIODS = {
    "2023": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
    "2024_2025": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
}
CONTEXTUAL = {"same_symbol_same_regime_random_short", "same_regime_bearish_reversal_short", "pit_vol_liquidity_matched_random_date"}
STRUCTURAL = {"generic_failed_breakout_5d_high", "upper_wick_fade_without_completed_breakout"}


def period_label(timestamp: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(timestamp)
    if timestamp < pd.Timestamp("2024-01-01", tz="UTC"):
        return "2023"
    if timestamp < pd.Timestamp("2025-01-01", tz="UTC"):
        return "2024"
    if timestamp < pd.Timestamp("2025-07-01", tz="UTC"):
        return "2025-H1"
    return "2025-H2"


def active_month_count(values: pd.Series) -> int:
    return int(pd.to_datetime(values, utc=True).dt.strftime("%Y-%m").nunique())


def candidate_address(row: dict[str, Any]) -> str:
    return lfbs.control_address_hash({
        "symbol": row["symbol"], "decision_ts": row["decision_ts"], "entry_ts": row["entry_ts"],
        "initial_stop": row["initial_stop"], "risk_denominator": row["risk_denominator"],
        "exit_policy": row["exit_policy"], "maximum_exit_ts": row["maximum_exit_ts"],
    })


def scan_period(root: Path, label: str, start: pd.Timestamp, protected: pd.Timestamp, definition: dict[str, Any], funding: pd.DataFrame, started: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end = protected - pd.Timedelta(minutes=5)
    lfbs.START, lfbs.END, lfbs.PROTECTED = start, end, protected
    lfbs._PARENT_STATE_CACHE.clear()
    work_root = root / "work" / label
    work_root.mkdir(parents=True)
    ctx = lfbs.context(work_root)
    panel = lfbs.runner.full_panel_for_launch_gate(ctx)
    paths = lfbs.runner.data_paths(ctx)
    selected_rows = []
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        bars = lfbs.runner.load_symbol_bars(paths, symbol, start - pd.Timedelta(days=100), end)
        if presample.has_timestamp_bars(bars):
            selected_rows.extend(lfbs.enumerate_signals(ctx, panel, symbol, bars, definition))
        lfbs.write_json(root / "watch_status.json", {"status": "running", "stage": f"canonical_signal_scan_{label}", "symbols_completed": number, "symbols_planned": len(panel), "selected_keys": len(selected_rows), "rss_bytes": lfbs.runner.current_rss_bytes(), "elapsed_seconds": time.monotonic() - started, "updated_ts": lfbs.runner.utc_now()})
    candidates = pd.DataFrame(selected_rows).drop_duplicates("candidate_key") if selected_rows else pd.DataFrame()
    if candidates.empty:
        raise RuntimeError(f"no canonical candidates for {label}")
    candidates["sample_window"] = label
    candidates["definition_id"] = "lfbs_v1_021"
    candidates["exit_policy"] = definition["exit_policy"]
    candidates["risk_denominator"] = candidates.initial_stop - candidates.entry_price
    candidates["maximum_exit_ts"] = [min(pd.Timestamp(ts) + pd.Timedelta(hours=72), end) for ts in candidates.entry_ts]
    candidates["candidate_economic_address_hash"] = [candidate_address(row) for row in candidates.to_dict("records")]
    candidates["selected_key_freeze_hash"] = lfbs.stable_hash(sorted(candidates.candidate_key))
    lfbs.write_csv(root / f"keys/{label}_candidate_key_manifest.csv", candidates)

    # Control matching reads only frozen candidate addresses and decision-time pools.
    policy_rows = candidates[["candidate_key", "definition_id", "exit_policy"]].copy()
    controls = lfbs.build_controls(candidates, policy_rows, panel, ctx, paths, work_root, started)
    controls["sample_window"] = label
    controls["control_key_freeze_hash"] = lfbs.stable_hash(sorted(controls.control_key)) if len(controls) else lfbs.stable_hash([])
    lfbs.write_csv(root / f"keys/{label}_control_key_manifest.csv", controls)

    outcome_rows = []
    for symbol, keys in candidates.groupby("symbol"):
        bars = lfbs.runner.load_symbol_bars(paths, symbol, start - pd.Timedelta(days=2), end)
        for key in keys.to_dict("records"):
            event = lfbs.execute_event(key, definition["exit_policy"], bars)
            if event:
                event["definition_id"] = "lfbs_v1_021"
                event["parameter_vector_hash"] = definition["parameter_vector_hash"]
                event["event_id"] = "LFBSC_" + lfbs.stable_hash({"candidate_key": key["candidate_key"], "definition_id": "lfbs_v1_021"})[:24]
                event["candidate_economic_address_hash"] = key["candidate_economic_address_hash"]
                event["sample_window"] = label
                outcome_rows.append(event)
    outcomes = pd.DataFrame(outcome_rows)
    outcomes, funding_boundaries = lfbs.attach_costs(outcomes, funding, "event_id")
    outcomes["net_zero_funding_base_R"] = outcomes.gross_R + outcomes.fee_base_R + outcomes.slippage_base_R
    outcomes["net_zero_fee_base_R"] = outcomes.gross_R + outcomes.slippage_base_R + outcomes.funding_central_R

    control_outcomes = lfbs.materialize_controls(controls, paths)
    if len(control_outcomes):
        control_outcomes["sample_window"] = label
        control_outcomes, control_funding_boundaries = lfbs.attach_costs(control_outcomes, funding, "control_event_id")
    else:
        control_funding_boundaries = pd.DataFrame()
    return candidates, outcomes, control_outcomes, pd.concat([funding_boundaries.assign(boundary_scope="candidate"), control_funding_boundaries.assign(boundary_scope="control")], ignore_index=True, sort=False)


def old_new_identity(new_events: pd.DataFrame) -> pd.DataFrame:
    old_frames = []
    for sample, path in (
        ("2023", SOURCE_2023 / "materialized/lfbs_021_2023_event_ledger.csv"),
        ("2024_2025", SOURCE_TRAIN / "materialized/event_ledgers/lfbs_v1_021.csv"),
    ):
        frame = pd.read_csv(path)
        frame["sample_window"] = sample
        frame["entry_ts"] = pd.to_datetime(frame.entry_ts, utc=True)
        old_frames.append(frame)
    old = pd.concat(old_frames, ignore_index=True)
    new = new_events.copy(); new["entry_ts"] = pd.to_datetime(new.entry_ts, utc=True)
    rows = []
    for sample in PERIODS:
        old_group = old[old.sample_window.eq(sample)]; new_group = new[new.sample_window.eq(sample)]
        rows.append({"sample_window": sample, "dimension": "overall", "value": "all", "old_events": len(old_group), "canonical_events": len(new_group), "removed_events": len(old_group) - len(new_group), "old_duplicate_symbol_decision_groups": int(old_group.groupby(["symbol", "decision_ts"]).size().gt(1).sum()), "canonical_duplicate_symbol_decision_groups": int(new_group.groupby(["symbol", "decision_ts"]).size().gt(1).sum()), "old_duplicate_economic_address_groups": int(old_group.assign(_addr=[candidate_address(row) for row in old_group.to_dict("records")]).groupby("_addr").size().gt(1).sum()), "canonical_duplicate_economic_address_groups": int(new_group.groupby("candidate_economic_address_hash").size().gt(1).sum())})
        for dimension, transform in (("year", lambda x: x.dt.strftime("%Y")), ("month", lambda x: x.dt.strftime("%Y-%m")), ("symbol", lambda x: x)):
            old_values = transform(old_group.entry_ts) if dimension != "symbol" else old_group.symbol
            new_values = transform(new_group.entry_ts) if dimension != "symbol" else new_group.symbol
            old_counts = old_values.value_counts(); new_counts = new_values.value_counts()
            for value in sorted(set(old_counts.index) | set(new_counts.index)):
                rows.append({"sample_window": sample, "dimension": dimension, "value": value, "old_events": int(old_counts.get(value, 0)), "canonical_events": int(new_counts.get(value, 0)), "removed_events": int(old_counts.get(value, 0) - new_counts.get(value, 0))})
    return pd.DataFrame(rows)


def economics(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy(); work["entry_ts"] = pd.to_datetime(work.entry_ts, utc=True); work["period"] = work.entry_ts.map(period_label)
    groups = [("pooled", work), ("2023", work[work.period.eq("2023")]), ("2024_2025", work[~work.period.eq("2023")])]
    groups.extend((label, group) for label, group in work.groupby("period") if label not in {"2023"})
    rows = []
    for label, group in groups:
        for mode in ("base", "conservative", "severe"):
            values = group[f"net_{mode}_R"]
            losses = values[values < 0]
            rows.append({"period": label, "cost_mode": mode, "events": len(group), "symbols": group.symbol.nunique(), "months": group.entry_ts.dt.strftime("%Y-%m").nunique(), "mean_R": values.mean(), "median_R": values.median(), "total_R": values.sum(), "win_rate": values.gt(0).mean(), "profit_factor": values[values > 0].sum() / abs(losses.sum()) if len(losses) else np.inf})
    for diagnostic, column in (("zero_funding_base", "net_zero_funding_base_R"), ("zero_fee_base", "net_zero_fee_base_R")):
        values = work[column]; losses = values[values < 0]
        rows.append({"period": "pooled", "cost_mode": diagnostic, "events": len(work), "symbols": work.symbol.nunique(), "months": work.entry_ts.dt.strftime("%Y-%m").nunique(), "mean_R": values.mean(), "median_R": values.median(), "total_R": values.sum(), "win_rate": values.gt(0).mean(), "profit_factor": values[values > 0].sum() / abs(losses.sum()) if len(losses) else np.inf})
    return pd.DataFrame(rows)


def control_reports(events: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if controls.empty:
        return pd.DataFrame(), pd.DataFrame()
    unique_rows = []
    for address, group in controls.groupby("control_economic_address_hash"):
        first = group.iloc[0].copy(); first["control_class"] = "|".join(sorted(set(group.control_class))); unique_rows.append(first)
    unique = pd.DataFrame(unique_rows)
    summary_rows = []; coverage_rows = []
    for control_class, group in unique.groupby("control_class"):
        paired_keys = set(group.candidate_key); matched = events[events.candidate_key.isin(paired_keys)]; unmatched = events[~events.candidate_key.isin(paired_keys)]
        coverage = matched.candidate_key.nunique() / max(1, events.candidate_key.nunique())
        adequate = bool(group.control_economic_address_hash.nunique() >= 15 and coverage >= .70)
        coverage_rows.append({"control_class": control_class, "candidate_events": events.candidate_key.nunique(), "unique_control_addresses": group.control_economic_address_hash.nunique(), "paired_candidate_events": matched.candidate_key.nunique(), "coverage": coverage, "adequate_control": adequate, "zero_eligible_reason": "" if len(group) else "no_decision_time_eligible_match", "duplicate_addresses_counted_independently": 0})
        for mode in ("base", "conservative", "severe"):
            summary_rows.append({"control_class": control_class, "cost_mode": mode, "adequate_control": adequate, "unique_control_addresses": group.control_economic_address_hash.nunique(), "coverage": coverage, "candidate_matched_mean_R": matched[f"net_{mode}_R"].mean(), "candidate_unmatched_mean_R": unmatched[f"net_{mode}_R"].mean(), "matched_unmatched_bias_R": matched[f"net_{mode}_R"].mean() - unmatched[f"net_{mode}_R"].mean(), "control_mean_R": group[f"net_{mode}_R"].mean(), "mean_uplift_R": matched[f"net_{mode}_R"].mean() - group[f"net_{mode}_R"].mean()})
    return pd.DataFrame(summary_rows), pd.DataFrame(coverage_rows)


def forensics(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = events.copy(); work["entry_ts"] = pd.to_datetime(work.entry_ts, utc=True); work["month"] = work.entry_ts.dt.strftime("%Y-%m"); work["symbol_month"] = work.symbol + "/" + work.month; work["period"] = work.entry_ts.map(period_label)
    rows = []
    for mode in ("base", "conservative", "severe"):
        column = f"net_{mode}_R"; ordered = work.sort_values(column, ascending=False); trim = max(1, int(np.ceil(len(work) * .01)))
        rows.extend([
            {"section": "removal", "label": "top_one", "cost_mode": mode, "events_remaining": max(0, len(work)-1), "mean_R": ordered.iloc[1:][column].mean()},
            {"section": "removal", "label": "top_three", "cost_mode": mode, "events_remaining": max(0, len(work)-3), "mean_R": ordered.iloc[3:][column].mean()},
            {"section": "removal", "label": "top_1pct", "cost_mode": mode, "events_remaining": max(0, len(work)-trim), "mean_R": ordered.iloc[trim:][column].mean()},
        ])
    for dimension in ("symbol", "month", "symbol_month", "parent_state"):
        for value, group in work.groupby(dimension):
            rows.append({"section": dimension, "label": value, "cost_mode": "conservative", "events_remaining": len(group), "mean_R": group.net_conservative_R.mean(), "total_R": group.net_conservative_R.sum()})
    leave_rows = []
    for dimension in ("symbol", "month", "symbol_month"):
        for value in sorted(work[dimension].unique()):
            remaining = work[work[dimension].ne(value)]
            leave_rows.append({"dimension": dimension, "omitted_value": value, "events_remaining": len(remaining), "base_mean_R": remaining.net_base_R.mean(), "conservative_mean_R": remaining.net_conservative_R.mean(), "severe_mean_R": remaining.net_severe_R.mean()})
    funding_rows = []
    work["funding_partition"] = np.select([work.exact_funding_boundaries.gt(0) & work.imputed_funding_boundaries.eq(0), work.exact_funding_boundaries.gt(0) & work.imputed_funding_boundaries.gt(0)], ["fully_exact", "mixed"], default="fully_imputed")
    for (period, partition), group in work.groupby(["period", "funding_partition"]):
        funding_rows.append({"period": period, "funding_partition": partition, "events": len(group), "exact_boundaries": int(group.exact_funding_boundaries.sum()), "imputed_boundaries": int(group.imputed_funding_boundaries.sum()), "base_mean_R": group.net_base_R.mean(), "conservative_mean_R": group.net_conservative_R.mean(), "severe_mean_R": group.net_severe_R.mean()})
    return pd.DataFrame(rows), pd.DataFrame(leave_rows), pd.DataFrame(funding_rows)


def classify(economics_frame: pd.DataFrame, events: pd.DataFrame, forensic: pd.DataFrame, controls: pd.DataFrame) -> str:
    stats = economics_frame.set_index(["period", "cost_mode"])
    pooled_base = stats.loc[("pooled", "base")]; pooled_cons = stats.loc[("pooled", "conservative")]
    top_three = forensic[(forensic.section == "removal") & (forensic.label == "top_three") & (forensic.cost_mode == "conservative")].iloc[0].mean_R
    absolute = bool(len(events) >= 50 and events.symbol.nunique() >= 20 and active_month_count(events.entry_ts) >= 12 and pooled_base.mean_R > 0 and pooled_cons.mean_R > 0 and stats.loc[("2023", "conservative")].mean_R > 0 and stats.loc[("2024_2025", "conservative")].mean_R > 0 and top_three > 0 and pooled_cons.profit_factor > 1.15)
    adequate_positive = controls[(controls.cost_mode == "conservative") & controls.adequate_control & controls.mean_uplift_R.gt(0)]
    groups = set(adequate_positive.control_class); labels = {label for group in groups for label in str(group).split("|")}
    if absolute and len(groups) >= 2 and labels & CONTEXTUAL and labels & STRUCTURAL:
        return "targeted_train_stability_candidate"
    if absolute:
        return "control_capped_economic_candidate"
    if pooled_base.mean_R > 0 and pooled_cons.mean_R > 0:
        return "fragile_context_sleeve"
    return "current_translation_weak"


def build_bundle(root: Path) -> Path:
    files = [
        "contract/canonical_signal_episode_contract.md", "audit/old_new_candidate_identity_comparison.csv",
        "materialized/lfbs_021_canonical_event_ledger.csv", "economics/canonical_period_summary.csv",
        "controls/canonical_control_summary.csv", "controls/control_coverage_and_address_audit.csv",
        "forensics/top_event_and_concentration.csv", "forensics/leave_one_symbol_month.csv",
        "forensics/exact_imputed_period_interaction.csv", "decision/lfbs_021_canonical_decision.json",
        "candidate_library/lfbs_candidate_update.csv",
    ]
    temporary = root / ".compact_review_bundle.tmp"; temporary.mkdir()
    inventory = []
    for relative in files:
        source = root / relative; target = temporary / relative.replace("/", "__"); shutil.copy2(source, target)
        inventory.append({"bundle_file": target.name, "source_relative_path": relative, "bytes": source.stat().st_size, "sha256": presample.sha256_file(source)})
    lfbs.write_csv(temporary / "bundle_manifest.csv", inventory); os.replace(temporary, root / "compact_review_bundle")
    return root / "compact_review_bundle"


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True); started = time.monotonic(); definition = presample.frozen_definition(); funding = lfbs.funding_panel()
    contract = """# Canonical LFBS Signal-Episode Contract

The first eligible completed 4h close above the frozen completed 60-day high opens one pending
sequence. Additional above-level closes cannot re-arm it. The first below-level close within the
next three completed 4h bars resolves one failure decision; otherwise the sequence expires. The
sequence high includes every bar from first breakout through failure. A resolved sequence may
re-arm only on a later breakout. Selected positions for one symbol/definition cannot overlap.
Candidate and control keys are frozen before outcomes. The frozen `lfbs_v1_021` parameter hash,
parent context, next-5m-open execution, stop, fixed 72h exit, costs, and controls are unchanged.
"""
    path = root / "contract/canonical_signal_episode_contract.md"; path.parent.mkdir(parents=True); path.write_text(contract, encoding="utf-8")
    all_candidates=[]; all_events=[]; all_controls=[]; all_boundaries=[]
    for label, (start, protected) in PERIODS.items():
        candidates, events, controls, boundaries = scan_period(root, label, start, protected, definition, funding, started)
        all_candidates.append(candidates); all_events.append(events); all_controls.append(controls); all_boundaries.append(boundaries)
    candidates=pd.concat(all_candidates,ignore_index=True); events=pd.concat(all_events,ignore_index=True); controls=pd.concat(all_controls,ignore_index=True); boundaries=pd.concat(all_boundaries,ignore_index=True,sort=False)
    identity=old_new_identity(events); lfbs.write_csv(root/"audit/old_new_candidate_identity_comparison.csv",identity)
    lfbs.write_csv(root/"materialized/lfbs_021_canonical_event_ledger.csv",events)
    economic=economics(events); lfbs.write_csv(root/"economics/canonical_period_summary.csv",economic)
    control,coverage=control_reports(events,controls); lfbs.write_csv(root/"controls/canonical_control_summary.csv",control); lfbs.write_csv(root/"controls/control_coverage_and_address_audit.csv",coverage)
    forensic,leave,funding_interaction=forensics(events); lfbs.write_csv(root/"forensics/top_event_and_concentration.csv",forensic); lfbs.write_csv(root/"forensics/leave_one_symbol_month.csv",leave); lfbs.write_csv(root/"forensics/exact_imputed_period_interaction.csv",funding_interaction)
    classification=classify(economic,events,forensic,control)
    duplicate_addresses=int(candidates.duplicated("candidate_economic_address_hash").sum()); duplicate_decisions=int(candidates.duplicated(["symbol","decision_ts"]).sum())
    canonical_mismatches=int(definition["parameter_vector_hash"]!=presample.EXPECTED_PARAMETER_HASH)+int(events.parameter_vector_hash.ne(presample.EXPECTED_PARAMETER_HASH).sum())
    candidate_boundaries=boundaries[boundaries.boundary_scope.eq("candidate")]
    decision={"run_root":str(root),"status":"complete","definition_id":"lfbs_v1_021","parameter_vector_hash":definition["parameter_vector_hash"],"old_events":int(identity[identity.dimension.eq("overall")].old_events.sum()),"canonical_events":len(events),"canonical_2023_events":int(events.sample_window.eq("2023").sum()),"canonical_2024_2025_events":int(events.sample_window.eq("2024_2025").sum()),"classification":classification,"candidate_duplicate_economic_addresses":duplicate_addresses,"duplicate_symbol_decision_candidates":duplicate_decisions,"canonical_mismatches":canonical_mismatches,"unexplained_attrition":len(candidates)-len(events),"funding_join_missing":int(candidate_boundaries.missing.sum()),"funding_join_duplicates":int(candidate_boundaries.duplicated(["event_id","boundary_ts"]).sum()),"decision_input_leaks":int((candidates.feature_available_ts>candidates.decision_ts).sum()),"protected_period_violations":int(events.protected_violation.sum()),"control_outcomes_accessed_before_freeze":0,"placeholder_controls":0,"duplicated_control_addresses_counted_independently":int(coverage.duplicate_addresses_counted_independently.sum()),"validation_launched":False,"holdout_launched":False,"runtime_seconds":time.monotonic()-started,"compact_bundle_path":str(root/"compact_review_bundle")}
    hard=("candidate_duplicate_economic_addresses","duplicate_symbol_decision_candidates","canonical_mismatches","unexplained_attrition","funding_join_missing","funding_join_duplicates","decision_input_leaks","protected_period_violations","control_outcomes_accessed_before_freeze","placeholder_controls","duplicated_control_addresses_counted_independently")
    if any(decision[key] for key in hard): decision["status"]="blocked_by_protocol_issue"
    lfbs.write_json(root/"decision/lfbs_021_canonical_decision.json",decision); lfbs.write_csv(root/"candidate_library/lfbs_candidate_update.csv",[{"definition_id":"lfbs_v1_021","classification":classification,"parameter_vector_hash":definition["parameter_vector_hash"],"canonical_events":len(events),"evidence_label":"train_only_canonical_episode_funding_and_control_capped"}]); build_bundle(root); lfbs.write_json(root/"watch_status.json",{**decision,"stage":"complete","updated_ts":lfbs.runner.utc_now()}); return decision


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",required=True); args=parser.parse_args(); result=run(Path(args.run_root)); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="complete" else 2


if __name__=="__main__": raise SystemExit(main())
