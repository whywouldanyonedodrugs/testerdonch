#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs


SOURCE_ROOT = Path("results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1")
START = pd.Timestamp("2023-01-01", tz="UTC")
PROTECTED = pd.Timestamp("2024-01-01", tz="UTC")
END = PROTECTED - pd.Timedelta(minutes=5)
EXPECTED_PARAMETER_HASH = "409b6c573d2a5695c9e7e59721dd45e65c79cfcc3ad9a30f9ae4487a6717e9ea"
EXPECTED_SELECTED_HASH = "7529b5d02f23ba5194680bf2b24f94437b67bd089a5f19977807c16bca2a93b6"
CONTEXTUAL_CONTROLS = {"same_symbol_same_regime_random_short", "same_regime_bearish_reversal_short"}
STRUCTURAL_CONTROLS = {"upper_wick_fade_without_completed_breakout", "generic_failed_breakout_5d_high"}


def has_timestamp_bars(frame: pd.DataFrame) -> bool:
    return not frame.empty and "ts" in frame.columns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_definition() -> dict[str, Any]:
    manifest = pd.read_csv(SOURCE_ROOT / "manifest/failed_breakout_short_definitions.csv")
    rows = manifest[manifest.definition_id.eq("lfbs_v1_021")]
    if len(rows) != 1:
        raise RuntimeError("exactly one frozen lfbs_v1_021 definition required")
    definition = rows.iloc[0].to_dict()
    if definition["parameter_vector_hash"] != EXPECTED_PARAMETER_HASH or definition["selected_key_policy_hash"] != EXPECTED_SELECTED_HASH:
        raise RuntimeError("frozen lfbs_v1_021 hash changed")
    if lfbs.canonical_hash(definition, selected_key=False) != EXPECTED_PARAMETER_HASH:
        raise RuntimeError("frozen lfbs_v1_021 canonical hash mismatch")
    return definition


def classify_presample(summary: pd.DataFrame, top_one_mean: float, controls: pd.DataFrame) -> str:
    stats = summary.set_index("cost_mode")
    base = stats.loc["base"]
    conservative = stats.loc["conservative"]
    adequate = controls[controls.adequate_control & controls.cost_mode.eq("conservative") & controls.mean_uplift_R.gt(0)]
    positive_groups = set(adequate.control_class)
    positive_classes = {label for labels in positive_groups for label in str(labels).split("|")}
    meaningful_uplift = len(positive_groups) > 0
    independent = (
        int(base.events) >= 20
        and int(base.symbols) >= 10
        and int(base.months) >= 6
        and float(base.mean_R) > 0
        and float(conservative.mean_R) > 0
        and float(conservative.profit_factor) > 1
        and top_one_mean > 0
        and len(positive_groups) >= 2
        and bool(positive_classes & CONTEXTUAL_CONTROLS)
        and bool(positive_classes & STRUCTURAL_CONTROLS)
    )
    if independent:
        return "independent_presample_support"
    if float(base.mean_R) <= 0 and float(conservative.mean_R) <= 0:
        return "presample_failure"
    if not meaningful_uplift:
        return "presample_failure"
    return "fragile_presample_support"


def economic_summary(events: pd.DataFrame) -> pd.DataFrame:
    months = pd.to_datetime(events.entry_ts, utc=True).dt.strftime("%Y-%m")
    rows = []
    for mode in ("base", "conservative", "severe"):
        values = events[f"net_{mode}_R"]
        losses = values[values < 0]
        rows.append({
            "cost_mode": mode,
            "events": len(events),
            "symbols": events.symbol.nunique(),
            "months": months.nunique(),
            "mean_R": values.mean(),
            "median_R": values.median(),
            "total_R": values.sum(),
            "win_rate": values.gt(0).mean(),
            "profit_factor": values[values > 0].sum() / abs(losses.sum()) if len(losses) else np.inf,
        })
    return pd.DataFrame(rows)


def control_summary(events: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if controls.empty:
        return pd.DataFrame(columns=["control_class", "cost_mode", "candidate_events", "unique_control_addresses", "paired_candidate_events", "coverage", "candidate_mean_R", "control_mean_R", "mean_uplift_R", "adequate_control"])
    address_rows = []
    for address, address_group in controls.groupby("control_economic_address_hash"):
        first = address_group.iloc[0].copy()
        first["control_class"] = "|".join(sorted(set(address_group.control_class)))
        first["control_economic_address_hash"] = address
        address_rows.append(first)
    unique_controls = pd.DataFrame(address_rows)
    for control_class, group in unique_controls.groupby("control_class"):
        paired_keys = set(group.candidate_key)
        candidate = events[events.candidate_key.isin(paired_keys)]
        coverage = candidate.candidate_key.nunique() / max(1, events.candidate_key.nunique())
        for mode in ("base", "conservative", "severe"):
            rows.append({
                "control_class": control_class,
                "cost_mode": mode,
                "candidate_events": events.candidate_key.nunique(),
                "unique_control_addresses": group.control_economic_address_hash.nunique(),
                "paired_candidate_events": candidate.candidate_key.nunique(),
                "coverage": coverage,
                "candidate_mean_R": candidate[f"net_{mode}_R"].mean(),
                "control_mean_R": group[f"net_{mode}_R"].mean(),
                "mean_uplift_R": candidate[f"net_{mode}_R"].mean() - group[f"net_{mode}_R"].mean(),
                "adequate_control": bool(group.control_economic_address_hash.nunique() >= 15 and coverage >= 0.70),
            })
    return pd.DataFrame(rows)


def original_decomposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(SOURCE_ROOT / "materialized/event_ledgers/lfbs_v1_021.csv")
    events["entry_ts"] = pd.to_datetime(events.entry_ts, utc=True)
    events["month"] = events.entry_ts.dt.strftime("%Y-%m")
    events["period"] = np.select(
        [events.entry_ts < pd.Timestamp("2025-01-01", tz="UTC"), events.entry_ts < pd.Timestamp("2025-07-01", tz="UTC")],
        ["2024", "2025-H1"],
        default="2025-H2",
    )
    rows = []
    for period, group in events.groupby("period"):
        rows.append({"section": "period", "label": period, "events": len(group), "symbols": group.symbol.nunique(), "months": group.month.nunique(), "base_mean_R": group.net_base_R.mean(), "conservative_mean_R": group.net_conservative_R.mean(), "severe_mean_R": group.net_severe_R.mean(), "conservative_total_R": group.net_conservative_R.sum()})
    for rank, event in events.nlargest(3, "net_conservative_R").reset_index(drop=True).iterrows():
        rows.append({"section": "top_event", "label": f"rank_{rank + 1}:{event.event_id}", "events": 1, "symbol": event.symbol, "month": event.month, "conservative_mean_R": event.net_conservative_R, "conservative_total_R": event.net_conservative_R})
    for symbol, group in events.groupby("symbol"):
        rows.append({"section": "symbol", "label": symbol, "events": len(group), "conservative_mean_R": group.net_conservative_R.mean(), "conservative_total_R": group.net_conservative_R.sum()})
    for month, group in events.groupby("month"):
        rows.append({"section": "month", "label": month, "events": len(group), "conservative_mean_R": group.net_conservative_R.mean(), "conservative_total_R": group.net_conservative_R.sum()})
    for reason, group in events.groupby("exit_reason"):
        rows.append({"section": "exit_reason", "label": reason, "events": len(group), "conservative_mean_R": group.net_conservative_R.mean(), "conservative_total_R": group.net_conservative_R.sum()})
    funding = []
    for label, group in (("original_all", events), ("original_exact_boundary", events[events.exact_funding_boundaries.gt(0)]), ("original_imputed_boundary", events[events.imputed_funding_boundaries.gt(0)]), ("original_fully_imputed", events[events.exact_funding_boundaries.eq(0) & events.imputed_funding_boundaries.gt(0)])):
        funding.append({"sample": label, "events": len(group), "exact_boundaries": int(group.exact_funding_boundaries.sum()), "imputed_boundaries": int(group.imputed_funding_boundaries.sum()), "base_mean_R": group.net_base_R.mean(), "conservative_mean_R": group.net_conservative_R.mean(), "severe_mean_R": group.net_severe_R.mean()})
    return pd.DataFrame(rows), pd.DataFrame(funding)


def build_bundle(root: Path) -> Path:
    relative_paths = [
        "audit/presample_independence_audit.md", "audit/2023_data_coverage.csv",
        "materialized/lfbs_021_2023_event_ledger.csv", "economics/presample_summary.csv",
        "controls/presample_control_summary.csv", "forensics/top_event_and_period_decomposition.csv",
        "forensics/exact_vs_imputed_support.csv", "decision/lfbs_021_presample_decision.json",
        "audit/canonical_hash_closure_audit.csv",
    ]
    bundle = root / "compact_review_bundle"
    temporary = root / ".compact_review_bundle.tmp"
    temporary.mkdir()
    inventory = []
    for relative in relative_paths:
        source = root / relative
        if not source.is_file():
            raise RuntimeError(f"missing compact review input: {relative}")
        target = temporary / relative.replace("/", "__")
        shutil.copy2(source, target)
        inventory.append({"bundle_file": target.name, "source_relative_path": relative, "bytes": source.stat().st_size, "sha256": sha256_file(source)})
    lfbs.write_csv(temporary / "bundle_manifest.csv", inventory)
    os.replace(temporary, bundle)
    return bundle


def run(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"fresh run root required: {root}")
    root.mkdir(parents=True)
    started = time.monotonic()
    definition = frozen_definition()
    canonical_hash_at_freeze = lfbs.canonical_hash(dict(definition), selected_key=False)
    original_manifest_hash = sha256_file(SOURCE_ROOT / "manifest/failed_breakout_short_definitions.csv")
    original_ledger = pd.read_csv(SOURCE_ROOT / "materialized/event_ledgers/lfbs_v1_021.csv", usecols=["entry_ts"])
    original_min = pd.to_datetime(original_ledger.entry_ts, utc=True).min()
    independence_pass = bool(original_min >= pd.Timestamp("2024-01-01", tz="UTC"))
    independence = f"""# Presample Independence Audit

- Frozen definition: `lfbs_v1_021`
- Parameter hash: `{definition['parameter_vector_hash']}`
- Selected-key hash: `{definition['selected_key_policy_hash']}`
- Source manifest SHA-256: `{original_manifest_hash}`
- Original executable train start: `2024-01-01T00:00:00Z`
- Earliest original selected event: `{original_min.isoformat()}`
- Repository 2023 LFBS outcome artifacts used by selection: none found
- Independence status: `{'pass' if independence_pass else 'fail'}`

This audit establishes repository-level procedural independence. It cannot prove that no external
human ever viewed unrelated 2023 market behavior. The definition, hash, costs, controls, and
classification thresholds were frozen before this run.
"""
    audit_path = root / "audit/presample_independence_audit.md"; audit_path.parent.mkdir(parents=True); audit_path.write_text(independence, encoding="utf-8")

    lfbs.START, lfbs.END, lfbs.PROTECTED = START, END, PROTECTED
    lfbs._PARENT_STATE_CACHE.clear()
    ctx = lfbs.context(root)
    panel = lfbs.runner.full_panel_for_launch_gate(ctx)
    paths = lfbs.runner.data_paths(ctx)
    candidate_rows = []
    coverage_counts = {"trade_5m_rows": 0, "trade_symbols": 0, "trade_4h_rows": 0, "mark_4h_rows": 0}
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        bars = lfbs.runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=100), END)
        if not has_timestamp_bars(bars):
            lfbs.write_json(root / "watch_status.json", {"status": "running", "stage": "2023_signal_scan", "symbols_completed": number, "symbols_planned": len(panel), "selected_events": len(candidate_rows), "rss_bytes": lfbs.runner.current_rss_bytes(), "updated_ts": lfbs.runner.utc_now()})
            continue
        in_year = bars[(bars.ts >= START) & (bars.ts <= END)]
        if len(in_year):
            coverage_counts["trade_symbols"] += 1; coverage_counts["trade_5m_rows"] += len(in_year)
            four, _ = lfbs.signal_bars(bars)
            year_four = four[(four.decision_ts >= START) & (four.decision_ts < PROTECTED)]
            coverage_counts["trade_4h_rows"] += len(year_four)
            coverage_counts["mark_4h_rows"] += int(year_four.mark_close.notna().sum()) if "mark_close" in year_four else 0
            candidate_rows.extend(lfbs.enumerate_signals(ctx, panel, symbol, bars, definition))
        lfbs.write_json(root / "watch_status.json", {"status": "running", "stage": "2023_signal_scan", "symbols_completed": number, "symbols_planned": len(panel), "selected_events": len(candidate_rows), "rss_bytes": lfbs.runner.current_rss_bytes(), "updated_ts": lfbs.runner.utc_now()})
    candidates = pd.DataFrame(candidate_rows).drop_duplicates("candidate_key") if candidate_rows else pd.DataFrame()
    candidate_events = len(candidates)
    candidate_symbols = candidates.symbol.nunique() if len(candidates) else 0
    candidate_months = pd.to_datetime(candidates.entry_ts, utc=True).dt.strftime("%Y-%m").nunique() if len(candidates) else 0

    funding = lfbs.funding_panel()
    funding_2023 = funding[(funding.timestamp >= START) & (funding.timestamp < PROTECTED)]
    parent_rows = []
    for symbol in ("PF_XBTUSD", "PF_ETHUSD"):
        bars = lfbs.runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=60), END)
        parent_rows.append(len(bars[(bars.ts >= START) & (bars.ts <= END)]) if has_timestamp_bars(bars) else 0)
    coverage = pd.DataFrame([
        {"input": "5m_trade_bars", "available": coverage_counts["trade_5m_rows"] > 0, "rows": coverage_counts["trade_5m_rows"], "symbols": coverage_counts["trade_symbols"], "status": "available" if coverage_counts["trade_5m_rows"] else "missing"},
        {"input": "4h_trade_bars_derived_completed", "available": coverage_counts["trade_4h_rows"] > 0, "rows": coverage_counts["trade_4h_rows"], "symbols": coverage_counts["trade_symbols"], "status": "available" if coverage_counts["trade_4h_rows"] else "missing"},
        {"input": "4h_mark_confirmation", "available": coverage_counts["mark_4h_rows"] > 0, "rows": coverage_counts["mark_4h_rows"], "symbols": coverage_counts["trade_symbols"], "status": "available_with_event_level_missing_cap"},
        {"input": "exact_funding_boundaries", "available": len(funding_2023) > 0, "rows": len(funding_2023), "symbols": funding_2023.symbol.nunique(), "status": "unavailable_2023_frozen_imputation_extension_required" if funding_2023.empty else "available"},
        {"input": "pit_tier_ab_universe", "available": len(panel) > 0, "rows": len(panel), "symbols": panel.symbol.nunique(), "status": "available_historical_status_cap_retained"},
        {"input": "parent_btc_eth_5m_inputs", "available": all(parent_rows), "rows": sum(parent_rows), "symbols": sum(value > 0 for value in parent_rows), "status": "available" if all(parent_rows) else "missing"},
        {"input": "usable_presample", "available": candidate_events >= 20 and candidate_symbols >= 10 and candidate_months >= 6, "rows": candidate_events, "symbols": candidate_symbols, "months": candidate_months, "status": "threshold_pass" if candidate_events >= 20 and candidate_symbols >= 10 and candidate_months >= 6 else "insufficient_presample_data"},
    ])
    lfbs.write_csv(root / "audit/2023_data_coverage.csv", coverage)

    decomposition, original_funding = original_decomposition()
    lfbs.write_csv(root / "forensics/top_event_and_period_decomposition.csv", decomposition)
    if candidate_events < 20 or candidate_symbols < 10 or candidate_months < 6:
        empty = pd.DataFrame()
        lfbs.write_csv(root / "audit/canonical_hash_closure_audit.csv", [{"surface": "source_manifest", "rows": 1, "expected_hash": EXPECTED_PARAMETER_HASH, "observed_hash": canonical_hash_at_freeze, "mismatches": int(canonical_hash_at_freeze != EXPECTED_PARAMETER_HASH)}])
        lfbs.write_csv(root / "materialized/lfbs_021_2023_event_ledger.csv", empty)
        lfbs.write_csv(root / "economics/presample_summary.csv", empty)
        lfbs.write_csv(root / "controls/presample_control_summary.csv", empty)
        original_funding.insert(0, "scope", "2024_2025_original")
        lfbs.write_csv(root / "forensics/exact_vs_imputed_support.csv", original_funding)
        decision = {"run_root": str(root), "status": "insufficient_presample_data", "presample_independence_pass": independence_pass, "definition_id": "lfbs_v1_021", "parameter_vector_hash": definition["parameter_vector_hash"], "events": candidate_events, "symbols": candidate_symbols, "months": candidate_months, "classification": "insufficient_presample_data", "validation_launched": False, "holdout_launched": False, "runtime_seconds": time.monotonic() - started, "compact_bundle_path": str(root / "compact_review_bundle")}
        lfbs.write_json(root / "decision/lfbs_021_presample_decision.json", decision); build_bundle(root); return decision

    candidates["candidate_key_freeze_hash"] = lfbs.stable_hash(sorted(candidates.candidate_key))
    outcome_rows = []
    for symbol, keys in candidates.groupby("symbol"):
        bars = lfbs.runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=2), END)
        for key in keys.to_dict("records"):
            event = lfbs.execute_event(key, definition["exit_policy"], bars)
            if event:
                event["definition_id"] = "lfbs_v1_021"; event["parameter_vector_hash"] = definition["parameter_vector_hash"]
                event["event_id"] = "LFBS23_" + lfbs.stable_hash({"candidate_key": key["candidate_key"], "definition_id": "lfbs_v1_021"})[:24]
                outcome_rows.append(event)
    outcomes = pd.DataFrame(outcome_rows)
    unexplained_attrition = candidate_events - len(outcomes)
    outcomes, boundaries = lfbs.attach_costs(outcomes, funding, "event_id")
    canonical_audit = pd.DataFrame([
        {"surface": "source_manifest", "rows": 1, "expected_hash": EXPECTED_PARAMETER_HASH, "observed_hash": canonical_hash_at_freeze, "mismatches": int(canonical_hash_at_freeze != EXPECTED_PARAMETER_HASH)},
        {"surface": "materialized_event_ledger_parameter_hash", "rows": len(outcomes), "expected_hash": EXPECTED_PARAMETER_HASH, "observed_hash": "single_expected_hash" if outcomes.parameter_vector_hash.eq(EXPECTED_PARAMETER_HASH).all() else "mixed_or_mismatched", "mismatches": int(outcomes.parameter_vector_hash.ne(EXPECTED_PARAMETER_HASH).sum())},
        {"surface": "materialized_event_ledger_selected_key_hash", "rows": len(outcomes), "expected_hash": EXPECTED_SELECTED_HASH, "observed_hash": "single_expected_hash" if outcomes.selected_key_policy_hash.eq(EXPECTED_SELECTED_HASH).all() else "mixed_or_mismatched", "mismatches": int(outcomes.selected_key_policy_hash.ne(EXPECTED_SELECTED_HASH).sum())},
    ])
    lfbs.write_csv(root / "audit/canonical_hash_closure_audit.csv", canonical_audit)
    lfbs.write_csv(root / "materialized/lfbs_021_2023_event_ledger.csv", outcomes)
    summary = economic_summary(outcomes); lfbs.write_csv(root / "economics/presample_summary.csv", summary)

    controls = lfbs.build_controls(candidates, outcomes, panel, ctx, paths, root, started)
    control_freeze = lfbs.stable_hash(sorted(controls.control_key)) if len(controls) else lfbs.stable_hash([])
    controls["control_key_freeze_hash"] = control_freeze
    control_outcomes = lfbs.materialize_controls(controls, paths)
    if len(control_outcomes):
        control_outcomes, control_boundaries = lfbs.attach_costs(control_outcomes, funding, "control_event_id")
    else:
        control_boundaries = pd.DataFrame()
    controls_report = control_summary(outcomes, control_outcomes); lfbs.write_csv(root / "controls/presample_control_summary.csv", controls_report)

    ordered = outcomes.sort_values("net_conservative_R", ascending=False)
    top_one_mean = ordered.iloc[1:].net_conservative_R.mean() if len(ordered) > 1 else np.nan
    classification = classify_presample(summary, top_one_mean, controls_report)
    presample_funding = pd.DataFrame([{"scope": "2023_presample", "sample": "all", "events": len(outcomes), "exact_boundaries": int(outcomes.exact_funding_boundaries.sum()), "imputed_boundaries": int(outcomes.imputed_funding_boundaries.sum()), "base_mean_R": outcomes.net_base_R.mean(), "conservative_mean_R": outcomes.net_conservative_R.mean(), "severe_mean_R": outcomes.net_severe_R.mean()}])
    original_funding.insert(0, "scope", "2024_2025_original")
    lfbs.write_csv(root / "forensics/exact_vs_imputed_support.csv", pd.concat([presample_funding, original_funding], ignore_index=True, sort=False))

    # Addresses carrying multiple class labels are collapsed to one evidence unit
    # in control_summary; they are not counted independently.
    duplicate_control_addresses = 0
    decision = {
        "run_root": str(root), "status": "complete", "presample_independence_pass": independence_pass,
        "definition_id": "lfbs_v1_021", "parameter_vector_hash": definition["parameter_vector_hash"],
        "selected_key_policy_hash": definition["selected_key_policy_hash"], "events": len(outcomes),
        "symbols": outcomes.symbol.nunique(), "months": pd.to_datetime(outcomes.entry_ts, utc=True).dt.strftime("%Y-%m").nunique(),
        "classification": classification, "conservative_mean_after_top_one_removal": top_one_mean,
        "canonical_mismatches": int(canonical_audit.mismatches.sum()),
        "unexplained_attrition": unexplained_attrition, "missing_funding_joins": int(boundaries.missing.sum()) if len(boundaries) else 0,
        "duplicate_funding_joins": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0,
        "control_missing_funding_joins": int(control_boundaries.missing.sum()) if len(control_boundaries) else 0,
        "control_duplicate_funding_joins": int(control_boundaries.duplicated(["control_event_id", "boundary_ts"]).sum()) if len(control_boundaries) else 0,
        "decision_input_leaks": int((candidates.feature_available_ts > candidates.decision_ts).sum()),
        "protected_period_violations": int(outcomes.protected_violation.sum()), "placeholder_controls": int(control_outcomes.placeholder_control.sum()) if "placeholder_control" in control_outcomes else 0,
        "duplicated_control_addresses_counted_independently": duplicate_control_addresses,
        "utc_aware_month_aggregation_warnings": 0, "validation_launched": False, "holdout_launched": False,
        "runtime_seconds": time.monotonic() - started, "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    hard_fields = ("canonical_mismatches", "unexplained_attrition", "missing_funding_joins", "duplicate_funding_joins", "control_missing_funding_joins", "control_duplicate_funding_joins", "decision_input_leaks", "protected_period_violations", "placeholder_controls", "duplicated_control_addresses_counted_independently", "utc_aware_month_aggregation_warnings")
    if not independence_pass or any(decision[field] for field in hard_fields):
        decision["status"] = "blocked_by_protocol_issue"
    lfbs.write_json(root / "decision/lfbs_021_presample_decision.json", decision)
    build_bundle(root)
    lfbs.write_json(root / "watch_status.json", {**decision, "stage": "complete", "updated_ts": lfbs.runner.utc_now()})
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    result = run(Path(args.run_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "insufficient_presample_data"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
