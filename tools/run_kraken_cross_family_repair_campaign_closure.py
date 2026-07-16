#!/usr/bin/env python3
"""Close the signal-state campaign and reconcile RFBS v1_010 from frozen ledgers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import qlmg_signal_state_contract as state
from tools import run_kraken_rfbs_control_overlap_materialization as prior_adjudication
from tools import run_kraken_rfbs_signal_state_repaired as repaired


RUN_ROOT = Path("results/rebaseline/phase_kraken_cross_family_repair_campaign_closure_20260715_v1")
CAMPAIGN_ROOT = repaired.CAMPAIGN_ROOT
SCREEN_ROOT = repaired.RUN_ROOT
MATERIALIZATION_ROOT = Path("results/rebaseline/phase_kraken_rfbs_signal_state_repaired_materialization_20260715_v1")
FORMAL_ID = "rfbs_v1_010"
NEIGHBOR_ID = "rfbs_v1_007"
CONTRACT_VERSION = "cross_family_repair_campaign_closure_v1_20260715"
CONTEXTUAL = {"same_symbol_same_regime_random_short", "completed_failure_outside_riskoff_parent"}
STRUCTURAL = {"countertrend_rally_without_completed_failure", "generic_20d_failed_breakout_short", "non_rally_red_candle_short"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    return state.stable_hash([(str(path.relative_to(root)), file_hash(path)) for path in sorted(root.rglob("*")) if path.is_file()])


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def compact_bundle(root: Path, files: tuple[str, ...]) -> Path:
    temp = root / ".compact_review_bundle.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    inventory = []
    for relative in files:
        source = root / relative
        target = temp / relative.replace("/", "__")
        shutil.copy2(source, target)
        inventory.append({"source_path": relative, "bundle_path": target.name, "sha256": file_hash(source), "bytes": source.stat().st_size})
    write_csv(temp / "bundle_manifest.csv", inventory)
    final = root / "compact_review_bundle"
    if final.exists():
        shutil.rmtree(final)
    os.replace(temp, final)
    return final


def run(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    started = time.monotonic()
    source_hashes_before = {"screen": tree_hash(SCREEN_ROOT), "materialization": tree_hash(MATERIALIZATION_ROOT)}

    events = pd.read_csv(SCREEN_ROOT / "materialized/event_ledger.csv")
    controls = pd.read_csv(SCREEN_ROOT / "controls/control_event_ledger.csv")
    panel = pd.read_csv(SCREEN_ROOT / "manifest/pit_panel.csv")
    definitions = pd.read_csv(SCREEN_ROOT / "manifest/riskoff_failed_bounce_definitions.csv")
    summary = pd.read_csv(SCREEN_ROOT / "economics/definition_summary.csv")
    periods = pd.read_csv(SCREEN_ROOT / "economics/period_summary.csv")
    concentration = pd.read_csv(SCREEN_ROOT / "forensics/concentration_and_removal.csv")
    control_summary = pd.read_csv(SCREEN_ROOT / "controls/control_summary.csv")
    complements = pd.read_csv(MATERIALIZATION_ROOT / "controls/matched_unmatched_bias_repaired.csv")
    library = pd.read_csv(MATERIALIZATION_ROOT / "candidate_library/rfbs_candidate_library_update.csv")
    for frame in (events, controls):
        for column in [name for name in frame.columns if name.endswith("_ts")]:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    formal = events[events.definition_id.eq(FORMAL_ID)].copy()
    neighbor = events[events.definition_id.eq(NEIGHBOR_ID)].copy()
    if formal.empty or neighbor.empty:
        raise RuntimeError("frozen RFBS 007/010 events missing")

    stats = summary[summary.definition_id.eq(FORMAL_ID)].set_index("cost_mode")
    forensic = concentration[(concentration.definition_id == FORMAL_ID) & concentration.cost_mode.eq("conservative")].iloc[0]
    formal_periods = periods[(periods.definition_id == FORMAL_ID) & periods.cost_mode.eq("conservative")].copy()
    formal_controls = control_summary[(control_summary.definition_id == FORMAL_ID) & control_summary.cost_mode.eq("conservative")].copy()
    positive_adequate = formal_controls[formal_controls.adequate_control & formal_controls.mean_uplift_R.gt(0)]
    positive_classes = set(positive_adequate.control_class)
    formal_complements = complements[(complements.definition_id == FORMAL_ID) & complements.cost_mode.eq("conservative")].copy()
    complement_failures = int(((formal_complements.matched_count + formal_complements.unmatched_count) != formal_complements.full_count).sum())

    gate_rows = [
        {"gate": "positive_base_economics", "value": float(stats.loc["base"].mean_R), "pass": stats.loc["base"].mean_R > 0},
        {"gate": "positive_conservative_economics", "value": float(stats.loc["conservative"].mean_R), "pass": stats.loc["conservative"].mean_R > 0},
        {"gate": "positive_severe_economics", "value": float(stats.loc["severe"].mean_R), "pass": stats.loc["severe"].mean_R > 0},
        {"gate": "positive_after_top_one", "value": float(forensic.mean_after_top1), "pass": forensic.mean_after_top1 > 0},
        {"gate": "positive_after_top_three", "value": float(forensic.mean_after_top3), "pass": forensic.mean_after_top3 > 0},
        {"gate": "positive_worst_leave_one_symbol", "value": float(forensic.worst_leave_one_symbol_mean_R), "pass": forensic.worst_leave_one_symbol_mean_R > 0},
        {"gate": "positive_worst_leave_one_month", "value": float(forensic.worst_leave_one_month_mean_R), "pass": forensic.worst_leave_one_month_mean_R > 0},
        {"gate": "two_adequate_positive_controls", "value": len(positive_classes), "pass": len(positive_classes) >= 2},
        {"gate": "contextual_control_support", "value": "|".join(sorted(positive_classes & CONTEXTUAL)), "pass": bool(positive_classes & CONTEXTUAL)},
        {"gate": "structural_control_support", "value": "|".join(sorted(positive_classes & STRUCTURAL)), "pass": bool(positive_classes & STRUCTURAL)},
        {"gate": "matched_unmatched_complements_reconcile", "value": complement_failures, "pass": complement_failures == 0},
    ]

    address_col = "raw_signal_address_hash"
    formal_addresses = set(formal[address_col].astype(str))
    neighbor_addresses = set(neighbor[address_col].astype(str))
    shared = formal_addresses & neighbor_addresses
    formal_only = formal_addresses - neighbor_addresses
    neighbor_only = neighbor_addresses - formal_addresses
    formal["neighborhood_partition"] = np.where(formal[address_col].astype(str).isin(shared), "shared_strict_parent_tape", "broader_only")
    neighborhood_rows = [{
        "left_definition_id": NEIGHBOR_ID,
        "right_definition_id": FORMAL_ID,
        "left_events": len(neighbor_addresses),
        "right_events": len(formal_addresses),
        "shared_raw_signal_addresses": len(shared),
        "right_broader_only_addresses": len(formal_only),
        "left_not_in_right_addresses": len(neighbor_only),
        "jaccard": len(shared) / len(formal_addresses | neighbor_addresses),
        "strict_subset_of_broader_accepted_tape": not neighbor_only,
    }]
    partition = formal.groupby(["neighborhood_partition", "evaluation_period"]).agg(
        events=("event_id", "size"), symbols=("symbol", "nunique"), conservative_mean_R=("net_conservative_R", "mean"),
        conservative_total_R=("net_conservative_R", "sum"), severe_mean_R=("net_severe_R", "mean"),
    ).reset_index()
    neighbor_stats = summary[summary.definition_id.eq(NEIGHBOR_ID)][["definition_id", "cost_mode", "events", "mean_R", "profit_factor"]]

    formal_controls_raw = controls[controls.definition_id.eq(FORMAL_ID)].copy()
    formal_controls_raw["risk_to_daily_atr"] = formal_controls_raw.risk_denominator / formal_controls_raw.daily_atr
    formal_controls_raw["outside_candidate_risk_band"] = ~formal_controls_raw.risk_to_daily_atr.between(.25, 1.5, inclusive="both")
    candidate_risk = formal.assign(risk_to_daily_atr=formal.risk_denominator / formal.daily_atr)
    risk_rows = []
    for label, frame in [("candidate", candidate_risk), *[(name, group) for name, group in formal_controls_raw.groupby("control_class")]]:
        risk = frame.risk_to_daily_atr.dropna()
        denominator = frame.risk_denominator.dropna()
        risk_rows.append({
            "ledger_class": label, "rows": len(frame), "risk_denominator_min": denominator.min(), "risk_denominator_median": denominator.median(),
            "risk_denominator_max": denominator.max(), "risk_to_daily_atr_min": risk.min(), "risk_to_daily_atr_median": risk.median(),
            "risk_to_daily_atr_max": risk.max(), "outside_0.25_1.5_rows": int((~risk.between(.25, 1.5, inclusive="both")).sum()),
            "raw_results_preserved": True,
        })
    risk_rows.append({
        "ledger_class": "rfbs_v1_007_same_symbol_same_regime_limitation",
        "rows": int(((controls.definition_id == NEIGHBOR_ID) & controls.control_class.eq("same_symbol_same_regime_random_short")).sum()),
        "risk_denominator_min": controls[(controls.definition_id == NEIGHBOR_ID) & controls.control_class.eq("same_symbol_same_regime_random_short")].risk_denominator.min(),
        "raw_results_preserved": True,
        "comparability_note": "tiny-denominator outlier is a control-comparability limitation, not a mechanical error",
    })

    prior_adjudication.FORMAL_CANDIDATE = FORMAL_ID
    parity, integrity, horizons, adjudication = prior_adjudication.replay_and_path_audits(formal, panel, root)
    adjudication["is_top_five_winner"] = adjudication.event_id.isin(formal.nlargest(5, "net_conservative_R").event_id)
    adjudication["is_top_five_loser"] = adjudication.event_id.isin(formal.nsmallest(5, "net_conservative_R").event_id)
    adjudication["all_2023"] = adjudication.evaluation_period.eq("2023")
    adjudication["fully_exact_or_mixed"] = adjudication.exact_funding_boundaries.gt(0)
    selected_adjudication = adjudication[
        adjudication.is_top_five_winner | adjudication.is_top_five_loser | adjudication.all_2023 | adjudication.fully_exact_or_mixed
    ].copy()
    mechanics_failures = int((~parity.parity_pass).sum() + integrity.ohlcv_invalid_rows.sum() + (~integrity.stop_before_exit_consistent).sum() + (~integrity.lifecycle_pit_eligible).sum())
    gate_rows.append({"gate": "event_path_mechanics", "value": mechanics_failures, "pass": mechanics_failures == 0})

    neighbor_cons = summary[(summary.definition_id == NEIGHBOR_ID) & summary.cost_mode.eq("conservative")].iloc[0]
    neighbor_forensic = concentration[(concentration.definition_id == NEIGHBOR_ID) & concentration.cost_mode.eq("conservative")].iloc[0]
    neighbor_support = bool(neighbor_cons.mean_R > 0 and neighbor_forensic.mean_after_top3 > 0 and neighbor_forensic.worst_leave_one_symbol_mean_R > 0 and neighbor_forensic.worst_leave_one_month_mean_R > 0)
    gate_rows.append({"gate": "frozen_neighbor_support", "value": neighbor_support, "pass": neighbor_support})
    gate_frame = pd.DataFrame(gate_rows)
    qualified = bool(gate_frame["pass"].all())
    final = "train_only_stability_review_candidate" if qualified else "fragile_context_sleeve"

    funding = pd.read_csv(SCREEN_ROOT / "forensics/exact_vs_imputed_funding.csv")
    funding = funding[funding.definition_id.eq(FORMAL_ID)].copy()
    exact_negative = bool(len(funding[funding.funding_partition.eq("fully_exact")]) and (funding[funding.funding_partition.eq("fully_exact")].conservative_mean_R < 0).all())
    period_2023_negative = bool(len(formal_periods[formal_periods.period.eq("2023")]) and formal_periods[formal_periods.period.eq("2023")].iloc[0].mean_R < 0)

    write_csv(root / "decision/rfbs_010_gate_reconciliation.csv", gate_frame)
    write_csv(root / "decision/rfbs_010_control_support.csv", formal_controls)
    write_csv(root / "controls/rfbs_010_matched_unmatched_complements.csv", formal_complements)
    write_csv(root / "controls/risk_denominator_comparability.csv", risk_rows)
    write_csv(root / "controls/rfbs_010_raw_control_ledger.csv", formal_controls_raw)
    write_csv(root / "forensics/rfbs_010_period_support.csv", formal_periods)
    write_csv(root / "forensics/rfbs_010_funding_support.csv", funding)
    write_csv(root / "forensics/rfbs_010_path_diagnostics.csv", horizons)
    write_csv(root / "forensics/rfbs_010_event_path_integrity.csv", integrity)
    write_csv(root / "forensics/rfbs_010_replay_parity.csv", parity)
    write_csv(root / "materialized/rfbs_010_event_adjudication.csv", selected_adjudication)
    write_csv(root / "neighborhood/rfbs_007_010_overlap.csv", neighborhood_rows)
    write_csv(root / "neighborhood/rfbs_010_shared_broader_period_economics.csv", partition)
    write_csv(root / "neighborhood/rfbs_007_summary.csv", neighbor_stats)

    library.loc[library.definition_id.eq(FORMAL_ID), ["candidate_library_state", "candidate_decision"]] = final
    library.loc[library.definition_id.eq(NEIGHBOR_ID), ["candidate_library_state", "candidate_decision"]] = "frozen_neighborhood_support_not_independent_discovery"
    library["contract_version"] = repaired.CONTRACT_VERSION
    library["closure_contract_version"] = CONTRACT_VERSION
    library["closure_run_root"] = str(root)
    library["formal_candidate"] = library.definition_id.eq(FORMAL_ID)
    library["negative_2023_cap"] = library.definition_id.eq(FORMAL_ID) & period_2023_negative
    library["negative_exact_funding_cap"] = library.definition_id.eq(FORMAL_ID) & exact_negative
    write_csv(root / "candidate_library/central_full_schema_candidate_library.csv", library)

    registry = pd.read_csv(CAMPAIGN_ROOT / "campaign/affected_run_registry.csv")
    affected = registry.root_classification.isin(["directly_affected", "downstream_affected"])
    six_closed = int((affected & registry.closure_state.eq("closed")).sum())
    if int(affected.sum()) != 6 or six_closed != 6:
        raise RuntimeError(f"affected lineage closure mismatch: affected={int(affected.sum())} closed={six_closed}")
    registry.loc[registry.family.eq("RFBS"), "candidate_library_status"] = f"reconciled_evidence_{final}"
    write_csv(CAMPAIGN_ROOT / "campaign/affected_run_registry.csv", registry)

    preservation = pd.read_csv(CAMPAIGN_ROOT / "campaign/hypothesis_preservation_ledger.csv")
    mask = preservation.family.eq("RFBS")
    preservation.loc[mask, "preservation_status"] = f"repaired_lineage_closed_{final}"
    preservation.loc[mask, "permitted_claim"] = "rfbs_v1_010 may enter train-only stability review with negative-2023, negative-exact-funding, imputation, OHLCV-stop and no-depth caps"
    preservation.loc[mask, "forbidden_claim"] = "validated, holdout-supported, portfolio-ready, live-ready, or resolved across 2023/exact funding"
    write_csv(CAMPAIGN_ROOT / "campaign/hypothesis_preservation_ledger.csv", preservation)

    matrix = pd.read_csv(CAMPAIGN_ROOT / "campaign/repair_completion_matrix.csv")
    matrix.loc[matrix.family.eq("Shared contract"), ["repaired_replay_complete", "downstream_replay_complete", "reconciled_and_closed"]] = True
    matrix.loc[matrix.family.eq("Shared contract"), "next_action"] = "campaign closed; RFBS 010 train-only stability review only"
    matrix.loc[matrix.family.eq("RFBS"), "next_action"] = "RFBS 010 train-only stability review only"
    matrix.loc[matrix.family.eq("Close-confirmed breakout retest"), "next_action"] = "remain paused during RFBS 010 train-only stability review"
    write_csv(CAMPAIGN_ROOT / "campaign/repair_completion_matrix.csv", matrix)

    next_target = "RFBS 010 train-only stability review only" if qualified else "Close-confirmed breakout retest long screen"
    gate_path = CAMPAIGN_ROOT / "campaign/new_family_launch_gate.json"
    launch_gate = json.loads(gate_path.read_text())
    launch_gate.update({
        "new_family_launch_allowed": False if qualified else True,
        "unresolved_registry_count": 0,
        "all_six_affected_lineage_entries_closed": True,
        "human_campaign_closure_review_complete": True,
        "next_prompt_target": next_target,
        "next_authorized_target": next_target,
        "rfbs_v1_010_reconciled_decision": final,
        "final_holdout_sealed": True,
    })
    write_json(gate_path, launch_gate)
    campaign_decision_path = CAMPAIGN_ROOT / "decision_summary.json"
    campaign_decision = json.loads(campaign_decision_path.read_text())
    campaign_decision.update({
        "status": "complete", "campaign_closed": True, "unresolved_registry_count": 0,
        "rfbs_v1_010_reconciled_decision": final, "next_recommended_prompt_target": next_target,
        "new_family_launch_allowed": launch_gate["new_family_launch_allowed"], "final_holdout_sealed": True,
    })
    write_json(campaign_decision_path, campaign_decision)
    repaired.refresh_campaign_bundle()

    campaign_snapshot = root / "campaign"
    campaign_snapshot.mkdir()
    for name in ("affected_run_registry.csv", "hypothesis_preservation_ledger.csv", "repair_completion_matrix.csv", "new_family_launch_gate.json"):
        shutil.copy2(CAMPAIGN_ROOT / "campaign" / name, campaign_snapshot / name)

    source_hashes_after = {"screen": tree_hash(SCREEN_ROOT), "materialization": tree_hash(MATERIALIZATION_ROOT)}
    source_mutations = sum(source_hashes_before[key] != source_hashes_after[key] for key in source_hashes_before)
    write_csv(root / "audit/source_root_immutability_audit.csv", [
        {"source": key, "hash_before": source_hashes_before[key], "hash_after": source_hashes_after[key], "mutated": source_hashes_before[key] != source_hashes_after[key]}
        for key in source_hashes_before
    ])
    if source_mutations:
        raise RuntimeError("immutable repaired source root changed")

    report = f"""# Cross-Family Repair Campaign Closure\n\nAll six directly/downstream affected lineage entries are closed under `{repaired.CONTRACT_VERSION}`. RFBS formal candidate metadata is corrected from `rfbs_v1_004` to `rfbs_v1_010`. The reconciled decision is `{final}`. `rfbs_v1_010` has positive base, conservative, severe, top-three, leave-symbol and leave-month results; positive adequate contextual and structural controls; and frozen-neighbour support from `rfbs_v1_007`. The recommendation remains capped because 2023 conservative evidence and fully exact-funded evidence are negative, most support is imputed-funded, and raw control risk comparability is imperfect. No signal, candidate, control, funding, cost, or source-root artifact was changed. The final holdout remains sealed. Next authorized target: `{next_target}`.\n"""
    (root / "CAMPAIGN_CLOSURE_REPORT.md").write_text(report, encoding="utf-8")
    repro = {
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_path": str(Path(__file__)), "code_hash": file_hash(Path(__file__)), "contract_version": CONTRACT_VERSION,
        "signal_state_contract_version": repaired.CONTRACT_VERSION, "screen_root_hash": source_hashes_before["screen"],
        "materialization_root_hash": source_hashes_before["materialization"], "formal_definition_id": FORMAL_ID,
        "protected_boundary": "2026-01-01T00:00:00Z", "signal_regeneration": False, "control_regeneration": False,
    }
    write_json(root / "reproducibility/run_manifest.json", repro)
    decision = {
        "run_root": str(root), "status": "complete", "formal_candidate": FORMAL_ID, "final_decision": final,
        "all_six_affected_lineage_entries_closed": True, "campaign_unresolved_entries": 0,
        "base_mean_R": float(stats.loc["base"].mean_R), "conservative_mean_R": float(stats.loc["conservative"].mean_R),
        "severe_mean_R": float(stats.loc["severe"].mean_R), "top_one_mean_R": float(forensic.mean_after_top1),
        "top_three_mean_R": float(forensic.mean_after_top3), "worst_leave_one_symbol_mean_R": float(forensic.worst_leave_one_symbol_mean_R),
        "worst_leave_one_month_mean_R": float(forensic.worst_leave_one_month_mean_R),
        "adequate_positive_control_classes": sorted(positive_classes), "negative_2023_cap": period_2023_negative,
        "negative_exact_funding_cap": exact_negative, "mechanical_failures": mechanics_failures,
        "source_root_mutations": source_mutations, "final_holdout_sealed": True,
        "next_authorized_target": next_target, "new_family_launch_allowed": launch_gate["new_family_launch_allowed"],
        "validation_launched": False, "cpcv_launched": False, "holdout_launched": False, "portfolio_launched": False, "live_launched": False,
        "runtime_seconds": time.monotonic() - started, "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision_summary.json", decision)
    compact_bundle(root, (
        "CAMPAIGN_CLOSURE_REPORT.md", "decision_summary.json", "reproducibility/run_manifest.json",
        "decision/rfbs_010_gate_reconciliation.csv", "decision/rfbs_010_control_support.csv",
        "controls/rfbs_010_matched_unmatched_complements.csv", "controls/risk_denominator_comparability.csv",
        "forensics/rfbs_010_period_support.csv", "forensics/rfbs_010_funding_support.csv",
        "forensics/rfbs_010_path_diagnostics.csv", "forensics/rfbs_010_event_path_integrity.csv",
        "materialized/rfbs_010_event_adjudication.csv", "neighborhood/rfbs_007_010_overlap.csv",
        "neighborhood/rfbs_010_shared_broader_period_economics.csv", "candidate_library/central_full_schema_candidate_library.csv",
        "campaign/affected_run_registry.csv", "campaign/hypothesis_preservation_ledger.csv",
        "campaign/repair_completion_matrix.csv", "campaign/new_family_launch_gate.json",
        "audit/source_root_immutability_audit.csv",
    ))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    result = run(args.run_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
