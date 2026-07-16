#!/usr/bin/env python3
"""Build the cross-family signal-state repair inventory without reading outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from tools import qlmg_evidence_contracts as evidence
from tools import qlmg_signal_state_contract as signal_state


DEFAULT_ROOT = Path("results/rebaseline/phase_kraken_cross_family_signal_state_repair_campaign_20260715_v1")
AUDIT_ROOT = Path("results/rebaseline/phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v1")
REPAIRED_REFERENCE = Path("results/rebaseline/phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1")

DIRECT_RUNNERS = {
    "tools/run_kraken_liquid_failed_breakout_short_screen.py": "nominal_72h_preblock_before_actual_stop_or_structural_exit",
    "tools/run_kraken_backside_blowoff_short_screen.py": "nominal_7d_preblock_before_definition_exit",
    "tools/run_kraken_riskoff_failed_bounce_short_screen.py": "nominal_7d_preblock_before_definition_exit",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def line_numbers(text: str, tokens: tuple[str, ...]) -> str:
    return "|".join(
        str(number)
        for number, line in enumerate(text.splitlines(), 1)
        if any(token in line for token in tokens)
    )


def repository_scan() -> list[dict[str, Any]]:
    runners = sorted(Path("tools").glob("run_kraken_*.py"))
    support = [Path("tools/qlmg_screening_core.py"), Path("tools/qlmg_short_event_generators.py")]
    rows: list[dict[str, Any]] = []
    for path in [*runners, *support]:
        text = path.read_text()
        rel = str(path)
        state_lines = line_numbers(text, ("blocked_until", "position_blocked_until", "next_allowed_ts", "next_allowed"))
        nominal_lines = line_numbers(text, ("Timedelta(hours=72)", "Timedelta(days=7)", "max_holding_hours", "maximum_hold"))
        actual_exit_lines = line_numbers(text, ('blocked_until[key["symbol"]]=pd.Timestamp(event["exit_ts"])', 'blocked_until[key["symbol"]] = pd.Timestamp(event["exit_ts"])', "prior_actual_exit_ts"))
        if rel in DIRECT_RUNNERS:
            classification = "directly_affected_pre_outcome_signal_suppression"
            unsafe = True
            rationale = DIRECT_RUNNERS[rel]
        elif rel == "tools/run_kraken_delayed_flush_reclaim_long_screen.py":
            classification = "provenance_only_original_defect_repaired_elsewhere"
            unsafe = True
            rationale = "nominal_7d_preblock; source root already blocked and repaired replay exists"
        elif rel == "tools/run_kraken_delayed_flush_reclaim_signal_state_repair.py":
            classification = "valid_repaired_reference"
            unsafe = False
            rationale = "parent-neutral raw tape and definition-local actual-exit state"
        elif rel in {
            "tools/run_kraken_lfbs_021_frozen_2023_presample_confirmation.py",
            "tools/run_kraken_lfbs_021_canonical_episode_adjudication.py",
            "tools/run_kraken_rfbs_control_overlap_materialization.py",
        }:
            classification = "downstream_consumer_of_affected_identity"
            unsafe = False
            rationale = "no independent nominal-hold enumerator; lineage consumes affected runner or tape"
        elif state_lines:
            classification = "reviewed_actual_exit_or_non_signal_state"
            unsafe = False
            rationale = "state token present but no unclassified pre-outcome nominal-hold suppression"
        elif nominal_lines:
            classification = "reviewed_outcome_horizon_boundary_or_configuration_only"
            unsafe = False
            rationale = "nominal hold token is not selected-key position suppression"
        else:
            classification = "no_preblock_pattern_detected"
            unsafe = False
            rationale = "no nominal-hold position-state pattern"
        rows.append({
            "runner": rel,
            "file_sha256": sha256_file(path),
            "scope": "active_post_qa_runner" if path.name.startswith("run_kraken_") else "shared_support_module",
            "state_token_lines": state_lines,
            "nominal_hold_lines": nominal_lines,
            "actual_exit_state_lines": actual_exit_lines,
            "pre_outcome_signal_suppression": unsafe,
            "classification": classification,
            "rationale": rationale,
            "economic_outcomes_read": False,
        })
    return rows


def registry_rows() -> list[dict[str, Any]]:
    def row(
        family: str, hypothesis: str, runner: str, root: str, decision: str, positives: str,
        classification: str, defect: str, rerun: bool, replay: bool, placeholder: str,
        library: str, closure: str,
    ) -> dict[str, Any]:
        return {
            "family": family, "hypothesis": hypothesis, "runner": runner, "original_root": root,
            "original_decision": decision, "prior_positive_variants": positives,
            "root_classification": classification, "direct_or_downstream_defect_status": defect,
            "required_rerun": rerun, "required_downstream_replay": replay,
            "repaired_root_placeholder": placeholder, "candidate_library_status": library,
            "closure_state": closure,
        }

    quarantine = "prior_decision_quarantined_pending_repaired_evidence"
    return [
        row("LFBS", "liquid failed-breakout short", "tools/run_kraken_liquid_failed_breakout_short_screen.py", "results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1", "context_sleeve_lfbs_v1_021", "lfbs_v1_021", "directly_affected", "premature_72h_selected_key_preblock", True, True, "<LFBS_REPAIRED_SCREEN_ROOT>", quarantine, "open_repair_required"),
        row("LFBS", "frozen independent 2023 presample for lfbs_v1_021", "tools/run_kraken_lfbs_021_frozen_2023_presample_confirmation.py", "results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1", "fragile_presample_support", "lfbs_v1_021", "downstream_affected", "imports affected LFBS enumerator and source definition", False, True, "<LFBS_REPAIRED_2023_PRESAMPLE_ROOT>", quarantine, "open_downstream_replay_required"),
        row("LFBS", "canonical 2023-2025 episode adjudication", "tools/run_kraken_lfbs_021_canonical_episode_adjudication.py", "results/rebaseline/phase_kraken_lfbs_021_canonical_episode_adjudication_20260713_v1", "fragile_context_sleeve", "lfbs_v1_021", "downstream_affected", "directly invokes affected LFBS enumerator and consumes prior roots", False, True, "<LFBS_REPAIRED_CANONICAL_ADJUDICATION_ROOT>", quarantine, "open_downstream_replay_required"),
        row("Backside blowoff", "backside-confirmed blowoff short", "tools/run_kraken_backside_blowoff_short_screen.py", "results/rebaseline/phase_kraken_backside_blowoff_short_screen_20260713_v1", "screen_closed_no_materialization_candidate", "bcbs_v1_002|bcbs_v1_008", "directly_affected", "premature_7d_selected_key_preblock", True, False, "<BACKSIDE_REPAIRED_SCREEN_ROOT>", quarantine, "open_repair_required"),
        row("RFBS", "risk-off failed-bounce swing short", "tools/run_kraken_riskoff_failed_bounce_short_screen.py", "results/rebaseline/phase_kraken_riskoff_failed_bounce_short_screen_20260714_v1", "materialization_candidate", "rfbs_v1_001|rfbs_v1_004|rfbs_v1_007|rfbs_v1_010", "directly_affected", "premature_7d_selected_key_preblock", True, True, "<RFBS_REPAIRED_SCREEN_ROOT>", quarantine, "open_repair_required"),
        row("RFBS", "control overlap closure and targeted materialization", "tools/run_kraken_rfbs_control_overlap_materialization.py", "results/rebaseline/phase_kraken_rfbs_control_overlap_materialization_20260714_v1", "fragile_context_sleeve", "rfbs_v1_001|rfbs_v1_004|rfbs_v1_007|rfbs_v1_010", "downstream_affected", "consumes affected RFBS candidate and control identity", False, True, "<RFBS_REPAIRED_CONTROL_OVERLAP_ROOT>", quarantine, "open_downstream_replay_required"),
        row("Delayed flush reclaim", "bar-proxy delayed flush reclaim long original", "tools/run_kraken_delayed_flush_reclaim_long_screen.py", "results/rebaseline/phase_kraken_delayed_flush_reclaim_long_screen_20260715_v1", "blocked_by_protocol_issue", "", "provenance_only", "original premature preblock already formally quarantined", False, False, "results/rebaseline/phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1", "diagnostic_only_provenance", "closed_by_repaired_reference"),
        row("Delayed flush reclaim", "bar-proxy delayed flush reclaim long repaired", "tools/run_kraken_delayed_flush_reclaim_signal_state_repair.py", "results/rebaseline/phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1", "current_translation_weak", "", "valid_repaired_reference", "none", False, False, "", "repaired_train_only_evidence_current", "closed"),
        row("Close-confirmed breakout retest", "rolling-range breakout retest reclaim long", "", "results/rebaseline/phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v1", "focused_mechanical_repair_required", "", "untested_blocked", "launch gate stopped before contract or outcomes", False, False, "<UNTESTED_AFTER_CAMPAIGN_CLOSURE>", "no_candidate_evidence_unstarted", "paused"),
        row("LFBS provenance", "failed startup attempt", "tools/run_kraken_lfbs_021_frozen_2023_presample_confirmation.py", "results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1_startup_failed_empty_symbol_schema_20260713T124821", "startup_failed", "", "provenance_only", "no completed decision tape", False, False, "", "provenance_only", "archived"),
        row("Backside provenance", "superseded invented EMA cap run", "tools/run_kraken_backside_blowoff_short_screen.py", "results/rebaseline/phase_kraken_backside_blowoff_short_screen_20260713_v1_invalid_invented_ema7d_cap_provenance", "invalid_provenance", "", "provenance_only", "superseded and independently invalid", False, False, "", "provenance_only", "archived"),
        row("RFBS provenance", "pre-memory-retention attempt", "tools/run_kraken_riskoff_failed_bounce_short_screen.py", "results/rebaseline/phase_kraken_riskoff_failed_bounce_short_screen_20260714_v1_pre_memory_retention_repair_provenance", "provenance_only", "", "provenance_only", "superseded execution provenance", False, False, "", "provenance_only", "archived"),
        row("RFBS provenance", "pre-nesting-explanation closure", "tools/run_kraken_rfbs_control_overlap_materialization.py", "results/rebaseline/phase_kraken_rfbs_control_overlap_materialization_20260714_v1_pre_nesting_explanation_provenance", "provenance_only", "", "provenance_only", "superseded reporting provenance", False, False, "", "provenance_only", "archived"),
        row("A1/compression", "liquid leader continuation and compression", "tools/run_kraken_a1_full_180shard.py", "results/rebaseline/phase_kraken_a1_compression_full_180shard_funding_corrected_20260711_v1_20260711_194859", "unchanged_by_signal_state_campaign", "", "unaffected_decision_bearing", "no nominal-hold pre-outcome suppression detected", False, False, "", "unchanged", "outside_campaign_closed"),
        row("TSMOM v6", "time-series momentum", "tools/run_kraken_family_engine_aggregate_first_sweep.py", "results/rebaseline/phase_kraken_full_tsmom_v6_aggregate_20260707_v1", "unchanged_by_signal_state_campaign", "", "unaffected_decision_bearing", "no nominal-hold pre-outcome suppression detected", False, False, "", "unchanged", "outside_campaign_closed"),
        row("Prior-high reclaim v2", "close-confirmed prior-high/reclaim", "tools/run_kraken_prior_high_v2_canonical_scan.py", "results/rebaseline/phase_kraken_prior_high_reclaim_v2_canonical_train_scan_20260712_v1", "unchanged_by_signal_state_campaign", "", "unaffected_decision_bearing", "no nominal-hold pre-outcome suppression detected", False, False, "", "unchanged", "outside_campaign_closed"),
        row("C2", "post-catalyst continuation base", "tools/run_kraken_c2_sample_limited_economic_tranche.py", "results/rebaseline/phase_kraken_c2_sample_limited_economic_tranche_20260713_v1", "unchanged_by_signal_state_campaign", "", "unaffected_decision_bearing", "no nominal-hold pre-outcome suppression detected", False, False, "", "unchanged", "outside_campaign_closed"),
    ]


def dependency_rows() -> list[dict[str, Any]]:
    return [
        {"upstream_root": "results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1", "downstream_root": "results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1", "dependency": "frozen_definition_and_affected_enumerator", "identity_bearing": True, "required_replay_order": 2},
        {"upstream_root": "results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1", "downstream_root": "results/rebaseline/phase_kraken_lfbs_021_canonical_episode_adjudication_20260713_v1", "dependency": "source_train_identity_and_affected_enumerator", "identity_bearing": True, "required_replay_order": 3},
        {"upstream_root": "results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1", "downstream_root": "results/rebaseline/phase_kraken_lfbs_021_canonical_episode_adjudication_20260713_v1", "dependency": "source_2023_identity_and_comparison", "identity_bearing": True, "required_replay_order": 3},
        {"upstream_root": "results/rebaseline/phase_kraken_riskoff_failed_bounce_short_screen_20260714_v1", "downstream_root": "results/rebaseline/phase_kraken_rfbs_control_overlap_materialization_20260714_v1", "dependency": "candidate_control_and_outcome_identity", "identity_bearing": True, "required_replay_order": 2},
        {"upstream_root": "results/rebaseline/phase_kraken_backside_blowoff_short_screen_20260713_v1", "downstream_root": "results/rebaseline/phase_kraken_riskoff_failed_bounce_short_screen_20260714_v1", "dependency": "candidate_library_metadata_update_only", "identity_bearing": False, "required_replay_order": 0},
        {"upstream_root": "results/rebaseline/phase_kraken_delayed_flush_reclaim_long_screen_20260715_v1", "downstream_root": "results/rebaseline/phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1", "dependency": "frozen_definition_repaired_replay", "identity_bearing": True, "required_replay_order": 0},
        {"upstream_root": str(AUDIT_ROOT), "downstream_root": "results/rebaseline/phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v1", "dependency": "mandatory_launch_gate", "identity_bearing": False, "required_replay_order": 0},
    ]


def hypothesis_rows() -> list[dict[str, Any]]:
    return [
        {"family": "LFBS", "hypothesis": "completed failed breakout can support a liquid short sleeve", "prior_variants": "lfbs_v1_021", "preservation_status": "quarantined_not_rejected", "permitted_claim": "hypothesis remains open pending repaired screen and downstream replay", "forbidden_claim": "prior fragile support is decision-bearing"},
        {"family": "Backside blowoff", "hypothesis": "backside confirmation after parabolic extension may improve short timing", "prior_variants": "bcbs_v1_002|bcbs_v1_008", "preservation_status": "quarantined_not_rejected", "permitted_claim": "variants remain frozen repair targets", "forbidden_claim": "prior positives or weak family closure are decision-bearing"},
        {"family": "RFBS", "hypothesis": "failed countertrend bounce in risk-off state may support a short sleeve", "prior_variants": "rfbs_v1_001|rfbs_v1_004|rfbs_v1_007|rfbs_v1_010", "preservation_status": "quarantined_not_rejected", "permitted_claim": "frozen neighbourhood remains a repair target", "forbidden_claim": "materialization candidate or fragile sleeve conclusion is decision-bearing"},
        {"family": "Delayed flush reclaim", "hypothesis": "delayed reclaim after a downside flush may support a long", "prior_variants": "", "preservation_status": "closed_under_valid_repaired_reference", "permitted_claim": "current translation weak on repaired train-only evidence", "forbidden_claim": "validated or family rejected"},
        {"family": "Close-confirmed breakout retest", "hypothesis": "retest and reclaim after a rolling-range breakout may support a long", "prior_variants": "", "preservation_status": "untested_paused", "permitted_claim": "mechanical contract not yet run", "forbidden_claim": "any economic conclusion"},
    ]


def completion_rows() -> list[dict[str, Any]]:
    return [
        {"family": "Shared contract", "priority": 0, "runner_migrated": True, "regression_tests_passed": True, "repaired_replay_complete": False, "downstream_replay_complete": False, "reconciled_and_closed": False, "next_action": "apply frozen contract to LFBS only"},
        {"family": "LFBS", "priority": 1, "runner_migrated": False, "regression_tests_passed": False, "repaired_replay_complete": False, "downstream_replay_complete": False, "reconciled_and_closed": False, "next_action": "repair LFBS raw tape and definition-local actual-exit state"},
        {"family": "Backside blowoff", "priority": 2, "runner_migrated": False, "regression_tests_passed": False, "repaired_replay_complete": False, "downstream_replay_complete": True, "reconciled_and_closed": False, "next_action": "wait until LFBS lineage closes"},
        {"family": "RFBS", "priority": 3, "runner_migrated": False, "regression_tests_passed": False, "repaired_replay_complete": False, "downstream_replay_complete": False, "reconciled_and_closed": False, "next_action": "wait until LFBS and backside lineages close"},
        {"family": "Delayed flush reclaim", "priority": 0, "runner_migrated": True, "regression_tests_passed": True, "repaired_replay_complete": True, "downstream_replay_complete": True, "reconciled_and_closed": True, "next_action": "none; valid architecture reference; do not rerun"},
        {"family": "Close-confirmed breakout retest", "priority": 4, "runner_migrated": False, "regression_tests_passed": False, "repaired_replay_complete": False, "downstream_replay_complete": False, "reconciled_and_closed": False, "next_action": "remain paused until campaign closure"},
    ]


def compact_bundle(root: Path, files: list[str]) -> None:
    bundle = root / "compact_review_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    manifest_rows = []
    for relative in files:
        source = root / relative
        target = bundle / relative.replace("/", "__")
        shutil.copy2(source, target)
        manifest_rows.append({"source_path": relative, "bundle_path": target.name, "sha256": sha256_file(target), "bytes": target.stat().st_size})
    write_csv(bundle / "bundle_manifest.csv", manifest_rows)


def run(root: Path) -> dict[str, Any]:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"run root must be fresh: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not AUDIT_ROOT.exists() or not REPAIRED_REFERENCE.exists():
        raise FileNotFoundError("required audit or repaired architecture reference missing")

    scan = repository_scan()
    registry = registry_rows()
    dependencies = dependency_rows()
    hypotheses = hypothesis_rows()
    completion = completion_rows()
    write_csv(root / "audit/repository_preblock_scan.csv", scan)
    write_csv(root / "campaign/affected_run_registry.csv", registry)
    write_csv(root / "campaign/downstream_dependency_map.csv", dependencies)
    write_csv(root / "campaign/hypothesis_preservation_ledger.csv", hypotheses)
    write_csv(root / "campaign/repair_completion_matrix.csv", completion)

    contract = f"""# Cross-Family Signal-State Contract\n\nVersion: `{signal_state.SIGNAL_STATE_CONTRACT_VERSION}`.\n\n1. Family code emits every mechanically valid parent-neutral raw signal without position or nominal maximum-hold blocking. Repeated bars from one unresolved mechanical setup are deduplicated by immutable setup identity, not elapsed holding time.\n2. Raw rows are explicitly sorted, unique-address checked, content-hashed, and frozen before outcomes.\n3. Parent policies are PIT projections of the same raw tape. Every parent feature timestamp must be no later than `decision_ts`; strict policy addresses must be a subset of the broad policy addresses.\n4. Each definition allocates a fresh chronological non-overlap state machine. A signal is skipped only while that symbol and definition has a position open under its actual executable `exit_ts`.\n5. Every skip records definition, candidate, symbol, new entry, prior trade, prior entry, prior actual exit, and reason. No state is shared across definitions.\n6. Accepted rows are content-hashed. Eligible definition rows must reconcile exactly to accepted trades plus overlap skips plus outcome exclusions.\n7. Rankable evidence must pass `qlmg_evidence_contracts.assert_rankable_signal_state_contract`; missing version or hashes fail closed.\n"""
    (root / "contract").mkdir(parents=True, exist_ok=True)
    (root / "contract/signal_state_contract_v1.md").write_text(contract)

    direct_count = sum(row["root_classification"] == "directly_affected" for row in registry)
    downstream_count = sum(row["root_classification"] == "downstream_affected" for row in registry)
    unresolved = sum(row["closure_state"].startswith("open_") for row in registry)
    gate = {
        "signal_state_contract_version": evidence.SIGNAL_STATE_CONTRACT_VERSION,
        "new_family_launch_allowed": False,
        "directly_affected_roots": direct_count,
        "downstream_affected_roots": downstream_count,
        "unresolved_registry_count": unresolved,
        "conditions_to_open": [
            "all_directly_affected_runners_migrated_and_tested",
            "all_required_repaired_replays_complete",
            "all_downstream_identity_consumers_replayed",
            "all_hash_and_non_overlap_reconciliations_closed",
        ],
        "paused_family": "close_confirmed_breakout_retest_long",
        "next_prompt_target": "LFBS only",
    }
    write_json(root / "campaign/new_family_launch_gate.json", gate)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    reproducibility = {
        "run_type": "source_inventory_contract_freeze_no_economic_rerun",
        "commit_hash": commit,
        "contract_version": signal_state.SIGNAL_STATE_CONTRACT_VERSION,
        "audit_root": str(AUDIT_ROOT),
        "valid_architecture_reference": str(REPAIRED_REFERENCE),
        "economic_outcomes_read": False,
        "strategies_rerun": False,
        "historical_roots_modified": False,
        "source_hashes": {
            "tools/qlmg_signal_state_contract.py": sha256_file(Path("tools/qlmg_signal_state_contract.py")),
            "tools/qlmg_evidence_contracts.py": sha256_file(Path("tools/qlmg_evidence_contracts.py")),
            "tools/run_kraken_cross_family_signal_state_repair_campaign.py": sha256_file(Path(__file__)),
        },
    }
    write_json(root / "reproducibility/run_manifest.json", reproducibility)
    decision = {
        "run_root": str(root),
        "status": "complete",
        "final_decision": "repair_affected_lineages_before_new_family_launch",
        "directly_affected_roots": direct_count,
        "downstream_affected_roots": downstream_count,
        "unresolved_registry_count": unresolved,
        "discovered_affected_families": ["LFBS", "backside_blowoff", "RFBS"],
        "valid_repaired_reference_families": ["delayed_flush_reclaim"],
        "untested_blocked_families": ["close_confirmed_breakout_retest"],
        "shared_contract_implemented": True,
        "signal_state_contract_version": signal_state.SIGNAL_STATE_CONTRACT_VERSION,
        "new_family_launch_allowed": False,
        "economic_reruns_launched": False,
        "validation_launched": False,
        "cpcv_launched": False,
        "holdout_launched": False,
        "portfolio_work_launched": False,
        "live_work_launched": False,
        "prior_hypotheses_rejected": False,
        "next_recommended_prompt_target": "LFBS only",
        "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision_summary.json", decision)
    report = f"""# Cross-Family Signal-State Repair Campaign\n\nStatus: complete. No economic outcomes were read and no strategy was rerun.\n\n- Directly affected roots: {direct_count}\n- Downstream affected roots: {downstream_count}\n- Unresolved affected roots: {unresolved}\n- Shared contract: `{signal_state.SIGNAL_STATE_CONTRACT_VERSION}` implemented\n- New-family launch allowed: no\n- Next repair target: LFBS only\n\nPrior economic decisions for LFBS, backside blowoff, and RFBS are quarantined pending repaired evidence; no hypothesis is marked rejected. The delayed-flush repaired root remains the valid architecture reference and was not rerun.\n"""
    (root / "CAMPAIGN_REPORT.md").write_text(report)
    compact_bundle(root, [
        "decision_summary.json", "CAMPAIGN_REPORT.md", "contract/signal_state_contract_v1.md",
        "campaign/affected_run_registry.csv", "campaign/downstream_dependency_map.csv",
        "campaign/hypothesis_preservation_ledger.csv", "campaign/repair_completion_matrix.csv",
        "campaign/new_family_launch_gate.json", "audit/repository_preblock_scan.csv",
        "reproducibility/run_manifest.json",
    ])
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    decision = run(parse_args().run_root)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
