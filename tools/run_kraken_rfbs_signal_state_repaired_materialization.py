#!/usr/bin/env python3
"""Replay RFBS materialization/controls from repaired signal-state identities."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from tools import run_kraken_rfbs_control_overlap_materialization as adjudication
from tools import run_kraken_rfbs_signal_state_repaired as repaired


RUN_ROOT = Path("results/rebaseline/phase_kraken_rfbs_signal_state_repaired_materialization_20260715_v1")
SCREEN_ROOT = repaired.RUN_ROOT
PRESERVED_IDS = ("rfbs_v1_001", "rfbs_v1_004", "rfbs_v1_007", "rfbs_v1_010")


def _refresh_bundle(root: Path) -> None:
    bundle = root / "compact_review_bundle"
    manifest = pd.read_csv(bundle / "bundle_manifest.csv")
    sources = manifest.source_relative_path.astype(str).tolist()
    for extra in ("RFBS_LINEAGE_RECONCILIATION_REPORT.md", "audit/old_to_new_address_map.csv", "forensics/exact_vs_imputed_funding.csv"):
        if extra not in sources:
            sources.append(extra)
    temp = root / ".compact_review_bundle.refresh.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    rows = []
    for relative in sources:
        source = root / relative
        target = temp / relative.replace("/", "__")
        shutil.copy2(source, target)
        rows.append({"bundle_file": target.name, "source_relative_path": relative, "bytes": source.stat().st_size, "sha256": adjudication.file_hash(source)})
    repaired.write_csv(temp / "bundle_manifest.csv", rows)
    shutil.rmtree(bundle)
    os.replace(temp, bundle)


def _close_campaign(screen_root: Path, materialization_root: Path, classification: str) -> None:
    campaign = repaired.CAMPAIGN_ROOT
    registry_path = campaign / "campaign/affected_run_registry.csv"
    registry = pd.read_csv(registry_path)
    mask = registry.family.eq("RFBS")
    registry.loc[mask, "required_rerun"] = False
    registry.loc[mask, "required_downstream_replay"] = False
    registry.loc[mask & registry.root_classification.eq("directly_affected"), "repaired_root_placeholder"] = str(screen_root)
    registry.loc[mask & registry.root_classification.eq("downstream_affected"), "repaired_root_placeholder"] = str(materialization_root)
    registry.loc[mask, "candidate_library_status"] = f"repaired_evidence_{classification}"
    registry.loc[mask, "closure_state"] = "closed"
    repaired.write_csv(registry_path, registry)

    dependency_path = campaign / "campaign/downstream_dependency_map.csv"
    dependency = pd.read_csv(dependency_path).drop(columns=["status"], errors="ignore")
    mask = dependency.upstream_root.eq(str(repaired.SOURCE_ROOT))
    dependency.loc[mask, "replay_status"] = "repaired_screen_and_identity_bearing_downstream_replay_complete"
    repaired.write_csv(dependency_path, dependency)

    preservation_path = campaign / "campaign/hypothesis_preservation_ledger.csv"
    preservation = pd.read_csv(preservation_path).drop(columns=["next_allowed_action"], errors="ignore")
    mask = preservation.family.eq("RFBS")
    preservation.loc[mask, "preservation_status"] = "repaired_lineage_closed_not_rejected"
    preservation.loc[mask, "permitted_claim"] = "repaired train-only classification is decision-bearing subject to existing caps"
    preservation.loc[mask, "forbidden_claim"] = "prior quarantined result is decision-bearing"
    repaired.write_csv(preservation_path, preservation)

    matrix_path = campaign / "campaign/repair_completion_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    mask = matrix.family.eq("RFBS")
    for column in ("runner_migrated", "regression_tests_passed", "repaired_replay_complete", "downstream_replay_complete", "reconciled_and_closed"):
        matrix.loc[mask, column] = True
    matrix.loc[mask, "next_action"] = "none; lineage closed"
    matrix.loc[matrix.family.eq("Shared contract"), "next_action"] = "human campaign-closure review"
    repaired.write_csv(matrix_path, matrix)

    gate_path = campaign / "campaign/new_family_launch_gate.json"
    gate = json.loads(gate_path.read_text())
    gate.update({
        "new_family_launch_allowed": False,
        "unresolved_registry_count": 0,
        "directly_affected_roots": 0,
        "downstream_affected_roots": 0,
        "next_prompt_target": "Cross-family repair campaign closure and continuity reconciliation only",
        "rfbs_lineage_closed": True,
        "human_campaign_closure_review_required": True,
    })
    repaired.write_json(gate_path, gate)
    decision_path = campaign / "decision_summary.json"
    decision = json.loads(decision_path.read_text())
    decision.update({
        "directly_affected_roots": 0,
        "downstream_affected_roots": 0,
        "unresolved_registry_count": 0,
        "next_recommended_prompt_target": "Cross-family repair campaign closure and continuity reconciliation only",
        "rfbs_lineage_closed": True,
        "new_family_launch_allowed": False,
    })
    repaired.write_json(decision_path, decision)
    repaired.refresh_campaign_bundle()


def run(root: Path, screen_root: Path = SCREEN_ROOT) -> dict[str, Any]:
    screen = json.loads((screen_root / "decision_summary.json").read_text())
    if screen.get("status") != "complete":
        raise RuntimeError("repaired RFBS screen is not complete")
    decisions = pd.read_csv(screen_root / "decision/candidate_decisions.csv")
    additional = decisions[decisions.decision.eq("materialization_candidate")].definition_id.astype(str).tolist()
    target_ids = tuple(dict.fromkeys([*PRESERVED_IDS, *additional]))
    adjudication.TARGET_IDS = target_ids
    adjudication.SOURCE_ROOT = screen_root
    result = adjudication.run(root, screen_root)

    final_map = {
        "train_only_stability_review_candidate": "materialization_candidate",
        "fragile_context_sleeve": "fragile_context_sleeve",
        "current_translation_weak": "current_translation_weak",
        "focused_mechanical_repair_required": "focused_mechanical_repair_required",
    }
    classification = final_map[result["final_decision"]]
    address_source = screen_root / "audit/old_to_new_address_map.csv"
    (root / "audit").mkdir(exist_ok=True)
    shutil.copy2(address_source, root / "audit/old_to_new_address_map.csv")
    (root / "forensics").mkdir(exist_ok=True)
    shutil.copy2(screen_root / "forensics/exact_vs_imputed_funding.csv", root / "forensics/exact_vs_imputed_funding.csv")
    report = (
        f"# RFBS Lineage Reconciliation\n\nScreen root: `{screen_root}`. Downstream root: `{root}`. "
        f"Shared contract: `{repaired.CONTRACT_VERSION}`. Replayed definitions: {', '.join(target_ids)}. "
        f"Final frozen-lineage classification: `{classification}`. Prior RFBS conclusions remain provenance only; "
        "this replay uses repaired candidate identities, freshly frozen controls, and unchanged economics.\n"
    )
    (root / "RFBS_LINEAGE_RECONCILIATION_REPORT.md").write_text(report, encoding="utf-8")
    summary_path = root / "decision_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({
        "final_decision": classification,
        "final_lineage_classification": classification,
        "signal_state_contract_version": repaired.CONTRACT_VERSION,
        "repaired_screen_root": str(screen_root),
        "preserved_definitions": list(PRESERVED_IDS),
        "additional_unchanged_gate_definitions": [item for item in target_ids if item not in PRESERVED_IDS],
        "definitions_materialized": len(target_ids),
    })
    repaired.write_json(summary_path, summary)
    _refresh_bundle(root)
    if summary["status"] == "complete":
        _close_campaign(screen_root, root, classification)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--screen-root", type=Path, default=SCREEN_ROOT)
    args = parser.parse_args()
    result = run(args.run_root, args.screen_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
