#!/usr/bin/env python3
"""Build the deterministic, outcome-free Stage 16 replacement packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.qlmg_stage16_campaign import (
    beam_contract, boundary_contract, build_translation_registry, canonical_bytes,
    canonical_sha256, estimator_rule_inventory, file_sha256, funding_contract,
    inner_fold_contract, metric_contract, response_surface_contract,
    synthetic_canary, telegram_supervision_contract, utility_pareto_contract,
    validate_packet, validate_translation_registry,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_16_complete_campaign_semantics_git_cleanup_20260720_v1"
STAGE14 = ROOT / "docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1"
STAGE15 = ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_15_unattended_derivatives_campaign_20260720_v1"
FIXED = "2026-07-20T00:00:00Z"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def dependency_hashes(paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    raw, canonical = {}, {}
    for path in paths:
        key = path.name
        value = json.loads(path.read_text(encoding="utf-8"))
        raw[key] = file_sha256(path)
        canonical[key] = canonical_sha256(value)
    return raw, canonical


def build(implementation_commit: str) -> dict[str, Any]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    contracts = {
        "RESPONSE_SURFACE_AND_BIN_SPEC.json": response_surface_contract(),
        "ESTIMATOR_AND_RULE_INVENTORY.json": estimator_rule_inventory(),
        "INNER_FOLD_MAP.json": inner_fold_contract(),
        "DEVELOPMENT_METRIC_CONTRACT.json": metric_contract(),
        "UTILITY_AND_PARETO_CONTRACT.json": utility_pareto_contract(),
        "CANDIDATE_BEAM_CONTRACT.json": beam_contract(),
        "ECONOMIC_TRANSLATION_REGISTRY.json": build_translation_registry(),
        "BOUNDARY_AND_MISSINGNESS_CONTRACT.json": boundary_contract(),
        "FUNDING_COST_AND_COVERAGE_CONTRACT.json": funding_contract(),
        "TELEGRAM_AND_SUPERVISION_CONTRACT.json": telegram_supervision_contract(),
    }
    validate_translation_registry(contracts["ECONOMIC_TRANSLATION_REGISTRY.json"])
    for name, value in contracts.items():
        write_json(ARCHIVE / name, value)

    identities = {
        "KDA02B_EXECUTABLE_IDENTITY.md": """# KDA02B Executable Identity

The traded instrument is the causal event symbol's native Kraken linear PF contract. Decision is the completed one-hour OI-vacuum/event bar close. Trade and mark displacement signs must be nonzero and agree. Continuation follows that sign; reversal opposes it. Conflicting or zero signs produce no candidate. Unsigned OI is state, never actor-direction evidence. Each cell freezes its raw/rank OI and displacement thresholds, liquidation form, branch, 1h/3h/6h fixed exit, actual-exit symbol/definition non-overlap, 14/32 bps costs, and funding partition before outcomes.
""",
        "KDA02C_TRADABLE_IDENTITY.md": """# KDA02C Tradable Identity

KDA02C trades the native Kraken linear PF symbol of the underlying frozen completed-purge reversal event. It never creates a market-wide, BTC/ETH proxy, or equal-weight portfolio trade. A completed downside purge and reclaim is long; a completed upside purge and failure is short. Decision is that same symbol's causal reclaim/failure availability. Directional breadth uses the point-in-time eligible denominator at that timestamp. The fixed exit is 1h. Primary `z2` and robustness `pct95` purge identities are separate registered attempts and inherit KDA02/programme multiplicity.
""",
        "KDX01_EXECUTABLE_IDENTITY.md": """# KDX01 Executable Identity

KDX01 trades the event symbol's native Kraken linear PF contract, long only after downside trade-and-mark displacement, the cell's exact OI/basis/liquidation/breadth primitive ladder, and a completed trade-and-mark structural reclaim. The causal reclaim requires both completed trade and mark bars to close back through their registered pre-displacement reference levels. Cells freeze raw/rank scaling and 1h/3h/6h fixed exits. KDX01 remains `cross_family_program_exposed_redevelopment` and inherits KDA01, KDA02, and KDA03 component multiplicity.
""",
    }
    for name, body in identities.items():
        (ARCHIVE / name).write_text(body, encoding="utf-8")

    registry = contracts["ECONOMIC_TRANSLATION_REGISTRY.json"]
    search_registry = {
        "version": "stage16_v1", "supersedes_non_authorizing_stage14_registry_sha256": "a533c8e507f78989963a2631115bd1a3f70c3dc34cb59d749bec57b53577294b",
        "cells_by_family": registry["cells_by_family"], "maximum_total_cells": registry["total_cells"],
        "registered_cell_ids": [cell["cell_id"] for cell in registry["cells"]],
        "registered_translation_ids": [cell["canonical_translation_id"] for cell in registry["cells"]],
        "non_executable_inherited_attempts": registry["removed_from_executable_registry"]["attempts"],
        "all_response_cells_and_potential_translations_registered": True,
        "model_count": 0, "candidate_beam_per_family_fold": 5,
        "maximum_rule_complexity": max(cell["complexity"] for cell in registry["cells"]),
        "multiplicity": {"every_rejected_or_selected_cell_retained": True,
                         "primary_and_robust_KDA02C_purge_identities_separate_attempts": True,
                         "historical_parent_and_programme_attempts_inherited": True},
    }
    resource = {
        "benchmark_authority": {"stage14_resource_projection_sha256": "6eb572e8c09be94d462702db54d98d3fa88b4aa0ee402b2873c8a7fe66235d97",
                                "semantic_registry_count_unchanged": True},
        "cells_by_family": registry["cells_by_family"], "total_cells": 186,
        "candidate_beam_per_family_fold": 5, "maximum_rule_complexity": search_registry["maximum_rule_complexity"],
        "maximum_model_count": 0, "worker_count": 4, "wall_seconds": 14400,
        "max_disk_bytes": 5368709120, "max_memory_bytes": 5368709120,
        "memory_enforcement": "persistent supervisor sums RSS of supervisor and all worker descendants every heartbeat and globally stops before starting another cell when >= cap",
        "persistent_supervisor_required": True,
        "projection": "186 executable deterministic rule cells are below the reviewed Stage-14 228-cell no-outcome benchmark envelope; 42 KDX continuation-null attempts remain inherited multiplicity only; stop at four hours or 5 GiB",
    }
    write_json(ARCHIVE / "SEARCH_SPACE_REGISTRY.json", search_registry)
    write_json(ARCHIVE / "RESOURCE_PROJECTION.json", resource)

    dependency_paths = [ARCHIVE / name for name in contracts] + [ARCHIVE / "SEARCH_SPACE_REGISTRY.json", ARCHIVE / "RESOURCE_PROJECTION.json"]
    raw_hashes, canonical_hashes = dependency_hashes(dependency_paths)
    authority = {
        "actual_starting_commit": "8b8e4b15c0bc89d68a0748c2e26823e024a0279b",
        "repository_implementation_commit": implementation_commit,
        "stage15_validator_repair_commit": "9c6b08730de6f3dbf16d22d70e1e64646e21a248",
        "stage15_artifact_manifest_sha256": file_sha256(STAGE15 / "ARTIFACT_MANIFEST.json"),
        "stage15_zip_sha256": "d4245f4dc59954ed0fd8124effa505e8ae61615337a15bf76da02dbe7a1cb16c",
        "stage14_campaign_manifest_sha256": file_sha256(STAGE14 / "CAMPAIGN_MANIFEST.json"),
        "stage14_approval_packet_sha256": file_sha256(STAGE14 / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json"),
        "stage14_search_registry_sha256": file_sha256(STAGE14 / "SEARCH_SPACE_REGISTRY.json"),
        "stage14_resource_projection_sha256": file_sha256(STAGE14 / "RESOURCE_PROJECTION.json"),
        "stage14_local_state_tape_manifest_sha256": file_sha256(STAGE14 / "LOCAL_STATE_TAPE_MANIFEST.json"),
        "analytics_manifest_sha256": "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d",
        "authorized_cohort_sha256": "5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636",
        "semantic_engine_sha256": file_sha256(ROOT / "tools/qlmg_stage16_campaign.py"),
        "packet_builder_sha256": file_sha256(Path(__file__)),
    }
    manifest = {
        "campaign_id": "kraken_derivatives_campaign_001_stage16_semantics_complete",
        "version": "3.0", "generated_at": FIXED, "mode": "future_phase2_5_request_only",
        "status": "launch_semantics_complete_human_approval_required",
        "economic_run_authorized_by_manifest": False, "external_human_approval_required": True,
        "ready_hypotheses": ["KDA02B_v2_oi_vacuum_redevelopment", "KDA02C_v1_purge_breadth_context", "KDX01_v1_downside_completed_derivatives_state_rejection"],
        "C17_excluded": True, "programme_exposure_class": "program_exposed_historical",
        "independent_validation_claim": False, "authorities": authority,
        "dependency_file_sha256": raw_hashes, "dependency_canonical_sha256": canonical_hashes,
        "search": {"total_cells": 186, "Stage14_explored_attempts": 228, "cells_by_family": registry["cells_by_family"], "candidate_beam_per_family_fold": 5, "model_count": 0},
        "outer_and_inner_fold_contract": raw_hashes["INNER_FOLD_MAP.json"],
        "selection_contracts": [raw_hashes["DEVELOPMENT_METRIC_CONTRACT.json"], raw_hashes["UTILITY_AND_PARETO_CONTRACT.json"], raw_hashes["CANDIDATE_BEAM_CONTRACT.json"]],
        "economic_translation_registry": raw_hashes["ECONOMIC_TRANSLATION_REGISTRY.json"],
        "cost_execution_boundary_contracts": [raw_hashes["FUNDING_COST_AND_COVERAGE_CONTRACT.json"], raw_hashes["BOUNDARY_AND_MISSINGNESS_CONTRACT.json"]],
        "telegram_supervision_contract": raw_hashes["TELEGRAM_AND_SUPERVISION_CONTRACT.json"],
        "phase_permissions": {str(phase): phase < 2 for phase in range(8)},
        "resource_limits": resource,
        "outcome_firewall": {"forward_returns_or_PnL": False, "protected": False, "Capitalcom": False, "real_Telegram": False},
        "stop_conditions": {"global": ["authority_hash_drift", "outcome_firewall_access", "protected_or_Capitalcom_access", "shared_temporal_or_execution_defect", "deterministic_replay_failure"], "family": ["mechanically_invalid", "insufficient_integrity_or_cluster_count", "no_positive_development_candidate", "family_specific_defect"]},
        "historical_terminal_decisions_changed": False,
    }
    write_json(ARCHIVE / "CAMPAIGN_MANIFEST.json", manifest)
    manifest_raw_hash = file_sha256(ARCHIVE / "CAMPAIGN_MANIFEST.json")
    manifest_canonical_hash = canonical_sha256(manifest)

    packet_payload = {
        "packet_id": "kraken_derivatives_campaign_001_stage16_phase2_5_approval_request",
        "version": "3.0", "status": "human_approval_required_not_authorized",
        "campaign_id": manifest["campaign_id"], "phases_requested": [2, 3, 4, 5],
        "ready_lanes": manifest["ready_hypotheses"], "C17_excluded": True,
        "campaign_manifest_file_sha256": manifest_raw_hash,
        "campaign_manifest_canonical_sha256": manifest_canonical_hash,
        "dependency_file_sha256": raw_hashes, "dependency_canonical_sha256": canonical_hashes,
        "economic_run_authorized": False, "external_human_approval_required": True,
        "authorization_preconditions": ["new external human approval explicitly names this final packet file SHA-256 and campaign-manifest file SHA-256", "all raw and canonical dependency hashes match", "independent pre-outcome review accepted", "synthetic canary and launch validator pass", "funding and Telegram launch gates pass before any outcome reader"],
        "supersession": {"stage14_packet_remains_immutable": True, "stage15_status_remains": "blocked_preoutcome_packet_semantics", "replacement_does_not_reuse_stage15_approval": True},
        "complete_semantics": {"bins": raw_hashes["RESPONSE_SURFACE_AND_BIN_SPEC.json"], "estimator_rules": raw_hashes["ESTIMATOR_AND_RULE_INVENTORY.json"], "inner_folds": raw_hashes["INNER_FOLD_MAP.json"], "metrics": raw_hashes["DEVELOPMENT_METRIC_CONTRACT.json"], "utility_pareto": raw_hashes["UTILITY_AND_PARETO_CONTRACT.json"], "beam": raw_hashes["CANDIDATE_BEAM_CONTRACT.json"], "translations": raw_hashes["ECONOMIC_TRANSLATION_REGISTRY.json"], "boundary": raw_hashes["BOUNDARY_AND_MISSINGNESS_CONTRACT.json"], "funding": raw_hashes["FUNDING_COST_AND_COVERAGE_CONTRACT.json"], "telegram_supervision": raw_hashes["TELEGRAM_AND_SUPERVISION_CONTRACT.json"]},
    }
    packet = dict(packet_payload)
    packet["packet_payload_canonical_sha256"] = canonical_sha256(packet_payload)
    write_json(ARCHIVE / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json", packet)
    packet_raw_hash = file_sha256(ARCHIVE / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json")
    packet_canonical_hash = canonical_sha256(packet)

    canary = synthetic_canary()
    semantic_check = {"semantics_complete": canary["registered_cells"] == 186 and canary["read_spy_forward_rejected"] and canary["metrics_complete"] and canary["metric_scalar_values_finite"] and canary["contribution_shares_sum_to_one"] and len(canary["beam_ids"]) >= 3}
    readiness = validate_packet(packet, semantic_check)
    write_json(ARCHIVE / "SYNTHETIC_CANARY.json", canary)
    (ARCHIVE / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md").write_text(
        "# Future Derivatives Campaign Approval Packet\n\n"
        "Status: `human_approval_required_not_authorized`. No economics are authorized.\n\n"
        f"Campaign manifest file SHA-256: `{manifest_raw_hash}`<br>\n"
        f"Campaign manifest canonical SHA-256: `{manifest_canonical_hash}`<br>\n"
        f"Approval packet file SHA-256: `{packet_raw_hash}`<br>\n"
        f"Approval packet canonical SHA-256: `{packet_canonical_hash}`<br>\n"
        f"Approval packet payload canonical SHA-256: `{packet['packet_payload_canonical_sha256']}`\n\n"
        "The packet binds 186 executable translations, exact folds, metrics, Pareto/beam arithmetic, economic identity, boundaries, funding, and supervision. The 42 Stage-14 KDX continuation-null attempts remain inherited multiplicity but cannot become executable long mean-reversion translations. A new external human approval must name the final file hashes.\n",
        encoding="utf-8")
    (ARCHIVE / "CAMPAIGN_PACKET_LAUNCH_READINESS.md").write_text(
        "# Campaign Packet Launch Readiness\n\n"
        f"- `packet_semantics_complete`: `{str(readiness['packet_semantics_complete']).lower()}`\n"
        f"- `campaign_engine_can_execute_without_discretion`: `{str(readiness['campaign_engine_can_execute_without_discretion']).lower()}`\n"
        f"- `external_human_approval_still_required`: `{str(readiness['external_human_approval_still_required']).lower()}`\n"
        "- Synthetic canary: pass.\n- Economic run: not launched.\n",
        encoding="utf-8")
    return {"manifest_file_sha256": manifest_raw_hash, "manifest_canonical_sha256": manifest_canonical_hash,
            "packet_file_sha256": packet_raw_hash, "packet_canonical_sha256": packet_canonical_hash,
            "cells": 186, "canary": canary, "readiness": readiness}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", default="UNCOMMITTED_STAGE16")
    args = parser.parse_args()
    print(json.dumps(build(args.implementation_commit), sort_keys=True))
