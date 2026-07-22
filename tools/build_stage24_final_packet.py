#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.core_liquid_campaign.canonical import atomic_write_bytes, atomic_write_json, canonical_hash, sha256_file
from tools.core_liquid_campaign.campaign import require_population_authority_bindings
from tools.core_liquid_campaign.packet import _code_inventory
from tools.core_liquid_campaign.schema import CAMPAIGN_ID


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _require_reused_review_delta(
    repository_root: Path, head: str, review_commit: str, delta_audit_path: Path | None,
) -> dict[str, Any]:
    if delta_audit_path is None or not delta_audit_path.is_file():
        raise RuntimeError("reused full review requires a physical launch-interface delta audit")
    if subprocess.run(
        ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", review_commit, head],
        check=False,
    ).returncode != 0:
        raise RuntimeError("reused full review commit is not an ancestor of the packet implementation")
    audit = json.loads(delta_audit_path.read_text(encoding="utf-8"))
    changed = _git(repository_root, "diff", "--name-only", f"{review_commit}..{head}").splitlines()
    records = audit.get("changed_files")
    if (
        audit.get("schema") != "stage24_launch_packet_interface_delta_audit_v1"
        or audit.get("status") != "pass"
        or audit.get("base_reviewed_commit") != review_commit
        or audit.get("target_commit") != head
        or not isinstance(records, list)
        or sorted(str(record.get("path")) for record in records) != sorted(changed)
        or any(
            record.get("classification") not in {
                "packet_builder_interface", "fail_closed_runtime_interface",
                "shadow_no_outcome_canary", "focused_test", "task_archive",
            }
            for record in records
        )
        or any(audit.get(field) is not False for field in (
            "economic_semantics_changed", "registry_identity_changed", "control_identity_changed",
            "selection_arithmetic_changed", "cache_or_input_authority_changed",
            "protected_data_policy_changed",
        ))
        or audit.get("reused_evidence_verified") is not True
        or audit.get("zero_economic_units_executed") is not True
    ):
        raise RuntimeError("launch-interface delta audit is incomplete or broadened")
    return audit


def _inventory(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def _population_authority_mappings(
    output_root: Path,
    primary: dict[str, str],
    *,
    launch_population_authority: Path | None = None,
    kda02b_population_authority: Path | None = None,
) -> dict[str, dict[str, Any]]:
    launch_path = launch_population_authority or output_root / "LAUNCH_POPULATION_AUTHORITY.json"
    kda_path = kda02b_population_authority or output_root / "KDA02B_LAZY_POPULATION_MANIFEST.json"
    mappings = {
        "launch_population_authority": {
            "path": str(launch_path.resolve()),
            "bytes": launch_path.stat().st_size,
            "role": "complete_A1_A4_launch_population_authority",
            "sha256": primary["launch_population_authority"],
        },
        "kda02b_lazy_population_authority": {
            "path": str(kda_path.resolve()),
            "bytes": kda_path.stat().st_size,
            "role": "complete_KDA02B_launch_population_authority",
            "sha256": primary["kda02b_population_authority"],
        },
    }
    require_population_authority_bindings({"primary_hashes": primary, **mappings}, output_root)
    return mappings


def build(args: argparse.Namespace) -> dict[str, Any]:
    if _git(args.repository_root, "status", "--porcelain=v1"):
        raise RuntimeError("Stage 24 final packet requires a clean reviewed worktree")
    head = _git(args.repository_root, "rev-parse", "HEAD")
    if head != args.implementation_commit:
        raise RuntimeError("Stage 24 final packet commit differs from the reviewed implementation")
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or int(review.get("blocking_findings", -1)) != 0:
        raise RuntimeError("Stage 24 final packet requires an independent PASS with zero blocking findings")
    review_commit = str(review.get("implementation_commit", ""))
    delta_audit = None
    if review_commit != head:
        delta_audit = _require_reused_review_delta(args.repository_root, head, review_commit, args.delta_audit)
    for key, expected in (("economic_outcomes_opened", False), ("capitalcom_payload_opened", False)):
        if review.get(key) is not expected:
            raise RuntimeError(f"independent review firewall field differs: {key}")
    if int(review.get("protected_rows_opened", -1)) != 0:
        raise RuntimeError("independent review opened protected rows")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    gate_commit = str(gate.get("implementation_commit", ""))
    if (
        gate.get("status") != "pass"
        or gate_commit not in {head, review_commit}
        or (gate_commit != head and delta_audit is None)
    ):
        raise RuntimeError("production-readiness gate is not a PASS at the reviewed commit")
    cache = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    cache_artifacts = cache.get("artifacts", [])
    kda_artifacts = [
        row for row in cache_artifacts
        if row.get("campaign_partition", {}).get("phase") == "kda02b_adjudication"
    ]
    if (
        cache.get("artifact_inventory_sha256") != canonical_hash(cache_artifacts)
        or len(cache_artifacts) != 567
        or len(kda_artifacts) != 171
    ):
        raise RuntimeError("final cache authority does not contain the exact complete frame inventory")
    launch_population = json.loads(args.launch_population_authority.read_text(encoding="utf-8"))
    if launch_population.get("schema") != "stage24_launch_population_authority_v1" or launch_population.get("status") != "bound_outcome_free":
        raise RuntimeError("launch-population authority is invalid")
    kda_population = json.loads(args.kda_population_manifest.read_text(encoding="utf-8"))
    if (
        kda_population.get("schema") != "stage24_kda02b_lazy_population_index_v1"
        or kda_population.get("status") != "pass"
        or kda_population.get("economic_outcomes_opened") is not False
        or int(kda_population.get("protected_rows_opened", -1)) != 0
    ):
        raise RuntimeError("KDA02B population authority is invalid")
    reused = json.loads(args.reused_evidence_authority.read_text(encoding="utf-8"))
    if (
        reused.get("schema") != "stage24_reused_shadow_evidence_authority_v1"
        or reused.get("status") != "pass"
        or reused.get("economic_outcomes_opened") is not False
        or reused.get("markers_deleted_rewritten_or_recomputed") is not False
    ):
        raise RuntimeError("reused-evidence authority is invalid")
    reused_verification = json.loads(args.reused_startup_verification.read_text(encoding="utf-8"))
    if (
        reused_verification.get("schema") != "stage24_reused_evidence_startup_verification_v1"
        or reused_verification.get("status") != "pass"
        or int(reused_verification.get("total_file_count", -1)) != 6_373
        or int(reused_verification.get("unique_file_count", -1)) != 6_373
        or reused_verification.get("reused_evidence_authority_sha256") != sha256_file(args.reused_evidence_authority)
    ):
        raise RuntimeError("reused-evidence startup verification is invalid")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", "FINAL_EXECUTION_REGISTRY.jsonl",
        "FINAL_CONTROL_REGISTRY.jsonl", "A2_PARENT_COUNTERPART_REGISTRY.jsonl",
        "FOLD_GRAPH.json", "FAMILY_AXIS_SCHEMA.json",
    ):
        shutil.copy2(args.packet_root / name, args.output_root / name)
    shutil.copy2(args.execution_input_authority, args.output_root / "EXECUTION_INPUT_AUTHORITY.json")
    shutil.copy2(args.launch_population_authority, args.output_root / "LAUNCH_POPULATION_AUTHORITY.json")
    shutil.copy2(args.kda_population_manifest, args.output_root / "KDA02B_LAZY_POPULATION_MANIFEST.json")
    shutil.copy2(args.reused_evidence_authority, args.output_root / "REUSED_SHADOW_EVIDENCE_AUTHORITY.json")
    shutil.copy2(args.reused_startup_verification, args.output_root / "REUSED_EVIDENCE_STARTUP_VERIFICATION.json")
    if args.delta_audit is not None:
        shutil.copy2(args.delta_audit, args.output_root / "DELTA_IMPACT_AUDIT.json")
    kda_reconciliation_path = args.cache_manifest.parent.parent / "KDA02B_RECONCILIATION.json"
    if not kda_reconciliation_path.is_file():
        raise RuntimeError("final KDA02B reconciliation is absent")
    shutil.copy2(kda_reconciliation_path, args.output_root / "KDA02B_RECONCILIATION.json")
    shutil.copy2(args.gate, args.output_root / "PRODUCTION_READINESS_GATE.json")
    shutil.copy2(args.review, args.output_root / "FINAL_INDEPENDENT_REVIEW.json")
    code_inventory = _code_inventory(args.repository_root)
    atomic_write_json(args.output_root / "CODE_HASH_INVENTORY.json", code_inventory)

    build_report_path = args.cache_manifest.parent.parent / "PRODUCTION_FAMILY_INPUT_BUILD.json"
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    cache_authority = {
        "path": str(args.cache_manifest.resolve()),
        "bytes": args.cache_manifest.stat().st_size,
        "sha256": sha256_file(args.cache_manifest),
        "artifacts": len(cache_artifacts),
        "kda02b_artifacts": len(kda_artifacts),
        "components": len(cache.get("components", [])),
        "artifact_inventory_sha256": cache["artifact_inventory_sha256"],
        "component_inventory_sha256": cache.get("component_inventory_sha256"),
        "a1_population_table_manifest_sha256": build_report["a1_population_table_manifest_sha256"],
        "a3_population_table_manifest_sha256": build_report["a3_population_table_manifest_sha256"],
        "protected_rows": build_report["protected_rows"],
        "economic_outcomes_opened": build_report["economic_outcomes_opened"],
    }
    atomic_write_json(args.output_root / "CACHE_AUTHORITY_BINDING.json", cache_authority)
    local_dependencies = _inventory(args.output_root, excluded={
        "FINAL_CAMPAIGN_MANIFEST.json", "FINAL_HUMAN_APPROVAL_REQUEST.json",
        "FINAL_LAUNCH_TASK.md", "FINAL_PACKET_HASH_INVENTORY.json",
    })
    primary = {
        "strategy_registry": sha256_file(args.output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"),
        "execution_registry": sha256_file(args.output_root / "FINAL_EXECUTION_REGISTRY.jsonl"),
        "control_registry": sha256_file(args.output_root / "FINAL_CONTROL_REGISTRY.jsonl"),
        "a2_counterpart_registry": sha256_file(args.output_root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl"),
        "execution_input_authority": sha256_file(args.output_root / "EXECUTION_INPUT_AUTHORITY.json"),
        "launch_population_authority": sha256_file(args.output_root / "LAUNCH_POPULATION_AUTHORITY.json"),
        "kda02b_population_authority": sha256_file(args.output_root / "KDA02B_LAZY_POPULATION_MANIFEST.json"),
        "reused_evidence_authority": sha256_file(args.output_root / "REUSED_SHADOW_EVIDENCE_AUTHORITY.json"),
        "reused_evidence_startup_verification": sha256_file(args.output_root / "REUSED_EVIDENCE_STARTUP_VERIFICATION.json"),
        "cache_authority_manifest": sha256_file(args.cache_manifest),
        "code_inventory": sha256_file(args.output_root / "CODE_HASH_INVENTORY.json"),
        "production_readiness_gate": sha256_file(args.output_root / "PRODUCTION_READINESS_GATE.json"),
        "independent_review": sha256_file(args.output_root / "FINAL_INDEPENDENT_REVIEW.json"),
        "delta_impact_audit": sha256_file(args.output_root / "DELTA_IMPACT_AUDIT.json"),
        "kda02b_reconciliation": sha256_file(args.output_root / "KDA02B_RECONCILIATION.json"),
    }
    authority = json.loads((args.output_root / "EXECUTION_INPUT_AUTHORITY.json").read_text(encoding="utf-8"))
    population_authority_mappings = _population_authority_mappings(
        args.output_root,
        primary,
        launch_population_authority=args.launch_population_authority,
        kda02b_population_authority=args.kda_population_manifest,
    )
    manifest = {
        "schema": "stage24_final_executable_campaign_manifest_v1",
        "campaign_id": CAMPAIGN_ID,
        "repository": {"implementation_commit": head, "launch_from_clean_reviewed_descendant": True},
        "counts": {
            "registered_attempts": 11968,
            "unique_economic_executions": 11963,
            "controls": 800,
            "families": 5,
            "primary_outer_folds": 8,
            "kda02b_adjudication_folds": 9,
        },
        "primary_hashes": primary,
        **population_authority_mappings,
        "execution_input_authority": authority,
        "cache_authority": cache_authority,
        "resource_limits": {
            "workers": 4, "jobs_in_flight": 4, "aggregate_process_tree_rss_bytes": 10 * 1024**3,
            "campaign_output_bytes": 24 * 1024**3, "minimum_free_disk_bytes": 8 * 1024**3,
            "heartbeat_seconds": 1800, "graceful_stop_seconds": 300,
            "wall_time": "renewable_checkpoint_no_hard_stop",
        },
        "artifact_dependencies": local_dependencies,
        "independent_review": {"path": "FINAL_INDEPENDENT_REVIEW.json", "sha256": primary["independent_review"], "verdict": "PASS"},
        "authorization_state": "awaiting_one_exact_external_human_launch_approval",
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    }
    require_population_authority_bindings(manifest, args.output_root)
    atomic_write_json(args.output_root / "FINAL_CAMPAIGN_MANIFEST.json", manifest)
    request = {
        "schema": "stage24_final_human_approval_request_v1",
        "campaign_id": CAMPAIGN_ID,
        "final_campaign_manifest_sha256": sha256_file(args.output_root / "FINAL_CAMPAIGN_MANIFEST.json"),
        "repository_implementation_commit": head,
        "request": "Authorize one launch of the exact Stage-24-reviewed, hash-bound 11,968-row campaign and 800 controls under the attached manifest.",
        "authorized_if_approved": [
            "rankable 2023-2025 outcomes for the exact frozen campaign",
            "four-worker detached supervisor and exact conditional controls",
            "terminal reconciliation, independent post-run review, handoff and continuity",
        ],
        "still_prohibited": [
            "protected outcomes", "Capital.com payload", "new acquisition", "capture restart", "C17",
            "separate Phase 6", "account actions", "orders", "deployment", "live trading", "force push",
        ],
    }
    atomic_write_json(args.output_root / "FINAL_HUMAN_APPROVAL_REQUEST.json", request)
    launch = f"""# Exact Stage 24 launch task

Launch only after an external approval JSON binds all of:

- campaign `{CAMPAIGN_ID}`
- implementation commit `{head}`
- manifest `{sha256_file(args.output_root / 'FINAL_CAMPAIGN_MANIFEST.json')}`
- approval request `{sha256_file(args.output_root / 'FINAL_HUMAN_APPROVAL_REQUEST.json')}`
- execution registry `{primary['execution_registry']}`
- control registry `{primary['control_registry']}`
- cache authority `{primary['cache_authority_manifest']}`
- execution-input authority `{primary['execution_input_authority']}`
- launch-population authority `{primary['launch_population_authority']}`
- KDA02B population authority `{primary['kda02b_population_authority']}`
- reused-evidence authority `{primary['reused_evidence_authority']}`
- reused-evidence startup verification `{primary['reused_evidence_startup_verification']}`
- KDA02B reconciliation `{primary['kda02b_reconciliation']}`

Repeat authority, source, cache, resource, synthetic-canary and secure Telegram gates atomically. Launch through the reviewed detached service, require one reconciled real unit and the first scheduled heartbeat before health release, and preserve renewable checkpoint accounting without a hard wall stop. Do not access protected or Capital.com data.
"""
    atomic_write_bytes(args.output_root / "FINAL_LAUNCH_TASK.md", launch.encode("utf-8"))
    final_inventory = _inventory(args.output_root, excluded={"FINAL_PACKET_HASH_INVENTORY.json"})
    atomic_write_json(args.output_root / "FINAL_PACKET_HASH_INVENTORY.json", {
        "schema": "stage24_final_packet_hash_inventory_v1",
        "files": final_inventory,
        "inventory_sha256": canonical_hash(final_inventory),
    })
    return {
        "status": "final_packet_ready_for_exact_human_launch_approval",
        "implementation_commit": head,
        "manifest_sha256": sha256_file(args.output_root / "FINAL_CAMPAIGN_MANIFEST.json"),
        "approval_request_sha256": sha256_file(args.output_root / "FINAL_HUMAN_APPROVAL_REQUEST.json"),
        "registry_sha256": primary["strategy_registry"],
        "control_registry_sha256": primary["control_registry"],
        "cache_authority_manifest_sha256": primary["cache_authority_manifest"],
        "reused_evidence_authority_sha256": primary["reused_evidence_authority"],
        "launch_task_sha256": sha256_file(args.output_root / "FINAL_LAUNCH_TASK.md"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the final Stage 24 hash-bound launch-approval packet")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--execution-input-authority", type=Path, required=True)
    parser.add_argument("--launch-population-authority", type=Path, required=True)
    parser.add_argument("--kda-population-manifest", type=Path, required=True)
    parser.add_argument("--reused-evidence-authority", type=Path, required=True)
    parser.add_argument("--reused-startup-verification", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--delta-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
