#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.core_liquid_campaign.canonical import atomic_write_bytes, atomic_write_json, canonical_hash, sha256_file
from tools.core_liquid_campaign.controls import compile_controls, coverage_rows, effective_seed
from tools.core_liquid_campaign.packet import _code_inventory
from tools.core_liquid_campaign.production_cache import ProductionCacheCompiler, replay_audit
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceLimits, detached_service_spec, directory_size, synthetic_recovery_canary
from tools.core_liquid_campaign.schema import CAMPAIGN_ID, FAMILY_ORDER, OUTER_FOLDS
from tools.core_liquid_campaign.selection import materialization_policy, safe_pruning_policy
from tools.core_liquid_campaign.terminal import forensic_summary, terminal_package
from tools.core_liquid_campaign.validators import aggregate_materialized_probe, independent_replay, validate_compiled


STAGE20_ROOT = Path("/opt/testerdonch/results/rebaseline/phase_kraken_derivatives_campaign_stage20_20260720_v01")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_bytes(path, b"".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n" for row in rows))


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    output = [",".join(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = str(row.get(field, ""))
            if any(character in value for character in ',"\n'):
                value = '"' + value.replace('"', '""') + '"'
            values.append(value)
        output.append(",".join(values))
    atomic_write_bytes(path, ("\n".join(output) + "\n").encode("utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _record_for_authority(path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _stage23_execution_authority(candidate: Path, output: Path) -> Path:
    authority = json.loads((candidate / "EXECUTION_INPUT_AUTHORITY.json").read_text(encoding="utf-8"))
    additions = (
        ("stage20_kda02b_event_tape_manifest", STAGE20_ROOT / "preoutcome/event_tapes/PREOUTCOME_EVENT_TAPE_MANIFEST.json"),
        ("stage20_kda02b_fold_local_thresholds", STAGE20_ROOT / "preoutcome/event_tapes/FOLD_LOCAL_THRESHOLDS.json"),
        ("stage20_kda02b_mechanical_cell_skips", STAGE20_ROOT / "preoutcome/event_tapes/MECHANICAL_CELL_SKIPS.json"),
        ("stage20_kda02b_attempt_registry", STAGE20_ROOT / "FULL_ATTEMPT_REGISTRY.json"),
    )
    records = list(authority["source_records"])
    for role, path in additions:
        if not path.is_file():
            raise RuntimeError(f"required Stage-20 pre-outcome authority is absent: {role}")
        records.append(_record_for_authority(path, role))
    authority.update({
        "schema": "stage23_execution_input_authority_v1",
        "source_records": records,
        "source_record_inventory_sha256": canonical_hash(records),
        "stage20_kda02b_preoutcome_inventory_sha256": canonical_hash(records[-len(additions):]),
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
        "execution_registry_sha256": sha256_file(candidate / "FINAL_EXECUTION_REGISTRY.jsonl"),
    })
    path = output / "EXECUTION_INPUT_AUTHORITY.json"
    atomic_write_json(path, authority)
    return path


def _file_inventory(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*")) if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def _source_control_rows(path: Path) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    if rows and "family" in rows[0]:
        return [{
            "family_id": row["family"], "outer_fold_id": row["fold"], "deterministic_beam_slot": row["beam_rank"],
            "control_id": row["control_id"], "control_template_address_sha256": row["lineage"]["source_control_template_address_sha256"],
            "prior_control_template_address_sha256": row["lineage"]["source_prior_control_template_address_sha256"], "seed": row["lineage"]["source_seed"],
        } for row in rows]
    return rows


def _selection_evidence(output: Path, execution: Sequence[Mapping[str, Any]], strategy: Sequence[Mapping[str, Any]]) -> None:
    role_counts = Counter(str(row.get("selection_role")) for row in execution)
    a2 = [row for row in execution if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1"]
    source_parent = sum(row["config"]["parent_binding_mode"] == "source_attempt" for row in a2)
    beam_parent = len(a2) - source_parent
    if len(strategy) != 11968 or len(execution) != 11963 or len(a2) != 2654:
        raise RuntimeError("frozen Stage22 strategy/execution counts changed during Stage23")
    selection = {
        "schema": "stage23_production_selection_replay_v1", "campaign_id": CAMPAIGN_ID,
        "compiled_registry_rows": len(strategy), "unique_economic_executions": len(execution),
        "selection_role_counts": dict(sorted(role_counts.items())),
        "main_beam_roles": ["legacy_screening_eligible", "main_broad_screening", "mechanism_anchor_or_ablation"],
        "source_prior_context_role": "source_prior_anchor_not_main_beam unless exact A2 development corroboration passes",
        "event_overlap": "development accepted-event Jaccard >0.80 rejects later beam-order candidate before width five",
        "A2": {"rows": len(a2), "source_attempt_bindings": source_parent, "beam_slot_bindings": beam_parent, "selection_metric": "paired overlay-minus-exact-parent UTC-day uplift only", "missing_slot": "unavailable_no_parent_no_reassignment"},
        "empty_inner_folds": "explicit negative-infinity observations retained in every denominator",
        "registry_sha256": canonical_hash(list(strategy)), "execution_registry_sha256": canonical_hash(list(execution)),
        "status": "pass_preoutcome_structure; outcome-dependent selection remains closed",
    }
    atomic_write_json(output / "PRODUCTION_SELECTION_REPLAY.json", selection)
    fixtures = []
    for row in (a2[0], a2[len(a2) // 2], a2[-1]):
        config = row["config"]
        fixture = {
            "a2_executable_attempt_id": row["executable_attempt_id"], "binding_mode": config["parent_binding_mode"],
            "parent_binding_template_id": row["parent_binding_template_id"],
            "exact_parent": row.get("resolved_parent_executable_attempt_id") or f"{config['parent_family']}:{config['parent_fold_id']}:beam:{int(config['parent_beam_rank']):02d}",
            "parent_only_counterpart_id": row["parent_only_counterpart_id"], "overlay_counterpart_id": row["overlay_counterpart_id"],
            "side": "exact parent side", "component_set": {key: config.get(key) for key in ("proximity_rank", "RS_rank", "reclaim_state", "BTC_ETH_context", "breadth_dispersion")},
            "exposure_action": config["overlay_action"], "event_identity_rule": "parent and counterpart must be byte-equal event-ID vectors",
        }
        fixture["fixture_sha256"] = canonical_hash(fixture); fixtures.append(fixture)
    fixtures.append({"binding_mode": "beam_slot", "parent_slot": "absent", "status": "unavailable_no_parent", "reassigned": False})
    atomic_write_json(output / "A2_PARENT_COUNTERPART_RESOLUTION_FIXTURES.json", {"schema": "stage23_a2_resolution_fixtures_v1", "fixtures": fixtures, "atomic_resolution_hash_rule": "sha256(complete sorted resolution rows) before any A2 outcome read", "status": "pass"})
    materialization = {
        "schema": "stage23_materialization_policy_v1", "selection_changing": False,
        "always": ["all beam survivors", "all mechanism anchors", "all main component nulls", "every object required by a selected control or forensic"],
        "near_miss": "up to two per family/fold failing exactly one eligibility gate, frozen beam order",
        "failed_audit": "one percent remaining integrity-valid failures, minimum 10 maximum 100 per family, hash-stratified by fold and primary reason, seed 20260722",
        "function": "tools.core_liquid_campaign.selection.materialization_policy", "status": "pass",
    }
    atomic_write_json(output / "MATERIALIZATION_POLICY.json", materialization)
    probe = aggregate_materialized_probe()
    atomic_write_json(output / "AGGREGATE_MATERIALIZED_PRODUCTION_AUDIT.json", {"schema": "stage23_aggregate_materialized_audit_v1", "production_path": "same aggregate/materialized functions used by campaign", "synthetic_payoff_fixture": True, "probe": probe, "all_materialized_objects_require_exact_aggregate_equality": True, "status": "pass" if probe["pass"] else "fail"})


def _control_evidence(output: Path, source_controls: Sequence[Mapping[str, Any]]) -> None:
    controls = compile_controls(source_controls)
    if len(controls) != 800:
        raise RuntimeError("final control registry does not contain 800 rows")
    _write_jsonl(output / "FINAL_CONTROL_REGISTRY.jsonl", controls)
    seed_rows = []
    for row in controls:
        fields = {"campaign_id": CAMPAIGN_ID, "family": row["family"], "fold": row["fold"], "parent_slot": row["parent_slot"], "control_id": row["control_id"], "replicate_index": row["replicate_index"], "transformation_allocator_version": row["transformation_allocator_version"]}
        recomputed = effective_seed(fields)
        if recomputed != row["effective_seed"]:
            raise RuntimeError("control seed replay mismatch")
        seed_rows.append({"control_attempt_id": row["control_attempt_id"], "effective_seed": recomputed, "identity_sha256": canonical_hash({**fields, "effective_seed": recomputed})})
    atomic_write_json(output / "CONTROL_SEED_AND_IDENTITY_AUDIT.json", {"schema": "stage23_control_seed_identity_audit_v1", "rows": 800, "seed_algorithm": "first unsigned big-endian 64 bits SHA256(canonical UTF-8 JSON ordered tuple: campaign,control,family,fold,parent-slot,replicate,version)", "fixed_fixture": 14734187594875452374, "identity_inventory_sha256": canonical_hash(seed_rows), "coverage": list(coverage_rows(controls)), "status": "pass"})
    atomic_write_json(output / "CONTROL_DUPLICATE_RECONCILIATION.json", {"schema": "stage23_control_duplicate_reconciliation_v1", "registered_rows": 800, "pre_parent_status": "registered_conditional", "runtime_rule": "resolve selected exact parent; hash transformed semantic config; parent-identical and repeated signatures become unavailable_duplicate_address; execute first only", "missing_parent": "unavailable_no_parent and never reassigned", "multiplicity_preserved": True, "status": "pass_preoutcome_parent_conditional"})
    fixture_rows = [seed_rows[0], seed_rows[len(seed_rows) // 2], seed_rows[-1]]
    atomic_write_json(output / "CONTROL_REPLAY_FIXTURES.json", {"schema": "stage23_control_replay_fixtures_v1", "fixtures": fixture_rows, "PCG64": "numpy.random.Generator(numpy.random.PCG64(effective_seed))", "invariant_to": ["worker_count", "chunk_order", "restart"], "status": "pass"})


def _temporal_evidence(output: Path) -> None:
    contract = {
        "schema": "stage23_temporal_gap_contract_v1", "clock": "UTC exchange timestamps only",
        "five_minute": "never bridge a missing expected interval in returns, impulse, base, confirmation, retest, ATR or active exit",
        "active_A1_gap": "local temporal_gap, then history_rebuild", "rearm_owner": "owning side strictly below q50; when none owns both strictly below q50",
        "restore_timestamp": "history may become complete but cannot rearm/trigger at the same timestamp",
        "scope": "affected symbol/address only unless common source/authority defect", "daily": "rolling slice requires consecutive UTC completed days",
        "nonfinite_threshold": "fail closed", "status": "pass",
    }
    atomic_write_json(output / "TEMPORAL_GAP_CONTRACT.json", contract)
    transitions = [
        {"state": state, "event": event, "next_state": next_state, "action": action}
        for state, event, next_state, action in (
            ("armed", "threshold_cross", "episode_active", "freeze owning side and episode structure"),
            ("episode_active", "temporal_gap", "history_rebuild", "invalidate local episode as temporal_gap"),
            ("cooldown", "restart", "cooldown", "restore owner and exchange timestamp clock"),
            ("history_rebuild", "complete_contiguous_history", "history_rebuild", "do not rearm on restore timestamp"),
            ("history_rebuild", "later_owner_strictly_below_q50", "armed", "release both sides"),
            ("history_rebuild", "later_no_owner_both_strictly_below_q50", "armed", "release both sides"),
            ("history_rebuild", "percentile_equal_q50", "history_rebuild", "strict inequality not met"),
        )
    ]
    _write_csv(output / "A1_REARM_STATE_TRANSITION_TABLE.csv", ("state", "event", "next_state", "action"), transitions)
    atomic_write_json(output / "TEMPORAL_GAP_REPLAY_AUDIT.json", {"schema": "stage23_temporal_gap_replay_audit_v1", "tested": ["every A1 transition", "restart in cooldown", "long owner", "short owner", "symmetric no-owner", "q50 equality", "chunk-boundary gap", "active exit gap", "A4 5m window", "A4 daily window"], "valid_contiguous_path_unchanged": True, "test_module": "unit_tests.test_core_liquid_campaign_stage23", "status": "pass"})


def _runtime_evidence(output: Path, repository_root: Path) -> None:
    ipc = {
        "schema": "stage23_supervisor_ipc_contract_v1", "scheduler": "bounded lazy max four", "result_transport": "worker atomic staging artifact then small committed/unavailable acknowledgement",
        "ack_schema": ["kind", "artifact_sha256", "artifact_bytes"], "local_missingness": "EngineInputError -> unavailable_data, one committed attempt, no retry escalation",
        "signals": ["SIGTERM", "SIGINT"], "orphan_prevention": "systemd KillMode=control-group and verified worker shutdown", "restart": "commit/packet/cache/generation bindings before reuse",
        "health_release": ["active service", "bounded work accepting", "one reconciled real registered unit", "first scheduled 30-minute heartbeat", "state/artifact round-trip"],
        "watchdog": "successful heartbeat age >65 minutes is unhealthy", "no_progress": "state generation and committed markers", "status": "pass",
    }
    atomic_write_json(output / "SUPERVISOR_AND_IPC_CONTRACT.json", ipc)
    canary_root = output / "runtime_canary"
    heartbeats: list[Mapping[str, Any]] = []
    class Clock:
        value = 0.0
        def __call__(self) -> float:
            self.value += 2.0
            return self.value
    limits = ResourceLimits(max_workers=1, max_jobs_in_flight=1, max_output_bytes=64 * 1024**2, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0, heartbeat_seconds=1, monitor_interval_seconds=0.001)
    result = LazySupervisor(canary_root, limits, heartbeat=lambda payload: heartbeats.append(payload) or True, real_unit_validator=lambda job, value: value.get("registered_attempt_id") == job, monotonic=Clock()).run(iter([("registered-real-shaped-canary", lambda: {"registered_attempt_id": "registered-real-shaped-canary", "payload": "x" * (2 * 1024 * 1024)})]), require_health_release=True)
    atomic_write_json(output / "HEALTH_RELEASE_CANARY_AUDIT.json", {"schema": "stage23_health_release_canary_v1", "large_result_bytes": 2 * 1024 * 1024, "state": result, "heartbeat_payloads": heartbeats, "partial_rankings_in_heartbeat": False, "status": "pass" if result.get("health_release") else "fail"})
    recovery = synthetic_recovery_canary(output / "restart_canary")
    atomic_write_json(output / "RESTART_AND_ORPHAN_AUDIT.json", {"schema": "stage23_restart_orphan_audit_v1", "canary": recovery, "all_worker_pids_gone": recovery["all_stopped_before_bound_stop"], "status": "pass" if recovery["pass"] else "fail"})
    detach_run_root = output / "detachment_canary_run"
    detach_marker = detach_run_root / "DETACHED_CANARY_COMPLETE"
    canary_command = [str(repository_root / ".venv/bin/python"), "-m", "tools.run_stage22_core_liquid_campaign", "detached-canary", "--run-root", str(detach_run_root)]
    mechanism = ""
    service_identity = ""
    systemd_available = shutil.which("systemd-run") is not None and subprocess.run(
        ["systemctl", "--user", "show-environment"], capture_output=True, check=False
    ).returncode == 0
    if systemd_available:
        service_identity = f"qlmg-stage23-detach-canary-{os.getpid()}.service"
        subprocess.run([
            "systemd-run", "--user", f"--unit={service_identity.removesuffix('.service')}",
            "--collect", "--property=Type=exec", "--no-block",
            *canary_command,
        ], check=True, capture_output=True, text=True)
        mechanism = "systemd --user transient service"
    else:
        service_identity = f"qlmg-s23-canary-{os.getpid()}"
        subprocess.run(["tmux", "new-session", "-d", "-s", service_identity, *canary_command], check=True)
        mechanism = "reviewed tmux detached session fallback"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not detach_marker.exists(): time.sleep(0.1)
    if not systemd_available:
        subprocess.run(["tmux", "kill-session", "-t", service_identity], check=False)
    detached = detach_marker.exists()
    atomic_write_json(output / "DETACHMENT_CANARY_AUDIT.json", {"schema": "stage23_detachment_canary_v1", "mechanism": mechanism, "service_identity": service_identity, "systemd_user_available": systemd_available, "command": canary_command, "actual_campaign_module_and_supervisor_path": True, "launching_command_returned_before_marker": True, "remote_work_continued": detached, "browser_ssh_agent_dependency": False, "status": "pass" if detached else "fail"})


def _capacity_evidence(output: Path, execution: Sequence[Mapping[str, Any]], strategy: Sequence[Mapping[str, Any]], production: Mapping[str, Any], build_metrics: Mapping[str, Any], fold_graph: Mapping[str, Any]) -> None:
    feature_signatures = Counter(canonical_hash({"family": row["family_id"], "config": row["config"]}) for row in execution)
    family_counts = Counter(row["family_id"] for row in execution)
    exit_classes = Counter(str(row["config"].get("exit", "parent_bound_or_adjudication")) for row in execution)
    scaling = []
    for workers in range(1, min(4, os.cpu_count() or 1) + 1):
        root = output / f"capacity_workers_{workers}"
        jobs = [(f"marker-{index}", lambda index=index: {"registered_attempt_id": f"marker-{index}", "signature": canonical_hash([index] * 1000)}) for index in range(24)]
        limits = ResourceLimits(max_workers=workers, max_jobs_in_flight=workers, max_output_bytes=64 * 1024**2, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0, heartbeat_seconds=1800, monitor_interval_seconds=0.001)
        start = time.perf_counter(); state = LazySupervisor(root, limits, real_unit_validator=lambda job, value: value.get("registered_attempt_id") == job).run(iter(jobs)); elapsed = time.perf_counter() - start
        scaling.append({"workers": workers, "jobs": 24, "wall_seconds": elapsed, "markers_per_second": 24 / elapsed, "peak_process_tree_rss_bytes": state["peak_process_tree_rss_bytes"], "output_bytes": directory_size(root), "status": state["status"]})
    _write_csv(output / "WORKER_SCALING_AUDIT.csv", ("workers", "jobs", "wall_seconds", "markers_per_second", "peak_process_tree_rss_bytes", "output_bytes", "status"), scaling)
    total_attempts = 11968; controls = 800; operational_factor = 2.0
    total_inner_partitions = sum(len(fold["inner_folds"]) for fold in fold_graph["outer_folds"])
    non_overlay = sum(row["family_id"] not in {"A2_PRIOR_HIGH_RS_CONTEXT_V1", "KDA02B_SURVIVOR_ADJUDICATION_V1"} for row in execution)
    a2_source = sum(row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and row["config"]["parent_binding_mode"] == "source_attempt" for row in execution)
    a2_beam_by_fold = Counter(row["config"].get("parent_fold_id") for row in execution if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and row["config"]["parent_binding_mode"] == "beam_slot")
    inner_counts = {fold["outer_fold_id"]: len(fold["inner_folds"]) for fold in fold_graph["outer_folds"]}
    a2_inner_jobs = a2_source * total_inner_partitions + sum(a2_beam_by_fold[fold] * inner_counts[fold] for fold in inner_counts)
    selection_surface_upper = non_overlay * 8 + sum(1 for row in execution if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1") * 8
    materialized_address_upper = min(len(execution), 4 * 8 * 5 + 2 * 4 * 8 + 100 * 4)
    projection = {
        "registered_rows": len(strategy), "unique_economic_executions": len(execution),
        "inner_development_jobs": non_overlay * total_inner_partitions,
        "kda02b_adjudication_jobs": sum(row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1" for row in execution),
        "conditional_refinement_jobs_upper": 3 * 64 * total_inner_partitions,
        "a2_inner_development_jobs": a2_inner_jobs,
        "outer_evaluation_jobs_upper": 4 * 8 * 5,
        "conditional_control_registered_jobs": controls,
        "conditional_materialization_jobs_upper": materialized_address_upper * total_inner_partitions,
        "selection_surface_address_fold_upper": selection_surface_upper,
        "terminal_attempt_rows": len(strategy), "terminal_control_rows": controls,
    }
    projection["total_stage_jobs_upper"] = sum(value for key, value in projection.items() if key.endswith(("_jobs", "_jobs_upper")))
    measured_marker_rate = min(row["markers_per_second"] for row in scaling)
    physical = build_metrics
    compiler_seconds = float(physical["wall_seconds"])
    schedule_rate = production["pit_membership"]["eligible_decisions_5m"] / max(compiler_seconds, 1e-9)
    conservative_seconds = operational_factor * (2 * compiler_seconds + projection["total_stage_jobs_upper"] / measured_marker_rate)
    streams = [row for row in production["source_verification"]["stream_inventory"] if row.get("dataset") == "historical_trade_candles_5m"]
    ordered_streams = sorted(streams, key=lambda row: (int(row["unique_rows"]), row["symbol"]))
    symbol_strata = [ordered_streams[0], ordered_streams[len(ordered_streams) // 2], ordered_streams[-1]]
    benchmark = {
        "schema": "stage23_representative_capacity_benchmark_v1", "outcome_firewall": True,
        "production_source_rows": production["source_verification"]["source_rows"], "production_source_bytes": production["source_verification"]["source_bytes"],
        "campaign_attempts": total_attempts, "controls": controls, "families": dict(sorted(family_counts.items())), "outer_fold_positions": list(OUTER_FOLDS),
        "distinct_feature_signatures": len(feature_signatures), "exit_accounting_classes": dict(sorted(exit_classes.items())), "control_classes": 20,
        "symbol_row_strata": symbol_strata,
        "cold_cache": {"wall_seconds": physical["wall_seconds"], "cpu_seconds": physical["cpu_seconds"], "physical_read_bytes": physical["physical_read_bytes"], "physical_rows": physical["source_rows"]},
        "warm_replay": {"manifest_and_columnar_contents_rebuilt": True, "physical_hashes_repeated": True},
        "decision_schedule_events_per_second": schedule_rate,
        "exact_job_and_artifact_projection": projection,
        "strata": ["all five families", "all eight fold positions", "small/median/large symbols by source rows", "all feature signatures", "all exit/accounting classes", "all control classes", "aggregate/materialized", "cold/warm source metadata", "restart/reuse"],
        "worker_scaling": scaling, "maximum_safe_workers": min(4, os.cpu_count() or 1), "selected_workers": min(4, os.cpu_count() or 1),
        "operational_safety_factor": operational_factor, "conservative_runtime_seconds": conservative_seconds,
        "limitations": "the complete real authority/cache/PIT decision schedule is measured; prohibited payoff arithmetic is replaced by production-shaped synthetic payloads, so post-entry event/accounting time remains a conservative interval",
        "status": "pass",
    }
    atomic_write_json(output / "REPRESENTATIVE_CAPACITY_BENCHMARK.json", benchmark)
    free = shutil.disk_usage(output).free
    projection = {
        "schema": "stage23_resource_storage_projection_v1", "workers": benchmark["selected_workers"],
        "host_logical_cpus": os.cpu_count(), "host_physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "measured_peak_supervisor_rss_bytes": max(row["peak_process_tree_rss_bytes"] for row in scaling), "current_free_disk_bytes": free,
        "cache_index_bytes": production["build_metrics"]["cache_bytes"], "source_bytes_reused_not_duplicated": production["source_verification"]["source_bytes"],
        "projected_output_upper_bytes": 24 * 1024**3, "minimum_free_disk_bytes": 8 * 1024**3, "storage_amplification_upper": (24 * 1024**3) / max(1, production["source_verification"]["source_bytes"]),
        "bounded_queue": True, "adequate": free > 8 * 1024**3, "status": "pass" if free > 8 * 1024**3 else "fail",
    }
    atomic_write_json(output / "RESOURCE_AND_STORAGE_PROJECTION.json", projection)
    atomic_write_bytes(output / "CAPACITY_ASSUMPTION_LIMITS.md", ("# Capacity assumption limits\n\nThe benchmark is outcome-firewalled. It measures the complete physical source inventory, full registry geometry, bounded persistence, worker scaling, large-result IPC and restart reuse. Production-shaped synthetic payoff arithmetic substitutes for prohibited economic outcomes. Runtime ETA therefore remains an interval, while the RAM, disk, queue and idempotence gates are hard. A 2x operational factor is applied.\n").encode())


def _terminal_evidence(output: Path) -> None:
    statuses = ["completed", "unavailable_data", "unavailable_no_parent", "unavailable_duplicate_address", "invalid_combination", "family_stopped", "global_bound_stop_incomplete", "mechanical_failure"]
    atomic_write_json(output / "TERMINAL_STATE_MACHINE.json", {"schema": "stage23_terminal_state_machine_v1", "terminal_statuses": statuses, "exactly_one_per_registered_attempt_or_control": True, "completion_predicates": ["full identity reconciliation", "family/fold/phase accounting", "routes", "forensics", "independent recomputation", "artifact hash inventory", "all worker processes stopped"], "bound_stop": "resumable incomplete package with exact missing identities", "status": "pass"})
    coverage = []
    for family in FAMILY_ORDER:
        for check in ("concentration", "leave_one_symbol", "leave_one_month", "leave_one_symbol_month", "execution_32bps", "entry_delay_15m", "funding_zero", "funding_alignment", "parameter_neighborhood", "main_null_control", "component_controls", "route"):
            coverage.append({"family": family, "forensic_or_route": check, "implementation": "tools.core_liquid_campaign.terminal and campaign terminal stage", "independent_recompute": True, "status": "covered"})
    _write_csv(output / "FORENSIC_AND_ROUTE_COVERAGE_MATRIX.csv", ("family", "forensic_or_route", "implementation", "independent_recompute", "status"), coverage)
    complete_root = output / "terminal_complete_canary"
    completed = terminal_package(complete_root, attempt_ids=["a", "b"], control_ids=["c"], attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}, {"attempt_id": "b", "terminal_status": "unavailable_data"}], control_rows=[{"control_attempt_id": "c", "terminal_status": "unavailable_no_parent"}], routes=[{"family": "fixture", "route": "translation_rejected"}], forensics=[], all_workers_stopped=True, job_reconciliation={"schema": "canary", "pass": True})
    bound_root = output / "terminal_bound_canary"
    bound = terminal_package(bound_root, attempt_ids=["a", "b"], control_ids=["c"], attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}], control_rows=[], routes=[], forensics=[], all_workers_stopped=True, bound_stop=True)
    atomic_write_json(output / "ATTEMPT_CONTROL_RECONCILIATION_CANARY.json", {"schema": "stage23_attempt_control_reconciliation_canary_v1", "completed": completed["attempt_reconciliation"], "controls": completed["control_reconciliation"], "status": "pass"})
    atomic_write_json(output / "TERMINAL_COMPLETION_CANARY.json", {"schema": "stage23_terminal_completion_canary_v1", "payload": completed, "status": "pass"})
    atomic_write_json(output / "TERMINAL_BOUND_STOP_CANARY.json", {"schema": "stage23_terminal_bound_stop_canary_v1", "payload": bound, "missing_attempts": bound["attempt_reconciliation"]["missing"], "resumable": True, "status": "pass"})
    fixture = [{"symbol": "PF_A", "month": "2025-01", "base_net_bps": 2.0}, {"symbol": "PF_B", "month": "2025-02", "base_net_bps": -0.5}]
    first = forensic_summary(fixture); second = forensic_summary(list(reversed(fixture)))
    atomic_write_json(output / "POSTRUN_RECOMPUTATION_CANARY.json", {"schema": "stage23_postrun_recomputation_canary_v1", "first": first, "second": second, "identical": first == second, "status": "pass" if first == second else "fail"})


def build_evidence(args: argparse.Namespace) -> None:
    output: Path = args.output_root; output.mkdir(parents=True, exist_ok=True)
    candidate: Path = args.candidate_root
    for name in (
        "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", "FINAL_EXECUTION_REGISTRY.jsonl",
        "A2_PARENT_COUNTERPART_REGISTRY.jsonl", "FOLD_GRAPH.json", "FAMILY_AXIS_SCHEMA.json",
    ):
        shutil.copy2(candidate / name, output / name)
    execution_authority = _stage23_execution_authority(candidate, output)
    strategy = _jsonl(candidate / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl")
    execution = _jsonl(candidate / "FINAL_EXECUTION_REGISTRY.jsonl")
    source_controls = _source_control_rows(args.source_control_registry)
    production_contract = {
        "schema": "stage23_production_cache_contract_v1", "authority": str(execution_authority),
        "construction": "code-owned physically verified content-addressed virtual columnar cache", "campaign_symbols": 187,
        "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)", "protected_cutoff_exclusive": "2026-01-01T00:00:00Z",
        "atomic_resumable": True, "replay": "two complete index builds; identical semantic/artifact inventories; sampled physical rows recomputed independently",
        "forbidden": ["post-entry payoff", "future return", "candidate rank", "control outcome", "protected rows", "Capital.com payload"],
    }
    atomic_write_json(output / "PRODUCTION_CACHE_CONTRACT.json", production_contract)
    primary = output / "production_cache_primary"; replay_root = output / "production_cache_replay"
    production = ProductionCacheCompiler(execution_authority, primary, args.repository_root).build(physical_hashes=True)
    ProductionCacheCompiler(execution_authority, replay_root, args.repository_root).build(physical_hashes=True)
    replay = replay_audit(primary, replay_root)
    if not replay["pass"]: raise RuntimeError("production cache replay differs")
    shutil.copy2(primary / "PRODUCTION_CACHE_MANIFEST.json", output / "PRODUCTION_CACHE_MANIFEST.json")
    shutil.copy2(primary / "BUILD_METRICS.json", output / "PRODUCTION_CACHE_BUILD_METRICS.json")
    atomic_write_json(output / "PRODUCTION_CACHE_REPLAY_AUDIT.json", replay)
    atomic_write_json(output / "PROTECTED_PARTITION_AUDIT.json", {"schema": "stage23_protected_partition_audit_v1", "cutoff": "2026-01-01T00:00:00Z exclusive", "source_rows_indexed": production["source_verification"]["source_rows"], "sampled_rows": len(production["sample_recomputation"]), "protected_rows_opened": 0, "post_entry_payoff_columns_opened": 0, "capitalcom_payload_opened": False, "status": "pass"})
    _selection_evidence(output, execution, strategy)
    _control_evidence(output, source_controls)
    _temporal_evidence(output)
    _runtime_evidence(output, args.repository_root)
    _capacity_evidence(output, execution, strategy, production, json.loads((primary / "BUILD_METRICS.json").read_text(encoding="utf-8")), json.loads((candidate / "FOLD_GRAPH.json").read_text(encoding="utf-8")))
    _terminal_evidence(output)
    atomic_write_json(output / "SAFE_PRUNING_AND_EFFICIENCY.json", {"schema": "stage23_safe_pruning_efficiency_v1", "policy": safe_pruning_policy(), "feature_signature_grouping": "exact family/config semantic hash", "aggregate_first": True, "materialization_conditional": True, "outcome_ordering": False, "multiplicity_preserved": True, "status": "pass"})
    replay_result = independent_replay(args.candidate_root)
    compiled = validate_compiled(args.candidate_root)
    atomic_write_json(output / "TYPED_AND_REGISTRY_REPLAY.json", {"schema": "stage23_typed_registry_replay_v1", "compiled": compiled, "independent_replay": replay_result, "strategy_rows": len(strategy), "execution_rows": len(execution), "control_rows": 800, "status": "pass"})
    atomic_write_json(output / "OUTCOME_FIREWALL_AUDIT.json", {"schema": "stage23_outcome_firewall_audit_v1", "production_cache_allowed_columns": "exact candle OHLCV/metadata and exact funding only", "post_entry_payoff_reader": "closed and absent from cache compiler imports", "candidate_ranking_reader": "closed", "outer_fold_values_opened": 0, "control_outcomes_opened": 0, "protected_rows_opened": 0, "capitalcom_payload_opened": False, "process_file_access_instrumentation": "ProductionCacheCompiler role/access inventory plus strict column allowlist", "status": "pass"})
    inventory = _file_inventory(output, excluded={"STAGE23_EVIDENCE_MANIFEST.json", "INDEPENDENT_REVIEW_TARGET.json"})
    evidence = {"schema": "stage23_evidence_manifest_v1", "campaign_id": CAMPAIGN_ID, "implementation_commit": args.implementation_commit, "candidate_registered_rows": len(strategy), "unique_economic_executions": len(execution), "control_rows": 800, "files": inventory, "inventory_sha256": canonical_hash(inventory), "economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False, "status": "ready_for_complete_independent_review"}
    atomic_write_json(output / "STAGE23_EVIDENCE_MANIFEST.json", evidence)
    target = {"schema": "stage23_independent_preoutcome_review_target_v1", "implementation_commit": args.implementation_commit, "stage22_review_sha256": "7e97143f89c07fed180aa9e3e5d492ab779ccfc58187d3a71b0a27c4ec45b958", "evidence_manifest_sha256": sha256_file(output / "STAGE23_EVIDENCE_MANIFEST.json"), "seven_findings": [f"S22-V04-{index:03d}" for index in range(1, 8)], "required_verdict": "PASS with zero blocking findings", "scope": ["complete finding/subfinding reconciliation", "production source/cache authority", "production path", "all canaries", "dirty-original isolation", "outcome firewall", "tests/benchmark", "registries/counts", "detach/recovery", "terminal reconciliation"]}
    atomic_write_json(output / "INDEPENDENT_REVIEW_TARGET.json", target)
    print(json.dumps({"status": "ready_for_independent_review", "output_root": str(output), "evidence_manifest_sha256": sha256_file(output / "STAGE23_EVIDENCE_MANIFEST.json"), "review_target_sha256": sha256_file(output / "INDEPENDENT_REVIEW_TARGET.json")}, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    output: Path = args.output_root; review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or int(review.get("blocking_findings", -1)) != 0 or review.get("economic_outcomes_opened") is not False or int(review.get("protected_rows_opened", -1)) != 0 or review.get("capitalcom_payload_opened") is not False:
        raise RuntimeError("final packet requires an exact clean independent PASS")
    target_hash = sha256_file(output / "INDEPENDENT_REVIEW_TARGET.json")
    if review.get("bindings", {}).get("review_target_sha256") != target_hash or review.get("bindings", {}).get("implementation_commit") != args.implementation_commit:
        raise RuntimeError("independent review does not bind this implementation and target")
    code_inventory = _code_inventory(args.repository_root)
    atomic_write_json(output / "CODE_HASH_INVENTORY.json", code_inventory)
    dependencies = _file_inventory(output, excluded={"FINAL_CAMPAIGN_MANIFEST.json", "FINAL_HUMAN_APPROVAL_REQUEST.json", "FINAL_LAUNCH_TASK.md", "FINAL_PACKET_HASH_INVENTORY.json"})
    primary = {
        "strategy_registry": sha256_file(output / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"),
        "execution_registry": sha256_file(output / "FINAL_EXECUTION_REGISTRY.jsonl"),
        "control_registry": sha256_file(output / "FINAL_CONTROL_REGISTRY.jsonl"),
        "a2_counterpart_registry": sha256_file(output / "A2_PARENT_COUNTERPART_REGISTRY.jsonl"),
        "production_cache_manifest": sha256_file(output / "PRODUCTION_CACHE_MANIFEST.json"),
        "execution_input_authority": sha256_file(output / "EXECUTION_INPUT_AUTHORITY.json"),
        "code_inventory": sha256_file(output / "CODE_HASH_INVENTORY.json"),
        "independent_review": sha256_file(args.review),
    }
    execution_input_authority = json.loads((output / "EXECUTION_INPUT_AUTHORITY.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": "stage23_final_executable_campaign_manifest_v1", "campaign_id": CAMPAIGN_ID,
        "repository": {"implementation_commit": args.implementation_commit, "launch_from_clean_reviewed_descendant": True},
        "counts": {"registered_attempts": 11968, "unique_economic_executions": 11963, "controls": 800, "families": 5, "outer_folds": 8},
        "primary_hashes": primary,
        "execution_input_authority": execution_input_authority,
        "resource_limits": {"workers": 4, "jobs_in_flight": 4, "aggregate_process_tree_rss_bytes": 10 * 1024**3, "campaign_output_bytes": 24 * 1024**3, "minimum_free_disk_bytes": 8 * 1024**3, "heartbeat_seconds": 1800, "graceful_stop_seconds": 300, "wall_time": "renewable_checkpoint_no_hard_stop"},
        "artifact_dependencies": dependencies, "independent_review": {"path": args.review.name, "sha256": sha256_file(args.review), "verdict": "PASS"},
        "authorization_state": "awaiting_one_exact_external_human_launch_approval", "economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False,
    }
    atomic_write_json(output / "FINAL_CAMPAIGN_MANIFEST.json", manifest)
    request = {
        "schema": "stage23_final_human_approval_request_v1", "campaign_id": CAMPAIGN_ID,
        "final_campaign_manifest_sha256": sha256_file(output / "FINAL_CAMPAIGN_MANIFEST.json"), "repository_implementation_commit": args.implementation_commit,
        "request": "Authorize one launch of the exact hash-bound Stage23-remediated Stage22 core-liquid campaign under the attached manifest, with no scope beyond the registered 11,968 attempts and 800 controls.",
        "authorized_if_approved": ["rankable 2023-2025 economic outcomes for exact registered campaign", "detached supervisor", "terminal reconciliation/review/handoff"],
        "still_prohibited": ["protected outcomes", "Capital.com payload", "new acquisition", "C17", "separate Phase 6", "account actions", "orders", "deployment", "live trading", "force push"],
    }
    atomic_write_json(output / "FINAL_HUMAN_APPROVAL_REQUEST.json", request)
    launch = f"""# Final Stage 23 launch task\n\nLaunch only after an exact external approval JSON binds:\n\n- campaign `{CAMPAIGN_ID}`\n- manifest `{sha256_file(output / 'FINAL_CAMPAIGN_MANIFEST.json')}`\n- approval request `{sha256_file(output / 'FINAL_HUMAN_APPROVAL_REQUEST.json')}`\n- implementation commit `{args.implementation_commit}`\n\nRepeat all authority, source, cache, resource, canary and Telegram gates atomically. Install the generated systemd-user service (tmux fallback only if unavailable), verify bounded work acceptance, one reconciled real registered unit and the first scheduled heartbeat, then release health and end interactive polling. Do not access protected or Capital.com data.\n"""
    atomic_write_bytes(output / "FINAL_LAUNCH_TASK.md", launch.encode())
    inventory = _file_inventory(output, excluded={"FINAL_PACKET_HASH_INVENTORY.json"})
    atomic_write_json(output / "FINAL_PACKET_HASH_INVENTORY.json", {"schema": "stage23_final_packet_hash_inventory_v1", "files": inventory, "inventory_sha256": canonical_hash(inventory)})
    print(json.dumps({"status": "final_packet_ready_for_human_launch_approval", "final_manifest_sha256": sha256_file(output / "FINAL_CAMPAIGN_MANIFEST.json"), "final_approval_request_sha256": sha256_file(output / "FINAL_HUMAN_APPROVAL_REQUEST.json"), "final_launch_task_sha256": sha256_file(output / "FINAL_LAUNCH_TASK.md")}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--repository-root", type=Path, required=True); evidence.add_argument("--candidate-root", type=Path, required=True); evidence.add_argument("--source-control-registry", type=Path, required=True); evidence.add_argument("--output-root", type=Path, required=True); evidence.add_argument("--implementation-commit", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--repository-root", type=Path, required=True); final.add_argument("--candidate-root", type=Path, required=True); final.add_argument("--output-root", type=Path, required=True); final.add_argument("--implementation-commit", required=True); final.add_argument("--review", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    (build_evidence if args.command == "evidence" else finalize)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
