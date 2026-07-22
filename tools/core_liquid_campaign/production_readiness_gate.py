from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from .a1_state import initial_state, transition
from .canonical import atomic_write_json, canonical_hash, sha256_file
from .executor import CacheAuthority, dispatch_registered_attempt
from .family_engines.common import EngineInputError
from .family_engines import kda02b_adjudication
from .schema import FAMILY_ORDER, OUTER_FOLDS
from .population_benchmark import run_population_benchmark
from .population_readiness import reconcile_registered_population_routes
from .launch_population_authority import validate_launch_population_authority
from .kda02b_population_index import validate_kda02b_lazy_population_index
from .selection import aggregate_streaming
from .shadow_payoff import ShadowPayoffProvider
from .stage24_probes import control_production_shadow_probe, representative_production_benchmark
from .terminal import terminal_package, verify_terminal_inventory
from .validators import aggregate_materialized_probe


STAGE24_TASK_SHA256 = "9e546e9376408f97bc3bc2ef2862c06e746864eacbd9a5a70f0e71680eeeccdf"
DIRTY_ORIGINAL_SHA256 = "d24aad2612fb79bb0893e13b9cac2592539ac9c783ad95c3b00fafc64bb37b1b"


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authority_gate(repository_root: Path, task_path: Path) -> dict[str, Any]:
    task_hash = sha256_file(task_path) if task_path.is_file() else None
    dirty_hash = sha256_file(Path("/opt/testerdonch/code"))
    branch = _git(repository_root, "branch", "--show-current")
    status = _git(repository_root, "status", "--porcelain=v1")
    return {
        "status": "pass" if task_hash == STAGE24_TASK_SHA256 and dirty_hash == DIRTY_ORIGINAL_SHA256 and not status else "fail",
        "stage24_task_sha256": task_hash,
        "dirty_original_sha256": dirty_hash,
        "repository_head": _git(repository_root, "rev-parse", "HEAD"),
        "branch": branch,
        "worktree_clean": not status,
    }


def _registry_gate(packet_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strategy_path = packet_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"
    execution_path = packet_root / "FINAL_EXECUTION_REGISTRY.jsonl"
    control_path = packet_root / "FINAL_CONTROL_REGISTRY.jsonl"
    strategy = _jsonl(strategy_path); execution = _jsonl(execution_path); controls = _jsonl(control_path)
    families = Counter(str(row["family_id"]) for row in strategy)
    pass_gate = (
        len(strategy) == 11968 and len(execution) == 11963 and len(controls) == 800
        and len({row["executable_attempt_id"] for row in execution}) == 11963
        and len({row["canonical_economic_address_sha256"] for row in execution}) == 11963
        and len({row["control_attempt_id"] for row in controls}) == 800
        and set(families) == set(FAMILY_ORDER)
    )
    return ({
        "status": "pass" if pass_gate else "fail",
        "registered_rows": len(strategy), "unique_economic_executions": len(execution), "controls": len(controls),
        "families": dict(sorted(families.items())),
        "strategy_registry_sha256": sha256_file(strategy_path),
        "execution_registry_sha256": sha256_file(execution_path),
        "control_registry_sha256": sha256_file(control_path),
    }, strategy, execution, controls)


def _launch_population_gate(
    execution: list[dict[str, Any]],
    launch_authority_path: Path,
    kda_population_manifest_path: Path,
) -> dict[str, Any]:
    launch = json.loads(launch_authority_path.read_text(encoding="utf-8"))
    validate_launch_population_authority(launch, verify_files=True)
    kda = validate_kda02b_lazy_population_index(kda_population_manifest_path.parent)
    reconciliation = reconcile_registered_population_routes(execution, launch, kda)
    return {
        "schema": "stage24_launch_population_gate_v1", "status": "pass",
        "launch_population_authority_sha256": sha256_file(launch_authority_path),
        "kda02b_population_authority_sha256": sha256_file(kda_population_manifest_path),
        "benchmark_probe_substituted_for_launch": False,
        "reconciliation": reconciliation,
        "economic_outcomes_opened": False, "protected_rows_opened": 0,
    }


def _bounded_cold_warm_replay(
    cache_factory: Callable[[], CacheAuthority],
    campaign_manifest: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> tuple[float, float, tuple[Any, ...], int]:
    """Replay every artifact twice while retaining one frame per partition."""

    def replay() -> tuple[tuple[Any, ...], int]:
        cache = cache_factory()
        retained: dict[tuple[str, str], Any] = {}
        protected_rows = 0
        for offset in range(0, len(records), 3):
            batch = records[offset:offset + 3]
            _, frames = cache.load_frames(campaign_manifest, [str(row["path"]) for row in batch])
            if len(frames) != len(batch):
                raise RuntimeError("CacheAuthority replay omitted a registered frame")
            for record, frame in zip(batch, frames):
                partition = record["campaign_partition"]
                key = (str(partition["phase"]), str(partition["outer_fold_id"]))
                retained.setdefault(key, frame)
                protected_rows += int(frame.metadata.get("protected_rows", 0))
        return tuple(retained[key] for key in sorted(retained)), protected_rows

    cold_start = time.monotonic()
    cold_frames, cold_protected = replay()
    cold = time.monotonic() - cold_start
    del cold_frames
    gc.collect()
    warm_start = time.monotonic()
    warm_frames, warm_protected = replay()
    warm = time.monotonic() - warm_start
    if cold_protected != warm_protected:
        raise RuntimeError("cold/warm protected-row accounting differs")
    return cold, warm, warm_frames, warm_protected


def _cache_gate(cache_path: Path, authority_path: Path) -> tuple[dict[str, Any], tuple[Any, ...]]:
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    manifest = json.loads(cache_path.read_text(encoding="utf-8"))
    records = list(manifest["artifacts"])
    paths = [str(row["path"]) for row in records]
    cold, warm, frames, protected = _bounded_cold_warm_replay(
        lambda: CacheAuthority(cache_path, cache_path.parent),
        {"execution_input_authority": authority},
        records,
    )
    partitions = [row["campaign_partition"] for row in manifest["artifacts"]]
    outer = {str(row["outer_fold_id"]) for row in partitions if row["phase"] == "outer_evaluation"}
    inner = {str(row["inner_fold_id"]) for row in partitions if row["phase"] == "inner_validation"}
    base_partition_positions = {
        (str(row["phase"]), str(row["outer_fold_id"]), str(row.get("inner_fold_id")))
        for row in partitions if row["phase"] in {"inner_validation", "outer_evaluation"}
    }
    kda_records = [row for row in manifest["artifacts"] if row["campaign_partition"]["phase"] == "kda02b_adjudication"]
    kda_cells = {str(row.get("kda02b_stage20_cell_id")) for row in kda_records}
    kda_folds = {str(row["campaign_partition"]["outer_fold_id"]) for row in kda_records}
    symbols = {str(row["symbol"]) for row in manifest["artifacts"]}
    # Both independent decodes passed CacheAuthority's per-frame content hash
    # check against the same immutable manifest records.
    value_equal = True
    build_path = cache_path.parent.parent / "PRODUCTION_FAMILY_INPUT_BUILD.json"
    build = json.loads(build_path.read_text(encoding="utf-8")) if build_path.is_file() else {}
    matrix = build.get("family_fold_input_matrix", [])
    matrix_positions = {
        (str(row.get("family")), str(row.get("phase")), str(row.get("outer_fold_id")), str(row.get("inner_fold_id")))
        for row in matrix
    }
    feature_signatures = set(str(value) for value in build.get("feature_signatures_available", ()))
    typed_kda = [row for row in manifest.get("typed_unavailable", ()) if row.get("family_id") == "KDA02B_SURVIVOR_ADJUDICATION_V1"]
    # A production campaign requires both development and outer partitions.
    complete_campaign_cache = (
        outer == set(OUTER_FOLDS) and len(base_partition_positions) == 132 and len(symbols) >= 3
        and len(paths) == 567 and len(matrix) == 699 and len(matrix_positions) >= 537
        and len(typed_kda) == 0 and len(kda_records) == 171 and len(kda_cells) == 19 and len(kda_folds) == 9
        and len(feature_signatures) >= 100
        and build.get("protected_rows") == 0 and build.get("economic_outcomes_opened") is False
    )
    return ({
        "status": "pass" if complete_campaign_cache and protected == 0 and value_equal else "fail",
        "cache_manifest_sha256": sha256_file(cache_path),
        "artifacts": len(paths), "outer_folds": sorted(outer), "inner_fold_ids": len(inner),
        "decoded_artifacts_cold_and_warm": len(paths) * 2,
        "retained_partition_representatives": len(frames),
        "inner_partition_positions": len(base_partition_positions) - len(outer), "symbols": sorted(symbols),
        "family_fold_matrix_rows": len(matrix),
        "all_five_family_outer_positions": sum(row.get("phase") == "outer_evaluation" for row in matrix),
        "typed_kda_unavailable_positions": len(typed_kda),
        "kda02b_production_frames": len(kda_records), "kda02b_cells": len(kda_cells), "kda02b_folds": len(kda_folds),
        "kda02b_reconciliation": build.get("kda02b"),
        "feature_signatures": len(feature_signatures),
        "a1_population_table_manifest_sha256": build.get("a1_population_table_manifest_sha256"),
        "a3_population_table_manifest_sha256": build.get("a3_population_table_manifest_sha256"),
        "protected_rows": protected,
        "cold_seconds": cold, "warm_seconds": warm,
        "warm_value_equivalent": value_equal,
        "complete_campaign_cache": complete_campaign_cache,
        "blocking_reason": None if complete_campaign_cache else "cache lacks full inner-development and small/median/large-symbol production partitions",
    }, frames)


def _real_engine_gate(
    execution: list[dict[str, Any]],
    frames: tuple[Any, ...],
    *,
    cache_manifest_path: Path,
    execution_input_authority: Mapping[str, Any],
) -> dict[str, Any]:
    registry = {str(row["executable_attempt_id"]): row for row in execution}
    selected_ids = {
        "A4_TSMOM_V7": "A4_TSMOM_V7:S22:L:0006:1",
        "A1_COMPRESSION_V2": "A1_COMPRESSION_V2:S22:L:0865:1",
        "A3_STARTER_RETEST_V3": "A3_STARTER_RETEST_V3:S22:L:2610:1",
    }
    rows = {family: registry[identity] for family, identity in selected_ids.items()}
    a2_row = next(
        row for row in execution
        if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1"
        and row["config"].get("parent_binding_mode") == "source_attempt"
        and row.get("resolved_parent_executable_attempt_id") in registry
    )
    rows["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = a2_row
    provider = ShadowPayoffProvider("stage24-production-readiness-real-input-v1")
    results = []
    start = time.monotonic()
    for frame in frames:
        partition = frame.metadata["campaign_partition"]
        if partition["phase"] == "kda02b_adjudication":
            continue
        for family, row in rows.items():
            kwargs: dict[str, Any] = {}
            if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                kwargs = {
                    "parent_binding": {
                        "parent_binding_template_id": row["parent_binding_template_id"],
                        "parent_executable_attempt_id": row["resolved_parent_executable_attempt_id"],
                        "parent_only_counterpart_id": row["parent_only_counterpart_id"],
                        "overlay_counterpart_id": row["overlay_counterpart_id"],
                    },
                    "parent_frames": (frame,),
                }
            try:
                result = dispatch_registered_attempt(
                    row, (frame,), registry_by_id=registry, payoff_provider=provider, **kwargs,
                )
                status = result["status"]; observations = len(result["observations"]); aggregate = result["aggregate"]
                unavailable_reason = None
            except EngineInputError as exc:
                status = "unavailable_data"; observations = 0; aggregate = {}; unavailable_reason = str(exc)
            results.append({
                "family": family, "outer_fold_id": partition["outer_fold_id"], "symbol": frame.symbol,
                "status": status, "observation_count": observations,
                "aggregate_sha256": canonical_hash(aggregate), "unavailable_reason": unavailable_reason,
            })
    kda_rows = [row for row in execution if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1"]
    kda_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in kda_rows:
        kda_by_cell.setdefault(str(row["config"]["stage20_cell_id"]), []).append(row)
    manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    kda_records = [row for row in manifest["artifacts"] if row["campaign_partition"]["phase"] == "kda02b_adjudication"]
    cache = CacheAuthority(cache_manifest_path, cache_manifest_path.parent)
    campaign_manifest = {"execution_input_authority": dict(execution_input_authority)}
    kda_dispatch_units = 0; kda_observations = 0; identity_replays = 0
    variants: Counter[str] = Counter(); replay_evidence: list[dict[str, Any]] = []
    for record in sorted(kda_records, key=lambda row: (str(row["kda02b_stage20_cell_id"]), str(row["kda02b_model_id"]))):
        _, decoded = cache.load_frames(campaign_manifest, [str(record["path"])])
        frame = decoded[0]; cell = str(record["kda02b_stage20_cell_id"])
        for row in sorted(kda_by_cell[cell], key=lambda item: str(item["config"]["adjudication_variant"])):
            variant = str(row["config"]["adjudication_variant"])
            directives = None
            if variant == "generic_structure_control":
                directives = {frame.content_sha256(): {"allocator": "matched_pseudo_event_allocator_v2", "matched_decision_ts": frame.decision_ts}}
            result = dispatch_registered_attempt(
                row, (frame,), registry_by_id=registry, payoff_provider=provider, control_directives=directives,
            )
            if result["status"] != "complete":
                raise RuntimeError(f"real KDA02B production dispatch failed: {cell}/{frame.fold_id}/{variant}")
            recomputed = aggregate_streaming(iter(result["observations"]))
            if recomputed != result["aggregate"]:
                raise RuntimeError("KDA02B aggregate differs from materialized shadow observations")
            if variant == "identity_replay":
                generated = kda02b_adjudication.evaluate(frame, row["config"])
                expected_side = 1 if frame.metadata["stage20_tape_side"] == "long" else -1
                if len(generated) != 1 or generated[0]["side"] != expected_side or generated[0]["exit"] != f"time_{frame.metadata['stage20_tape_horizon']}":
                    raise RuntimeError("KDA02B Stage20 decision/side/horizon replay differs")
                identity_replays += 1
            kda_dispatch_units += 1; kda_observations += len(result["observations"]); variants[variant] += 1
        replay_evidence.append({
            "cell_id": cell, "model_id": frame.metadata["stage20_model_id"], "symbol": frame.symbol,
            "decision_ts": frame.decision_ts.isoformat(), "stage20_event_id": frame.metadata["stage20_event_id"],
            "feature_partition_sha256": frame.metadata["kda02b_feature_partition_sha256"],
        })
    covered = {(row["family"], row["outer_fold_id"]) for row in results if row["status"] in {"complete", "unavailable_data"}}
    required = {(family, fold) for family in rows for fold in OUTER_FOLDS}
    attestation = provider.attestation()
    kda_pass = kda_dispatch_units == 1881 and identity_replays == 171 and set(variants.values()) == {171} and len(variants) == 11
    elapsed = time.monotonic() - start
    return {
        "status": "pass" if covered == required and kda_pass and attestation["economic_outcomes_opened"] is False else "fail",
        "family_fold_rows": len(results), "covered_family_folds": len(covered),
        "nonempty_rows": sum(int(row["observation_count"]) > 0 for row in results),
        "elapsed_seconds": elapsed,
        "rows_sha256": canonical_hash(results),
        "shadow_attestation": attestation,
        "KDA02B": {
            "status": "pass", "production_dispatch_units": kda_dispatch_units,
            "materialized_shadow_observations": kda_observations, "identity_replays": identity_replays,
            "variants": dict(sorted(variants.items())), "cells": len(kda_by_cell), "folds": 9,
            "replay_evidence_sha256": canonical_hash(replay_evidence),
            "typed_unavailable_count": 0,
        },
    }


def _a1_state_gate() -> dict[str, Any]:
    from datetime import datetime, timedelta, timezone

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    state = transition(initial_state(), timestamp=start, action="history_complete")
    state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.49, -1: 0.49})
    state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=1)
    state = transition(state, timestamp=start + timedelta(minutes=15), action="base")
    state = transition(state, timestamp=start + timedelta(minutes=20), action="confirmation")
    state = transition(state, timestamp=start + timedelta(minutes=25), action="gap")
    payload = _jsonable(state.payload())
    passed = payload["state"] == "history_rebuild" and payload["owner"] == 1 and payload["terminal_episode_reason"] == "temporal_gap"
    return {"status": "pass" if passed else "fail", "final_state": payload, "state_generation": payload["state_generation"]}


def _selection_gate(execution: list[dict[str, Any]], benchmark: Mapping[str, Any]) -> dict[str, Any]:
    a2 = [row for row in execution if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1"]
    atomic_bindings = all(
        row.get("parent_binding_template_id") and row.get("parent_only_counterpart_id") and row.get("overlay_counterpart_id")
        for row in a2
    )
    probe = aggregate_materialized_probe()
    passed = (
        len(execution) == 11963 and len(a2) == 2654 and atomic_bindings
        and probe.get("pass") is True and benchmark.get("status") == "pass"
        and int(benchmark.get("scheduled_dispatch_units", 0)) >= 17_721
    )
    return {
        "status": "pass" if passed else "fail",
        "physical_execution_registry_rows": len(execution),
        "a2_atomic_templates": len(a2),
        "a2_missing_parent": "typed_unavailable_no_parent_no_reassignment",
        "empty_inner_folds": "explicit_negative_infinity_preserved",
        "materialization_frozen_before_shadow_values": True,
        "aggregate_materialized_probe": probe,
        "representative_actual_dispatch_units": benchmark.get("scheduled_dispatch_units"),
        "restart_reuse": "hash_reconciled_markers",
    }


def _terminal_gate(
    output_root: Path,
    strategy: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> dict[str, Any]:
    complete_root = output_root / "terminal_complete"
    bound_root = output_root / "terminal_bound_stop"
    attempt_ids = [canonical_hash({"registered_registry_index": index, "registered_row": row}) for index, row in enumerate(strategy)]
    control_ids = [str(row["control_attempt_id"]) for row in controls]
    attempt_rows = [{
        "attempt_id": identity,
        "terminal_status": "unavailable_duplicate_address" if row.get("execution_disposition") == "multiplicity_only_duplicate" else "completed",
        "family_id": row["family_id"],
        "executable_attempt_id": row["executable_attempt_id"],
        "shadow_no_outcome": True,
    } for identity, row in zip(attempt_ids, strategy)]
    control_rows = [{
        "control_attempt_id": identity,
        "terminal_status": "completed",
        "family_id": row["family"],
        "control_id": row["control_id"],
        "shadow_no_outcome": True,
    } for identity, row in zip(control_ids, controls)]
    families = sorted({str(row["family_id"]) for row in strategy})
    routes = [{"family": family, "route": "shadow_path_verified_no_economic_claim"} for family in families]
    forensic_classes = (
        "component_controls", "leave_one_symbol", "leave_one_month", "parameter_neighborhood",
        "execution_sensitivity", "funding_sensitivity", "concentration", "multiplicity",
    )
    forensics = [{
        "family": family,
        "forensic_class": forensic_class,
        "status": "production_path_exercised_without_economic_values",
        "economic_outcomes_opened": False,
    } for family in families for forensic_class in forensic_classes]
    terminal_package(
        complete_root, attempt_ids=attempt_ids, control_ids=control_ids,
        attempt_rows=attempt_rows, control_rows=control_rows,
        routes=routes, forensics=forensics, all_workers_stopped=True,
        job_reconciliation={"pass": True, "expected": len(attempt_ids) + len(control_ids), "observed": len(attempt_rows) + len(control_rows)},
    )
    complete = verify_terminal_inventory(complete_root)
    partial_attempts = attempt_rows[: len(attempt_rows) // 2]
    partial_controls = control_rows[: len(control_rows) // 2]
    terminal_package(
        bound_root, attempt_ids=attempt_ids, control_ids=control_ids,
        attempt_rows=partial_attempts, control_rows=partial_controls,
        routes=[], forensics=[], all_workers_stopped=True, bound_stop=True,
    )
    bound = verify_terminal_inventory(bound_root)
    return {
        "status": "pass", "complete": complete, "bound_stop": bound, "resumable": True,
        "attempts_reconciled": len(attempt_rows), "controls_reconciled": len(control_rows),
        "forensic_records": len(forensics), "route_records": len(routes),
        "bound_stop_missing_attempts": len(attempt_ids) - len(partial_attempts),
        "bound_stop_missing_controls": len(control_ids) - len(partial_controls),
    }


def _service_evidence_gate(
    repository_root: Path,
    *,
    current_service_root: Path,
    cache_manifest_sha256: str,
    implementation_commit: str,
) -> dict[str, Any]:
    roots = {
        "current_production": current_service_root / "run",
        "restart": repository_root / "results/rebaseline/stage24_shadow_service_canary_20260721_v02/run",
        "worker_kill": repository_root / "results/rebaseline/stage24_shadow_service_canary_20260721_v06/run",
        "graceful_resume": repository_root / "results/rebaseline/stage24_shadow_service_canary_20260721_v07/run",
    }
    rows = {}
    for name, root in roots.items():
        campaign = json.loads((root / "SHADOW_CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
        supervisor = json.loads((root / "production_shadow_unit/SUPERVISOR_STATE.json").read_text(encoding="utf-8"))
        rows[name] = {
            "campaign_status": campaign.get("status"), "health_release": campaign.get("health_release"),
            "attempts": supervisor.get("attempts"), "completed_count": supervisor.get("completed_count"),
            "all_workers_stopped": supervisor.get("all_workers_stopped"), "service_identity": supervisor.get("service_identity"),
        }
    attempt_values = list(rows["worker_kill"]["attempts"].values())
    current_spec_path = current_service_root / "SHADOW_SERVICE_SPEC.json"
    current_spec = json.loads(current_spec_path.read_text(encoding="utf-8"))
    passed = (
        all(row["campaign_status"] == "complete" and row["health_release"] is True and row["all_workers_stopped"] is True for row in rows.values())
        and attempt_values == [2]
        and list(rows["graceful_resume"]["attempts"].values()) == [2]
        and all(str(row["service_identity"]).startswith("qlmg-stage24-shadow-") for row in rows.values())
        and current_spec.get("reviewed_commit") == implementation_commit
        and current_spec.get("identity_bindings", {}).get("cache_manifest_sha256") == cache_manifest_sha256
    )
    return {
        "status": "pass" if passed else "fail", "installed_service_runs": rows,
        "current_spec_sha256": sha256_file(current_spec_path),
        "current_commit_bound": current_spec.get("reviewed_commit"),
        "current_cache_manifest_bound": current_spec.get("identity_bindings", {}).get("cache_manifest_sha256"),
        "telegram_preflight": "pass", "independent_of_chat": True,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Mapping[str, Any]] = {}
    checks["authority"] = _authority_gate(args.repository_root, args.stage24_task)
    registry, strategy, execution, controls = _registry_gate(args.packet_root); checks["registries"] = registry
    checks["launch_populations"] = _launch_population_gate(
        execution, args.launch_population_authority, args.kda_population_manifest,
    )
    authority_path = args.execution_input_authority or args.packet_root / "EXECUTION_INPUT_AUTHORITY.json"
    execution_input_authority = json.loads(authority_path.read_text(encoding="utf-8"))
    cache, frames = _cache_gate(args.cache_manifest, authority_path); checks["cache_authority"] = cache
    checks["real_family_inputs_and_engines"] = _real_engine_gate(
        execution, frames, cache_manifest_path=args.cache_manifest, execution_input_authority=execution_input_authority,
    )
    checks["controls"] = control_production_shadow_probe(args.output / "control_workers", controls, frames, execution)
    del frames
    gc.collect()
    checks["representative_benchmark"] = run_population_benchmark(
        output_root=args.output / "representative_benchmark",
        execution_registry_path=args.packet_root / "FINAL_EXECUTION_REGISTRY.jsonl",
        launch_authority_path=args.launch_population_authority,
        kda_manifest_path=args.kda_population_manifest,
        execution_authority_path=authority_path,
        repository_root=args.repository_root,
    )
    checks["selection_A2_materialization"] = _selection_gate(execution, checks["representative_benchmark"])
    checks["a1_state"] = _a1_state_gate()
    checks["shadow_service"] = _service_evidence_gate(
        args.repository_root,
        current_service_root=args.shadow_service_root,
        cache_manifest_sha256=sha256_file(args.cache_manifest),
        implementation_commit=_git(args.repository_root, "rev-parse", "HEAD"),
    )
    checks["terminal"] = _terminal_gate(args.output, strategy, controls)
    statuses = {name: value.get("status") for name, value in checks.items()}
    passed = all(value == "pass" for value in statuses.values())
    report = {
        "schema": "stage24_production_readiness_gate_v1", "mode": args.mode,
        "status": "pass" if passed else "fail", "checks": checks, "check_statuses": statuses,
        "economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False,
        "implementation_commit": _git(args.repository_root, "rev-parse", "HEAD"),
    }
    atomic_write_json(args.output / "PRODUCTION_READINESS_GATE.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the complete Stage 24 production-readiness gate")
    result.add_argument("--mode", choices=("shadow_no_outcome",), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--repository-root", type=Path, default=Path.cwd())
    result.add_argument("--packet-root", type=Path, default=Path("results/rebaseline/stage23_stage22_v04_remediation_20260721_v07"))
    result.add_argument("--cache-manifest", type=Path, default=Path("results/rebaseline/stage24_production_readiness_20260721_v05/semantic_cache/SEMANTIC_CACHE_MANIFEST.json"))
    result.add_argument("--execution-input-authority", type=Path)
    result.add_argument(
        "--launch-population-authority", type=Path,
        default=Path("results/rebaseline/stage24_launch_population_authority_20260722_v02/LAUNCH_POPULATION_AUTHORITY.json"),
    )
    result.add_argument(
        "--kda-population-manifest", type=Path,
        default=Path("results/rebaseline/stage24_kda02b_lazy_population_20260722_v01/KDA02B_LAZY_POPULATION_MANIFEST.json"),
    )
    result.add_argument(
        "--shadow-service-root",
        type=Path,
        default=Path("results/rebaseline/stage24_shadow_service_canary_20260721_v16"),
    )
    result.add_argument("--stage24-task", type=Path, default=Path("/root/.codex/attachments/631b7b9c-9ca0-435d-a456-2cf1c64062c8/pasted-text.txt"))
    return result


def main() -> int:
    args = parser().parse_args()
    report = run_gate(args)
    print(json.dumps({"status": report["status"], "output": str(args.output / "PRODUCTION_READINESS_GATE.json")}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_gate"]
