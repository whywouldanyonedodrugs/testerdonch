from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_hash, sha256_file
from .controls import CONTROL_IDS, effective_seed, execute_control, reconcile_control_duplicates
from .runtime import LazySupervisor, ResourceLimits
from .schema import CAMPAIGN_ID, baseline_config, economic_address, normalize_config
from .shadow_payoff import ShadowPayoffProvider
from .executor import CacheAuthority, dispatch_registered_attempt, validate_registered_attempt
from .family_engines.common import EngineInputError
from .selection import aggregate_streaming
from .synthetic import a1_frame, a3_frame, a4_frame


UTC = timezone.utc


def _attempt(family: str, attempt_id: str) -> dict[str, Any]:
    config = normalize_config(family, baseline_config(family))
    return {
        "campaign_id": CAMPAIGN_ID,
        "family_id": family,
        "config": config,
        "execution_disposition": "execute_once",
        "executable_attempt_id": attempt_id,
        "canonical_economic_address_sha256": economic_address(family, config)[1],
        "duplicate_of_executable_attempt_id": None,
    }


def _fixtures() -> dict[str, dict[str, Any]]:
    anchors = (datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 15, tzinfo=UTC))
    a4 = _attempt("A4_TSMOM_V7", "parent-a4")
    a1 = _attempt("A1_COMPRESSION_V2", "parent-a1")
    a3 = _attempt("A3_STARTER_RETEST_V3", "parent-a3")
    result = {
        "A4_TSMOM_V7": {
            "parent": a4,
            "frames": (
                a4_frame(a4["config"], signal_sign=1, anchor=anchors[0]),
                a4_frame(a4["config"], signal_sign=-1, anchor=anchors[1]),
            ),
            "registry": {a4["executable_attempt_id"]: a4},
            "parent_binding": None,
        },
        "A1_COMPRESSION_V2": {
            "parent": a1,
            "frames": (a1_frame(a1["config"], anchor=anchors[0]), a1_frame(a1["config"], anchor=anchors[1])),
            "registry": {a1["executable_attempt_id"]: a1},
            "parent_binding": None,
        },
        "A3_STARTER_RETEST_V3": {
            "parent": a3,
            "frames": (a3_frame(a3["config"], anchor=anchors[0]), a3_frame(a3["config"], anchor=anchors[1])),
            "registry": {a3["executable_attempt_id"]: a3},
            "parent_binding": None,
        },
    }
    a2_parent = _attempt("A1_COMPRESSION_V2", "stage24-a2-parent")
    a2_config = normalize_config("A2_PRIOR_HIGH_RS_CONTEXT_V1", baseline_config("A2_PRIOR_HIGH_RS_CONTEXT_V1"))
    template = canonical_hash({"mode": "beam_slot", "parent_slot": "A1_COMPRESSION_V2:2024Q1:beam:01"})
    a2 = {
        "campaign_id": CAMPAIGN_ID,
        "family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1",
        "config": a2_config,
        "execution_disposition": "execute_if_parent_available",
        "executable_attempt_id": "stage24-parent-a2",
        "canonical_economic_address_sha256": economic_address("A2_PRIOR_HIGH_RS_CONTEXT_V1", a2_config)[1],
        "duplicate_of_executable_attempt_id": None,
        "parent_binding_template_id": template,
        "parent_only_counterpart_id": "stage24-parent-only",
        "overlay_counterpart_id": "stage24-overlay",
    }
    first = a1_frame(a2_parent["config"], anchor=anchors[0])
    second = a1_frame(a2_parent["config"], anchor=anchors[1])
    by_lookback = {key: dict(value) for key, value in second.context.cross_section_returns_by_lookback.items()}
    by_lookback[20][second.symbol] = -0.08
    second = replace(
        second,
        context=replace(
            second.context,
            cross_section_returns=dict(by_lookback[20]),
            cross_section_returns_by_lookback=by_lookback,
            source_sha256=canonical_hash({"stage24_control_fixture": "second_context"}),
        ),
    )
    result["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = {
        "parent": a2,
        "frames": (first, second),
        "registry": {a2["executable_attempt_id"]: a2, a2_parent["executable_attempt_id"]: a2_parent},
        "parent_binding": {
            "parent_binding_template_id": template,
            "parent_executable_attempt_id": a2_parent["executable_attempt_id"],
            "parent_only_counterpart_id": a2["parent_only_counterpart_id"],
            "overlay_counterpart_id": a2["overlay_counterpart_id"],
        },
    }
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _result_record(control_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "control_id": control_id,
        "status": result.get("status"),
        "event_ids": sorted(item.event_id for item in result.get("observations", ())),
        "ledger": result.get("ledger", ()),
        "aggregate": result.get("aggregate", {}),
        "allocation_unavailable": result.get("allocation_unavailable", ()),
    }
    payload = _jsonable(record)
    return {
        "control_id": control_id,
        "status": result.get("status"),
        "observation_count": len(result.get("observations", ())),
        "transformed_result_sha256": canonical_hash(payload),
    }


def execute_control_fixture(
    family: str,
    control_id: str,
    *,
    reverse: bool = False,
    registered_control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fixture = _fixtures()[family]
    frames = list(fixture["frames"])
    if reverse:
        frames.reverse()
    seed = 3 if control_id in {
        "A4_SIGN_PERMUTED_MAIN_NULL",
        "A2_CONTEXT_PERMUTED_MAIN_NULL",
        "A3_RETEST_TIME_PERMUTED_MAIN_NULL",
    } else 1
    if registered_control is None:
        control = {
            "family": family,
            "control_id": control_id,
            "effective_seed": seed,
            "economic_address_sha256": canonical_hash({"stage24_control": control_id}),
            "control_attempt_id": f"stage24-{control_id}",
            "execution_status": "execute_once",
        }
    else:
        if registered_control.get("family") != family or registered_control.get("control_id") != control_id:
            raise RuntimeError("registered control fixture identity differs")
        identity = {
            "campaign_id": CAMPAIGN_ID,
            "family": registered_control["family"],
            "fold": registered_control["fold"],
            "parent_slot": registered_control["parent_slot"],
            "control_id": registered_control["control_id"],
            "replicate_index": registered_control["replicate_index"],
            "transformation_allocator_version": registered_control["transformation_allocator_version"],
        }
        if int(registered_control["effective_seed"]) != effective_seed(identity):
            raise RuntimeError("registered control fixture seed differs")
        control = {**dict(registered_control), "execution_status": "execute_once"}
    kwargs: dict[str, Any] = {
        "registry_by_id": fixture["registry"],
        "payoff_provider": ShadowPayoffProvider("stage24-all-controls"),
    }
    if fixture["parent_binding"] is not None:
        kwargs.update({"parent_binding": fixture["parent_binding"], "parent_frames": frames})
    result = execute_control(control, fixture["parent"], frames, **kwargs)
    return _result_record(control_id, result)


def _read_supervisor_results(root: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for marker_path in sorted((root / "markers").glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        artifact = root / marker["artifact"]
        if sha256_file(artifact) != marker["artifact_sha256"]:
            raise RuntimeError("control-probe supervisor artifact hash mismatch")
        payload = json.loads(artifact.read_text(encoding="utf-8"))["result"]
        result[str(payload["control_id"])] = payload
    return result


def control_production_shadow_probe(
    output_root: Path,
    registered_controls: Sequence[Mapping[str, Any]] | None = None,
    production_frames: Sequence[Any] | None = None,
    execution_registry: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    identities = [(family, control_id) for family, ids in CONTROL_IDS.items() for control_id in ids]
    physical_by_class: dict[tuple[str, str], Mapping[str, Any]] = {}
    physical_frame_inventory = []
    if production_frames is not None:
        for frame in production_frames:
            if frame.metadata.get("production_input") is not True or frame.metadata.get("protected_rows") != 0 or not isinstance(frame.metadata.get("source_authority"), Mapping):
                raise RuntimeError("control probe received a non-production or unbound CacheAuthority frame")
            physical_frame_inventory.append(frame.content_sha256())
        if not physical_frame_inventory:
            raise RuntimeError("control probe received no physical CacheAuthority frames")
    if registered_controls is not None:
        if len(registered_controls) != 800 or len({str(row["control_attempt_id"]) for row in registered_controls}) != 800:
            raise RuntimeError("physical control registry is not the frozen 800-row multiplicity")
        candidates_by_class: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in sorted(registered_controls, key=lambda item: str(item["control_attempt_id"])):
            candidates_by_class.setdefault((str(row["family"]), str(row["control_id"])), []).append(row)
        if set(candidates_by_class) != set(identities):
            raise RuntimeError("physical control registry does not cover all 20 control classes")
        for family, control_id in identities:
            for candidate in candidates_by_class[(family, control_id)]:
                probe = execute_control_fixture(family, control_id, registered_control=candidate)
                if probe["status"] == "complete" and int(probe["observation_count"]) > 0:
                    physical_by_class[(family, control_id)] = candidate
                    break
            if (family, control_id) not in physical_by_class:
                raise RuntimeError(f"no physical control address produced the required nonempty fixture: {control_id}")

    actual_production_dispatch: dict[str, dict[str, Any]] = {}
    if production_frames is not None and registered_controls is not None and execution_registry is not None:
        registry = {str(row["executable_attempt_id"]): row for row in execution_registry}
        parent_ids = {
            "A4_TSMOM_V7": "A4_TSMOM_V7:S22:L:0006:1",
            "A1_COMPRESSION_V2": "A1_COMPRESSION_V2:S22:L:0865:1",
            "A3_STARTER_RETEST_V3": "A3_STARTER_RETEST_V3:S22:L:2610:1",
        }
        parents = {family: registry[identity] for family, identity in parent_ids.items()}
        a2 = next(
            row for row in execution_registry
            if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1"
            and row["config"].get("parent_binding_mode") == "source_attempt"
            and row.get("resolved_parent_executable_attempt_id") in registry
        )
        parents["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = a2
        available_frames = tuple(
            frame for frame in production_frames
            if frame.metadata.get("campaign_partition", {}).get("phase") != "kda02b_adjudication"
        )
        if not available_frames:
            raise RuntimeError("actual control production dispatch lacks non-KDA CacheAuthority frames")
        # One exact compatible frame per family keeps aggregate denominators
        # identical. Stochastic controls that require a wider allocation
        # population remain locally unavailable here; their nonempty semantics
        # are exercised by the production-shaped registered fixtures above.
        frame_by_family: dict[str, Any] = {}
        for family, parent in parents.items():
            for candidate_frame in available_frames:
                kwargs: dict[str, Any] = {}
                if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                    kwargs = {
                        "parent_binding": {
                            "parent_binding_template_id": parent["parent_binding_template_id"],
                            "parent_executable_attempt_id": parent["resolved_parent_executable_attempt_id"],
                            "parent_only_counterpart_id": parent["parent_only_counterpart_id"],
                            "overlay_counterpart_id": parent["overlay_counterpart_id"],
                        },
                        "parent_frames": (candidate_frame,),
                    }
                try:
                    probe = dispatch_registered_attempt(
                        parent, (candidate_frame,), registry_by_id=registry,
                        payoff_provider=ShadowPayoffProvider("stage24-physical-control-frame-selection-v1"), **kwargs,
                    )
                except EngineInputError:
                    continue
                if probe.get("status") == "complete":
                    frame_by_family[family] = candidate_frame
                    break
            if family not in frame_by_family:
                raise RuntimeError(f"no compatible physical CacheAuthority control frame: {family}")
        for family, control_id in identities:
            frames = (frame_by_family[family],)
            target_fold = str(frames[0].metadata["campaign_partition"]["outer_fold_id"])
            candidate = next(
                row for row in registered_controls
                if row["family"] == family and row["control_id"] == control_id and row["fold"] == target_fold
            )
            control = {**dict(candidate), "execution_status": "execute_once"}
            parent = parents[family]
            kwargs: dict[str, Any] = {}
            if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                kwargs = {
                    "parent_binding": {
                        "parent_binding_template_id": parent["parent_binding_template_id"],
                        "parent_executable_attempt_id": parent["resolved_parent_executable_attempt_id"],
                        "parent_only_counterpart_id": parent["parent_only_counterpart_id"],
                        "overlay_counterpart_id": parent["overlay_counterpart_id"],
                    },
                    "parent_frames": frames,
                }
            result = execute_control(
                control, parent, frames, registry_by_id=registry,
                payoff_provider=ShadowPayoffProvider("stage24-physical-control-production-path-v1"), **kwargs,
            )
            if result.get("status") not in {"complete", "unavailable_duplicate_address"}:
                raise RuntimeError(f"physical CacheAuthority control dispatch failed: {family}/{control_id}")
            actual_production_dispatch[control_id] = _result_record(control_id, result)
        if set(actual_production_dispatch) != {control_id for _, control_id in identities}:
            raise RuntimeError("actual CacheAuthority control dispatch omitted a control class")
    def bound(family: str, control_id: str) -> Mapping[str, Any] | None:
        return physical_by_class.get((family, control_id))
    sequential = {control_id: execute_control_fixture(family, control_id, registered_control=bound(family, control_id)) for family, control_id in identities}
    reverse = {control_id: execute_control_fixture(family, control_id, reverse=True, registered_control=bound(family, control_id)) for family, control_id in identities}
    if any(row["status"] != "complete" or int(row["observation_count"]) <= 0 for row in sequential.values()):
        raise RuntimeError("one or more control classes lacks a nonempty applicable fixture")
    if sequential != reverse:
        raise RuntimeError("control transformed result differs under reversed input order")

    worker_hashes: dict[str, str] = {}
    for workers in (1, min(4, __import__("os").cpu_count() or 1)):
        root = output_root / f"workers-{workers}"
        jobs = [
            (control_id, lambda family=family, control_id=control_id: execute_control_fixture(family, control_id, registered_control=bound(family, control_id)))
            for family, control_id in identities
        ]
        limits = ResourceLimits(
            max_workers=workers,
            max_jobs_in_flight=workers,
            max_output_bytes=1024**3,
            minimum_free_disk_bytes=1,
            minimum_free_disk_fraction=0.0,
            heartbeat_seconds=1800,
        )
        first = LazySupervisor(root, limits).run(iter(jobs))
        replay = LazySupervisor(root, limits).run(iter(reversed(jobs)))
        if first["status"] != "complete" or replay["status"] != "complete":
            raise RuntimeError("control worker/restart probe did not complete")
        records = {}
        for marker_path in sorted((root / "markers").glob("*.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            artifact = root / marker["artifact"]
            if sha256_file(artifact) != marker["artifact_sha256"]:
                raise RuntimeError("benchmark supervisor artifact hash mismatch")
            records[str(marker["job_id"])] = json.loads(artifact.read_text(encoding="utf-8"))["result"]
        if records != sequential:
            raise RuntimeError("control worker execution differs from sequential replay")
        worker_hashes[str(workers)] = canonical_hash(records)
    if len(set(worker_hashes.values())) != 1:
        raise RuntimeError("control transformed hashes differ by worker count")
    chunk_orders = {
        str(size): canonical_hash({
            control_id: execute_control_fixture(family, control_id, registered_control=bound(family, control_id))
            for offset in range(0, len(identities), size)
            for family, control_id in identities[offset:offset + size]
        })
        for size in (1, 3, 7, 20)
    }
    if len(set(chunk_orders.values())) != 1:
        raise RuntimeError("control transformed hashes differ by chunk size")

    fixture = _fixtures()["A4_TSMOM_V7"]
    singleton_control = {
        "family": "A4_TSMOM_V7", "control_id": "A4_SIGN_PERMUTED_MAIN_NULL", "effective_seed": 3,
        "economic_address_sha256": canonical_hash({"stage24_control": "singleton"}),
        "control_attempt_id": "stage24-singleton", "execution_status": "execute_once",
    }
    singleton_result = execute_control(
        singleton_control,
        fixture["parent"],
        fixture["frames"][:1],
        registry_by_id=fixture["registry"],
        payoff_provider=ShadowPayoffProvider("stage24-all-controls"),
    )
    missing = reconcile_control_duplicates(
        [{
            "control_attempt_id": "missing", "parent_slot": "missing", "family": "A4_TSMOM_V7",
            "fold": "2025Q2", "control_id": "A4_CONTEXT_REMOVED", "replicate_index": 0,
            "effective_seed": 0, "transformation_allocator_version": "stage22_exact_control_dispatch_v2",
        }],
        {},
    )[0]
    duplicate_fields = {
        "campaign_id": CAMPAIGN_ID, "family": "A4_TSMOM_V7", "fold": "2025Q2", "parent_slot": "slot",
        "control_id": "A4_VOL_SCALING_REMOVED", "replicate_index": 0,
        "transformation_allocator_version": "stage22_exact_control_dispatch_v2",
    }
    duplicate = reconcile_control_duplicates(
        [
            {**duplicate_fields, "control_attempt_id": "duplicate-1", "effective_seed": effective_seed(duplicate_fields)},
            {**duplicate_fields, "control_attempt_id": "duplicate-2", "effective_seed": effective_seed(duplicate_fields)},
        ],
        {"slot": fixture["parent"]},
    )
    return {
        "schema": "stage24_control_production_shadow_probe_v1",
        "status": "pass",
        "control_classes_dispatched": len(sequential),
        "nonempty_applicable_transformed_fixtures": sum(int(row["observation_count"]) > 0 for row in sequential.values()),
        "sequential_transformed_inventory_sha256": canonical_hash(sequential),
        "worker_transformed_inventory_sha256": worker_hashes,
        "chunk_size_transformed_inventory_sha256": chunk_orders,
        "physical_control_registry_rows": 0 if registered_controls is None else len(registered_controls),
        "physical_control_registry_bound": registered_controls is not None,
        "physical_cache_authority_frames": len(physical_frame_inventory),
        "physical_cache_authority_frame_inventory_sha256": canonical_hash(sorted(physical_frame_inventory)),
        "physical_cache_authority_bound": production_frames is not None,
        "actual_cacheauthority_control_classes_dispatched": len(actual_production_dispatch),
        "actual_cacheauthority_control_inventory_sha256": canonical_hash(actual_production_dispatch),
        "reversed_input_order": "pass",
        "restart_reuse": "pass",
        "singleton_case": singleton_result["status"],
        "missing_parent_case": missing["execution_status"],
        "duplicate_case_exercised": any(row["execution_status"] == "unavailable_duplicate_address" for row in duplicate),
        "economic_outcomes_opened": False,
    }


def representative_production_benchmark(
    output_root: Path,
    execution: Sequence[Mapping[str, Any]],
    *,
    cache_manifest_path: Path,
    execution_input_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a deterministic, outcome-firewalled production dispatcher sample."""
    registry = {str(row["executable_attempt_id"]): row for row in execution}
    compile_started = time.monotonic()
    for row in execution:
        validate_registered_attempt(row)
    compile_seconds = time.monotonic() - compile_started
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    outer_paths: dict[str, str] = {}
    for record in sorted(
        cache_manifest["artifacts"],
        key=lambda item: (item["campaign_partition"]["outer_fold_id"], item["symbol"], item["path"]),
    ):
        partition = record["campaign_partition"]
        if partition["phase"] == "outer_evaluation" and partition["outer_fold_id"] not in outer_paths:
            outer_paths[str(partition["outer_fold_id"])] = str(record["path"])
    if set(outer_paths) != {
        "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4",
    }:
        raise RuntimeError("representative benchmark lacks one cache artifact per outer fold")
    kda_paths = {
        (str(record["kda02b_stage20_cell_id"]), str(record["campaign_partition"]["outer_fold_id"])): str(record["path"])
        for record in cache_manifest["artifacts"]
        if record["campaign_partition"]["phase"] == "kda02b_adjudication"
    }
    if len(kda_paths) != 171:
        raise RuntimeError("representative benchmark lacks the exact 19-cell by nine-fold KDA02B cache")
    campaign_manifest = {"execution_input_authority": dict(execution_input_authority)}
    cache = CacheAuthority(cache_manifest_path, cache_manifest_path.parent)
    cache.preload_frames(campaign_manifest)
    if cache._decoded_frames:
        raise RuntimeError("benchmark parent retained decoded frames before worker fork")
    cache_manifest_sha256 = sha256_file(cache_manifest_path)

    def stratified(rows: Sequence[Mapping[str, Any]], count: int = 30) -> list[Mapping[str, Any]]:
        ordered = sorted(rows, key=lambda row: str(row["canonical_economic_address_sha256"]))
        if len(ordered) <= count:
            return ordered
        return [ordered[min(len(ordered) - 1, index * len(ordered) // count)] for index in range(count)]

    family_rows: dict[str, list[Mapping[str, Any]]] = {}
    for family in ("A4_TSMOM_V7", "A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"):
        family_rows[family] = stratified([row for row in execution if row["family_id"] == family])
    family_rows["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = stratified([
        row for row in execution
        if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1"
        and row["config"].get("parent_binding_mode") == "source_attempt"
        and row.get("resolved_parent_executable_attempt_id") in registry
    ])
    family_rows["KDA02B_SURVIVOR_ADJUDICATION_V1"] = stratified([
        row for row in execution if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1"
    ])
    if any(len(rows) < 30 for rows in family_rows.values()):
        raise RuntimeError("representative benchmark has fewer than thirty addresses in a family stratum")
    job_specs = [
        (family, fold, row, outer_paths[fold])
        for family, rows in family_rows.items() if family != "KDA02B_SURVIVOR_ADJUDICATION_V1"
        for fold in sorted(outer_paths)
        for row in rows
    ]
    job_specs.extend(
        ("KDA02B_SURVIVOR_ADJUDICATION_V1", fold, row, kda_paths[(str(row["config"]["stage20_cell_id"]), fold)])
        for row in family_rows["KDA02B_SURVIVOR_ADJUDICATION_V1"]
        for fold in sorted({item[1] for item in kda_paths})
    )

    def task(family: str, fold: str, row: Mapping[str, Any], artifact_path: str):
        def run() -> dict[str, Any]:
            provider = ShadowPayoffProvider("stage24-representative-production-benchmark-v1")
            _, frames = cache.load_frames(campaign_manifest, [artifact_path])
            frame = frames[0]
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
            if family == "KDA02B_SURVIVOR_ADJUDICATION_V1" and row["config"]["adjudication_variant"] == "generic_structure_control":
                kwargs["control_directives"] = {
                    frame.content_sha256(): {
                        "allocator": "matched_pseudo_event_allocator_v2",
                        "matched_decision_ts": frame.decision_ts,
                    }
                }
            try:
                result = dispatch_registered_attempt(row, (frame,), registry_by_id=registry, payoff_provider=provider, **kwargs)
                recomputed = aggregate_streaming(iter(result.get("observations", ())))
                if recomputed != result.get("aggregate"):
                    raise RuntimeError("representative aggregate differs from its materialized observations")
                status = str(result["status"])
                digest = canonical_hash({
                    "status": status,
                    "aggregate": result.get("aggregate", {}),
                    "events": sorted(item.event_id for item in result.get("observations", ())),
                })
                observations = len(result.get("observations", ()))
            except EngineInputError as exc:
                if family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
                    raise RuntimeError("KDA02B representative real shadow dispatch failed") from exc
                status = "unavailable_data"; observations = 0
                digest = canonical_hash({"status": status, "reason": str(exc)})
            return {
                "status": status,
                "registered_attempt_id": row["executable_attempt_id"],
                "family": family,
                "outer_fold_id": fold,
                "observation_count": observations,
                "result_sha256": digest,
                "cache_manifest_sha256": cache_manifest_sha256,
                "cache_artifact_path": artifact_path,
                "shadow_attestation": provider.attestation(),
            }
        return run

    limits = ResourceLimits(
        max_workers=min(4, os.cpu_count() or 1), max_jobs_in_flight=min(4, os.cpu_count() or 1),
        max_rss_bytes=10 * 1024**3, max_output_bytes=4 * 1024**3,
        minimum_free_disk_bytes=8 * 1024**3, heartbeat_seconds=1800,
    )
    full_root = output_root / "stratified-full"
    identity_bindings = {
        "cache_manifest_sha256": cache_manifest_sha256,
        "execution_registry_sha256": canonical_hash(list(execution)),
        "benchmark_provider": "stage24-representative-production-benchmark-v1",
    }
    started = time.monotonic()
    full = LazySupervisor(full_root, limits, heartbeat=lambda _payload: True, identity_bindings=identity_bindings).run(iter(
        (f"benchmark:{family}:{fold}:{row['executable_attempt_id']}", task(family, fold, row, artifact_path))
        for family, fold, row, artifact_path in job_specs
    ))
    full_seconds = time.monotonic() - started
    if full["status"] != "complete" or int(full["completed_count"]) != len(job_specs):
        raise RuntimeError("representative production benchmark did not complete")

    scaling_specs = job_specs[::max(1, len(job_specs) // 40)][:40]
    scaling: dict[str, Any] = {}
    inventories: dict[str, str] = {}
    for workers in range(1, min(4, os.cpu_count() or 1) + 1):
        root = output_root / f"scaling-{workers}"
        worker_limits = ResourceLimits(
            max_workers=workers, max_jobs_in_flight=workers, max_rss_bytes=10 * 1024**3,
            max_output_bytes=1024**3, minimum_free_disk_bytes=8 * 1024**3, heartbeat_seconds=1800,
        )
        start = time.monotonic()
        state = LazySupervisor(root, worker_limits, heartbeat=lambda _payload: True, identity_bindings=identity_bindings).run(iter(
            (f"scale:{family}:{fold}:{row['executable_attempt_id']}", task(family, fold, row, artifact_path))
            for family, fold, row, artifact_path in scaling_specs
        ))
        elapsed = time.monotonic() - start
        records = {}
        for marker_path in sorted((root / "markers").glob("*.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            artifact = root / marker["artifact"]
            if sha256_file(artifact) != marker["artifact_sha256"]:
                raise RuntimeError("benchmark supervisor artifact hash mismatch")
            records[str(marker["job_id"])] = json.loads(artifact.read_text(encoding="utf-8"))["result"]
        inventories[str(workers)] = canonical_hash(records)
        scaling[str(workers)] = {
            "seconds": elapsed, "completed": state["completed_count"],
            "peak_process_tree_rss_bytes": state.get("peak_process_tree_rss_bytes"),
            "peak_output_bytes": state.get("peak_output_bytes"),
        }
    if len(set(inventories.values())) != 1:
        raise RuntimeError("production benchmark result hashes differ by worker count")
    projected = full_seconds / max(1, len(job_specs)) * len(execution) * 132
    return {
        "schema": "stage24_representative_production_benchmark_v1",
        "status": "pass",
        "full_registry_compiled_rows": len(execution),
        "stratified_units": len(job_specs),
        "units_by_family_fold": 30,
        "outer_folds": len(outer_paths),
        "kda02b_outer_folds": 9,
        "kda02b_real_dispatch_units": sum(family == "KDA02B_SURVIVOR_ADJUDICATION_V1" for family, _, _, _ in job_specs),
        "worker_side_cache_decoding": True,
        "parent_decoded_frames_before_fork": 0,
        "full_sample_seconds": full_seconds,
        "full_sample_peak_process_tree_rss_bytes": full.get("peak_process_tree_rss_bytes"),
        "full_sample_peak_output_bytes": full.get("peak_output_bytes"),
        "registry_compile_seconds": compile_seconds,
        "worker_scaling": scaling,
        "worker_scaling_result_inventory_sha256": inventories,
        "conservative_projected_seconds_2x": projected * 2.0,
        "projection_basis": "measured real CacheAuthority FamilyInput engine/dispatcher/accounting sample; 2x operational safety factor",
        "economic_outcomes_opened": False,
        "real_post_entry_rows_opened": 0,
    }


__all__ = ["control_production_shadow_probe", "execute_control_fixture", "representative_production_benchmark"]
