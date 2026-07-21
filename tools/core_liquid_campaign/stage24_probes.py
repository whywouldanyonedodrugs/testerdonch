from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_hash, sha256_file
from .controls import CONTROL_IDS, effective_seed, execute_control, reconcile_control_duplicates
from .runtime import LazySupervisor, ResourceLimits
from .schema import CAMPAIGN_ID, baseline_config, economic_address, normalize_config
from .shadow_payoff import ShadowPayoffProvider
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


def execute_control_fixture(family: str, control_id: str, *, reverse: bool = False) -> dict[str, Any]:
    fixture = _fixtures()[family]
    frames = list(fixture["frames"])
    if reverse:
        frames.reverse()
    seed = 3 if control_id in {
        "A4_SIGN_PERMUTED_MAIN_NULL",
        "A2_CONTEXT_PERMUTED_MAIN_NULL",
        "A3_RETEST_TIME_PERMUTED_MAIN_NULL",
    } else 1
    control = {
        "family": family,
        "control_id": control_id,
        "effective_seed": seed,
        "economic_address_sha256": canonical_hash({"stage24_control": control_id}),
        "control_attempt_id": f"stage24-{control_id}",
        "execution_status": "execute_once",
    }
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


def control_production_shadow_probe(output_root: Path) -> dict[str, Any]:
    identities = [(family, control_id) for family, ids in CONTROL_IDS.items() for control_id in ids]
    sequential = {control_id: execute_control_fixture(family, control_id) for family, control_id in identities}
    reverse = {control_id: execute_control_fixture(family, control_id, reverse=True) for family, control_id in identities}
    if any(row["status"] != "complete" or int(row["observation_count"]) <= 0 for row in sequential.values()):
        raise RuntimeError("one or more control classes lacks a nonempty applicable fixture")
    if sequential != reverse:
        raise RuntimeError("control transformed result differs under reversed input order")

    worker_hashes: dict[str, str] = {}
    for workers in (1, min(4, __import__("os").cpu_count() or 1)):
        root = output_root / f"workers-{workers}"
        jobs = [
            (control_id, lambda family=family, control_id=control_id: execute_control_fixture(family, control_id))
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
        records = _read_supervisor_results(root)
        if records != sequential:
            raise RuntimeError("control worker execution differs from sequential replay")
        worker_hashes[str(workers)] = canonical_hash(records)
    if len(set(worker_hashes.values())) != 1:
        raise RuntimeError("control transformed hashes differ by worker count")

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
        "reversed_input_order": "pass",
        "restart_reuse": "pass",
        "singleton_case": singleton_result["status"],
        "missing_parent_case": missing["execution_status"],
        "duplicate_case_exercised": any(row["execution_status"] == "unavailable_duplicate_address" for row in duplicate),
        "economic_outcomes_opened": False,
    }


__all__ = ["control_production_shadow_probe", "execute_control_fixture"]
