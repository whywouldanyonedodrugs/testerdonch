from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .executor import dispatch_registered_attempt, validate_registered_attempt
from .family_engines.common import EngineInputError
from .kda02b_lazy_family_input import KDA02BLazyFamilyInputAdapter
from .lazy_production_inputs import LazyProductionFamilyInputAdapter
from .population_execution import LaunchPopulationSchedule
from .population_readiness import reconcile_registered_population_routes
from .runtime import LazySupervisor, ResourceLimits, directory_size
from .schema import OUTER_FOLDS
from .selection import aggregate_streaming
from .shadow_payoff import ShadowPayoffProvider


class PopulationBenchmarkError(RuntimeError):
    pass


DIRECT_FAMILIES = ("A4_TSMOM_V7", "A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stratified_rows(rows: Sequence[Mapping[str, Any]], count: int) -> tuple[Mapping[str, Any], ...]:
    """Select configuration-only strata with explicit exit/accounting coverage."""
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        config = row["config"]
        key = str(config.get("exit") or config.get("adjudication_variant") or config.get("overlay_action"))
        by_class[key].append(row)
    for values in by_class.values():
        values.sort(key=lambda row: str(row["canonical_economic_address_sha256"]))
    output: list[Mapping[str, Any]] = []
    while len(output) < min(count, len(rows)):
        progressed = False
        for key in sorted(by_class):
            if by_class[key] and len(output) < count:
                output.append(by_class[key].pop(0)); progressed = True
        if not progressed:
            break
    return tuple(output)


def benchmark_strata(execution: Sequence[Mapping[str, Any]], *, per_family_fold: int = 10) -> dict[str, tuple[Mapping[str, Any], ...]]:
    registry = {str(row["executable_attempt_id"]): row for row in execution}
    result: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for family in DIRECT_FAMILIES:
        result[family] = _stratified_rows([row for row in execution if row["family_id"] == family], per_family_fold)
    result["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = _stratified_rows([
        row for row in execution
        if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1"
        and row["config"].get("parent_binding_mode") == "source_attempt"
        and str(row.get("resolved_parent_executable_attempt_id")) in registry
    ], per_family_fold)
    if any(len(rows) != per_family_fold for rows in result.values()):
        raise PopulationBenchmarkError("a family has fewer than the required ten configuration-only strata")
    return result


def _partitions(schedule: LaunchPopulationSchedule) -> tuple[tuple[str, str, str | None], ...]:
    return tuple(sorted(schedule.partitions, key=lambda key: (OUTER_FOLDS.index(key[1]), key[0] != "inner_validation", str(key[2]))))


def _partition_task(
    *,
    partition_key: tuple[str, str, str | None],
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
    registry: Mapping[str, Mapping[str, Any]],
    schedule: LaunchPopulationSchedule,
    adapter: LazyProductionFamilyInputAdapter,
    source_sizes: Mapping[str, int],
) -> Callable[[], Mapping[str, Any]]:
    def run() -> Mapping[str, Any]:
        phase, outer, inner = partition_key
        provider = ShadowPayoffProvider("stage24-full-population-benchmark-v1")
        frame_seconds = 0.0; engine_seconds = 0.0
        completed = 0; unavailable = 0; materialized_equal = 0
        completed_by_family: Counter[str] = Counter(); unavailable_by_family: Counter[str] = Counter()
        signatures: set[str] = set(); symbols: set[str] = set(); result_rows = []
        for family in (*DIRECT_FAMILIES, "A2_PRIOR_HIGH_RS_CONTEXT_V1"):
            for row in strata[family]:
                parent = None
                parent_binding = None
                if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                    parent = registry[str(row["resolved_parent_executable_attempt_id"])]
                    parent_binding = {
                        "parent_binding_template_id": row["parent_binding_template_id"],
                        "parent_executable_attempt_id": row["resolved_parent_executable_attempt_id"],
                        "parent_only_counterpart_id": row["parent_only_counterpart_id"],
                        "overlay_counterpart_id": row["overlay_counterpart_id"],
                    }
                locators = schedule.representative_locators(
                    row, phase=phase, outer_fold_id=outer, inner_fold_id=inner,
                    symbol_size_bytes=source_sizes, parent_attempt=parent,
                )
                for locator in locators:
                    started = time.monotonic()
                    frame = adapter.frame(locator)
                    parent_frame = None
                    if parent is not None:
                        parent_frame = adapter.frame(replace(
                            locator,
                            family_id=str(parent["family_id"]),
                            executable_attempt_id=str(parent["executable_attempt_id"]),
                            canonical_economic_address_sha256=str(parent["canonical_economic_address_sha256"]),
                        ))
                    frame_seconds += time.monotonic() - started
                    frame.validate(); signatures.update(str(key) for key in frame.threshold_populations); symbols.add(frame.symbol)
                    kwargs: dict[str, Any] = {}
                    if parent is not None:
                        kwargs = {"parent_binding": parent_binding, "parent_frames": (parent_frame,)}
                    started = time.monotonic()
                    try:
                        result = dispatch_registered_attempt(
                            row, (frame,), registry_by_id=registry, payoff_provider=provider, **kwargs,
                        )
                    except EngineInputError as exc:
                        unavailable += 1
                        unavailable_by_family[family] += 1
                        result_rows.append({
                            "family_id": family, "attempt_id": row["executable_attempt_id"],
                            "symbol": frame.symbol, "status": "unavailable_data", "reason": str(exc),
                        })
                        engine_seconds += time.monotonic() - started
                        continue
                    engine_seconds += time.monotonic() - started
                    recomputed = aggregate_streaming(iter(result.get("observations", ())))
                    if recomputed != result.get("aggregate"):
                        raise PopulationBenchmarkError("aggregate/materialized benchmark replay differs")
                    completed += 1; materialized_equal += 1
                    completed_by_family[family] += 1
                    result_rows.append({
                        "family_id": family, "attempt_id": row["executable_attempt_id"],
                        "symbol": frame.symbol, "status": str(result["status"]),
                        "observations": len(result.get("observations", ())),
                        "result_sha256": canonical_hash({
                            "aggregate": result.get("aggregate"),
                            "events": sorted(item.event_id for item in result.get("observations", ())),
                        }),
                    })
        attestation = provider.attestation()
        if attestation["economic_outcomes_opened"] is not False or attestation["real_post_entry_rows_opened"] != 0:
            raise PopulationBenchmarkError("benchmark synthetic-payoff firewall differs")
        expected = 4 * len(next(iter(strata.values()))) * 3
        return {
            "status": "complete", "registered_attempt_id": f"population-partition:{phase}:{outer}:{inner}",
            "partition": {"phase": phase, "outer_fold_id": outer, "inner_fold_id": inner},
            "scheduled_units": expected, "completed_units": completed, "typed_unavailable_units": unavailable,
            "scheduled_units_by_family": {family: 30 for family in (*DIRECT_FAMILIES, "A2_PRIOR_HIGH_RS_CONTEXT_V1")},
            "completed_units_by_family": dict(sorted(completed_by_family.items())),
            "typed_unavailable_units_by_family": dict(sorted(unavailable_by_family.items())),
            "materialized_equal_units": materialized_equal, "frame_construction_seconds": frame_seconds,
            "engine_aggregate_seconds": engine_seconds, "symbols": sorted(symbols),
            "feature_signatures": sorted(signatures), "rows_sha256": canonical_hash(result_rows),
            "shadow_attestation": attestation,
        }
    return run


def _kda_task(
    *,
    outer_fold_id: str,
    rows_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    registry: Mapping[str, Mapping[str, Any]],
    adapter: KDA02BLazyFamilyInputAdapter,
) -> Callable[[], Mapping[str, Any]]:
    def run() -> Mapping[str, Any]:
        provider = ShadowPayoffProvider("stage24-full-population-benchmark-kda-v1")
        records = {}
        typed_unavailable_seen = 0
        for item in adapter.stream(outer_fold_id=outer_fold_id):
            if item.status == "typed_unavailable":
                typed_unavailable_seen += 1
                continue
            records.setdefault(str(item.cell_id), item)
            if len(records) == len(rows_by_cell):
                break
        if set(records) != set(rows_by_cell) or any(record.frame is None for record in records.values()):
            raise PopulationBenchmarkError("KDA02B sample lacks an eligible exact cell/fold frame")
        result_rows = []; materialized_equal = 0
        for cell_id, rows in sorted(rows_by_cell.items()):
            record = records[cell_id]
            assert record.frame is not None
            for row in rows:
                directives = None
                if row["config"]["adjudication_variant"] == "generic_structure_control":
                    directives = {record.frame.content_sha256(): {
                        "allocator": "matched_pseudo_event_allocator_v2", "matched_decision_ts": record.frame.decision_ts,
                    }}
                result = dispatch_registered_attempt(
                    row, (record.frame,), registry_by_id=registry,
                    payoff_provider=provider, control_directives=directives,
                )
                if aggregate_streaming(iter(result.get("observations", ()))) != result.get("aggregate"):
                    raise PopulationBenchmarkError("KDA02B aggregate/materialized benchmark replay differs")
                materialized_equal += 1
                result_rows.append({
                    "attempt_id": row["executable_attempt_id"], "status": result["status"],
                    "observations": len(result.get("observations", ())),
                    "result_sha256": canonical_hash({
                        "aggregate": result.get("aggregate"),
                        "events": sorted(item.event_id for item in result.get("observations", ())),
                    }),
                })
        scheduled = sum(len(rows) for rows in rows_by_cell.values())
        return {
            "status": "complete", "registered_attempt_id": f"kda02b-fold:{outer_fold_id}",
            "cells": len(rows_by_cell), "outer_fold_id": outer_fold_id,
            "scheduled_units": scheduled, "completed_units": scheduled,
            "typed_unavailable_units": 0, "typed_unavailable_index_rows_seen_before_sample_completion": typed_unavailable_seen,
            "materialized_equal_units": materialized_equal,
            "rows_sha256": canonical_hash(result_rows), "shadow_attestation": provider.attestation(),
        }
    return run


def _marker_results(root: Path) -> list[dict[str, Any]]:
    output = []
    for marker_path in sorted((root / "markers").glob("*.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8")); artifact = root / marker["artifact"]
        if sha256_file(artifact) != marker["artifact_sha256"]:
            raise PopulationBenchmarkError("benchmark marker/artifact hash differs")
        output.append(json.loads(artifact.read_text(encoding="utf-8"))["result"])
    return output


def run_population_benchmark(
    *,
    output_root: Path,
    execution_registry_path: Path,
    launch_authority_path: Path,
    kda_manifest_path: Path,
    execution_authority_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Run the real lazy production input/engine path over Stage-24 strata."""
    execution = _jsonl(execution_registry_path)
    for row in execution:
        validate_registered_attempt(row)
    registry = {str(row["executable_attempt_id"]): row for row in execution}
    launch = json.loads(launch_authority_path.read_text(encoding="utf-8"))
    kda_manifest = json.loads(kda_manifest_path.read_text(encoding="utf-8"))
    route_reconciliation = reconcile_registered_population_routes(execution, launch, kda_manifest)
    schedule = LaunchPopulationSchedule(launch_authority_path, sha256_file(launch_authority_path))
    adapter = LazyProductionFamilyInputAdapter.from_launch_population_authority(
        launch_authority_path=launch_authority_path,
        launch_authority_sha256=sha256_file(launch_authority_path),
        repository_root=repository_root,
        construction_mode="shadow_no_outcome",
    )
    kda_adapter = KDA02BLazyFamilyInputAdapter(
        index_root=kda_manifest_path.parent, authority_path=execution_authority_path,
        repository_root=repository_root, mode="shadow_no_outcome",
    )
    source_sizes = {str(row["symbol"]): int(row["bytes"]) for row in adapter.virtual_manifest["artifacts"] if row.get("symbol")}
    strata = benchmark_strata(execution)
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in execution:
        if row["family_id"] == "KDA02B_SURVIVOR_ADJUDICATION_V1":
            by_cell[str(row["config"]["stage20_cell_id"])].append(row)
    if len(by_cell) != 19 or any(len(rows) != 11 for rows in by_cell.values()):
        raise PopulationBenchmarkError("KDA02B benchmark registry is not 19 cells by 11 variants")

    def jobs() -> Iterable[tuple[str, Callable[[], Mapping[str, Any]]]]:
        for partition_key in _partitions(schedule):
            phase, outer, inner = partition_key
            job_id = f"partition:{phase}:{outer}:{inner}"
            yield job_id, _partition_task(
                partition_key=partition_key, strata=strata, registry=registry,
                schedule=schedule, adapter=adapter, source_sizes=source_sizes,
            )
        for outer in ("2023Q4", *OUTER_FOLDS):
            job_id = f"kda02b-fold:{outer}"
            yield job_id, _kda_task(
                outer_fold_id=outer, rows_by_cell={cell: tuple(rows) for cell, rows in by_cell.items()},
                registry=registry, adapter=kda_adapter,
            )

    workers = min(4, os.cpu_count() or 1)
    limits = ResourceLimits(
        max_workers=workers, max_jobs_in_flight=workers, max_rss_bytes=10 * 1024**3,
        max_output_bytes=4 * 1024**3, minimum_free_disk_bytes=8 * 1024**3,
        heartbeat_seconds=1800,
    )
    identity = {
        "execution_registry_sha256": sha256_file(execution_registry_path),
        "launch_population_authority_sha256": sha256_file(launch_authority_path),
        "kda02b_population_authority_sha256": sha256_file(kda_manifest_path),
        "execution_input_authority_sha256": sha256_file(execution_authority_path),
        "provider": "stage24-full-population-benchmark-v1",
    }
    full_root = output_root / "full-stratified"
    started = time.monotonic()
    full = LazySupervisor(full_root, limits, heartbeat=lambda _payload: True, identity_bindings=identity).run(jobs())
    full_seconds = time.monotonic() - started
    if full["status"] != "complete":
        raise PopulationBenchmarkError("full population benchmark stopped before completion")
    restart_started = time.monotonic()
    replay = LazySupervisor(full_root, limits, heartbeat=lambda _payload: True, identity_bindings=identity).run(jobs())
    restart_seconds = time.monotonic() - restart_started
    if replay["status"] != "complete" or replay["completed"] != full["completed"]:
        raise PopulationBenchmarkError("benchmark restart did not reuse the exact marker inventory")
    results = _marker_results(full_root)
    partition_results = [row for row in results if str(row["registered_attempt_id"]).startswith("population-partition:")]
    kda_results = [row for row in results if str(row["registered_attempt_id"]).startswith("kda02b-fold:")]
    scheduled = sum(int(row["scheduled_units"]) for row in results)
    completed = sum(int(row["completed_units"]) for row in results)
    unavailable = sum(int(row["typed_unavailable_units"]) for row in results)
    signatures = {value for row in partition_results for value in row["feature_signatures"]}
    strata_counts = Counter(
        (family, row["partition"]["phase"], row["partition"]["outer_fold_id"], str(row["partition"]["inner_fold_id"]))
        for row in partition_results for family in (*DIRECT_FAMILIES, "A2_PRIOR_HIGH_RS_CONTEXT_V1")
    )
    if len(partition_results) != 132 or len(kda_results) != 9:
        raise PopulationBenchmarkError("benchmark omitted a fold position or KDA02B cell/fold")
    for row in partition_results:
        if any(int(row.get("completed_units_by_family", {}).get(family, 0)) != 30 for family in (*DIRECT_FAMILIES, "A2_PRIOR_HIGH_RS_CONTEXT_V1")):
            raise PopulationBenchmarkError("a family/fold production stratum has fewer than thirty completed units")
    if any(value != 1 for value in strata_counts.values()):
        raise PopulationBenchmarkError("benchmark partition inventory is duplicated")
    output_bytes = directory_size(full_root)
    report = {
        "schema": "stage24_full_population_representative_benchmark_v2", "status": "pass",
        "route_reconciliation": route_reconciliation,
        "registry_compiled_rows": len(execution), "population_partitions": len(partition_results),
        "kda02b_cell_fold_positions": sum(int(row["cells"]) for row in kda_results), "scheduled_dispatch_units": scheduled,
        "completed_dispatch_units": completed, "typed_unavailable_dispatch_units": unavailable,
        "aggregate_materialized_equal_units": sum(int(row["materialized_equal_units"]) for row in results),
        "feature_signatures_exercised": len(signatures), "feature_signature_inventory_sha256": canonical_hash(sorted(signatures)),
        "source_size_strata_per_family_fold": ("small", "median", "large"),
        "configuration_strata_per_family_fold": 10, "dispatch_units_per_A1_A2_A3_A4_fold_position": 30,
        "full_seconds": full_seconds, "restart_reuse_seconds": restart_seconds,
        "restart_reused_markers": len(results), "peak_process_tree_rss_bytes": full.get("peak_process_tree_rss_bytes"),
        "output_bytes": output_bytes, "output_bytes_per_dispatch_unit": output_bytes / max(1, scheduled),
        "frame_construction_seconds": sum(float(row.get("frame_construction_seconds", 0.0)) for row in results),
        "engine_aggregate_seconds": sum(float(row.get("engine_aggregate_seconds", 0.0)) for row in results),
        "conservative_projected_seconds_2x": full_seconds * 2.0,
        "projection_basis": "complete opportunity census plus real lazy FamilyInput/engine/accounting strata; two-times measured wall time",
        "workers": workers, "bounded_jobs_in_flight": workers,
        "economic_outcomes_opened": False, "real_post_entry_rows_opened": 0,
        "protected_rows_opened": 0, "capitalcom_payload_opened": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "FULL_POPULATION_BENCHMARK.json", report)
    return report


__all__ = ["PopulationBenchmarkError", "benchmark_strata", "run_population_benchmark"]
