from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cache import build_semantic_cache
from .campaign import synthetic_production_path_job
from .canonical import atomic_write_bytes, atomic_write_json, canonical_hash, file_record, sha256_file
from .compiler import EXPECTED_FINAL_CONTROLS, SourcePaths, compile_deterministic, shared_semantic_cache_contract
from .executor import CacheAuthority
from .runtime import LazySupervisor, ResourceLimits, directory_size, physical_memory_bytes, process_tree_rss, synthetic_recovery_canary
from .schema import CAMPAIGN_ID, FAMILY_ORDER, baseline_config, economic_address, normalize_config
from .synthetic import frame_for_family, with_source_authority
from .validators import (
    aggregate_materialized_probe,
    end_to_end_family_probe,
    independent_replay,
    validate_compiled,
)


INHERITED_STAGE21_MANIFEST_SHA256 = "edeb26248831791b90c7aede801afd5a258ecf4f61710eb3146852b94b4f34c7"


EXECUTION_SOURCE_RECORDS = (
    (
        "price_and_instrument_source_manifest",
        "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/UNIT_VERIFICATION_SOURCE_MANIFEST.json",
        "9074a6bbefe9ab7fd3c579c6b90260f1b244c8e4f52f79e60f180a170e4110f0",
    ),
    (
        "funding_partition_manifest",
        "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/FUNDING_SOURCE_AND_PARTITION_MANIFEST.json",
        "518a70c53830f8a01395d5836403b027f2371bd03f2fee05ffd3346cfca4a78d",
    ),
    (
        "rankable_funding_package_manifest",
        "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2/RANKABLE_FUNDING_PACKAGE_MANIFEST.json",
        "e048915160a6f55b313568f9a2df908b196c04557fb719273ad07957efcb6f9b",
    ),
    (
        "rankable_funding_package",
        "/opt/testerdonch/results/rebaseline/phase_kraken_local_official_funding_export_20260720_v4/kraken_funding_rankable_2023_2025.zip",
        "6c0969727c882ca6439c57c3b7d03367e1b2d26ee46823e56b983756d026ef64",
    ),
    (
        "kraken_acquisition_manifest",
        "/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv",
        "f598cc1fb5714386923272399b98fa560c119c96fd5af33f5b30735f40cea420",
    ),
    (
        "campaign_universe_reconciliation",
        "docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1/KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.csv",
        "54e42b05c09a7e2ff9deec1e4f1f16e63a76dbd23979924ed70cbdfdc3a52d6b",
    ),
    (
        "terminal_lifecycle_source_ledger",
        "docs/agent/task_archive/20260717_donch_bt_stage_2a1_c01_reference_panel_20260717_v1/TERMINAL_LIFECYCLE_SOURCE_LEDGER.csv",
        "2fbe1cfcafc5d1c0f2d8ca91d75586fbe245070b69a4ec315d3b588450829146",
    ),
)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [",".join(fieldnames)]
    for row in rows:
        values = []
        for field in fieldnames:
            value = str(row.get(field, ""))
            if any(character in value for character in ',"\n'):
                value = '"' + value.replace('"', '""') + '"'
            values.append(value)
        lines.append(",".join(values))
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def sha256_file_from_records(records: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    payload = "".join(
        f"{record['sha256']}  {record['path']}\n"
        for record in sorted(records, key=lambda item: str(item["path"]))
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_inventory(repository_root: Path) -> dict[str, Any]:
    package = repository_root / "tools/core_liquid_campaign"
    discovered = [path.relative_to(repository_root).as_posix() for path in package.rglob("*.py") if path.is_file()]
    discovered.extend((
        "tools/build_stage22_core_liquid_campaign.py",
        "tools/build_stage23_final_packet.py",
        "tools/build_stage24_final_packet.py",
        "tools/run_stage22_core_liquid_campaign.py",
        "unit_tests/test_core_liquid_campaign.py",
        "unit_tests/test_core_liquid_campaign_stage23.py",
        "unit_tests/test_core_liquid_campaign_stage24.py",
    ))
    files = []
    for relative in sorted(set(discovered)):
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "complete_launch_tree": True,
        "discovery_contract": "all Python files below tools/core_liquid_campaign plus exact build/run CLIs and focused unit test",
        "files": files,
        "source_tree_sha256": sha256_file_from_records(files),
    }


def _registry_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_execution_input_authority(output_root: Path, repository_root: Path) -> dict[str, Any]:
    verified_sources = []
    for role, relative, expected_hash in EXECUTION_SOURCE_RECORDS:
        path = repository_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"execution source authority mismatch: {role}")
        verified_sources.append({"role": role, "path": relative, "bytes": path.stat().st_size, "sha256": expected_hash})
    pit_contract = {
        "schema": "stage22_pit_universe_contract_v1",
        "platform": "kraken_native_linear_pf",
        "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
        "membership": "only the 187 hash-bound campaign symbols; per-decision eligibility requires lifecycle-valid status and source_close_ts<=decision_ts",
        "liquidity": "lagged rolling quote-notional computed only from fully completed bars; rank is fitted on the point-in-time eligible population",
        "missingness": "missing lifecycle, eligibility, or required lagged liquidity is unavailable_no_trade; never zero-fill, forward-fill, or survivor substitute",
        "source_records": [record for record in verified_sources if record["role"] in {"price_and_instrument_source_manifest", "campaign_universe_reconciliation", "terminal_lifecycle_source_ledger"}],
        "protected_rows_opened": 0,
    }
    atomic_write_json(output_root / "PIT_UNIVERSE_CONTRACT.json", pit_contract)
    cache_contract_hash = sha256_file(output_root / "SHARED_SEMANTIC_CACHE_CONTRACT.json")
    authority = {
        "schema": "stage22_execution_input_authority_v1",
        "platform": "kraken_native_linear_pf",
        "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
        "source_manifest_sha256": verified_sources[0]["sha256"],
        "pit_universe_sha256": sha256_file(output_root / "PIT_UNIVERSE_CONTRACT.json"),
        "funding_manifest_sha256": verified_sources[1]["sha256"],
        "rankable_funding_package_sha256": next(record["sha256"] for record in verified_sources if record["role"] == "rankable_funding_package"),
        "cache_contract_sha256": cache_contract_hash,
        "fold_graph_sha256": sha256_file(output_root / "FOLD_GRAPH.json"),
        "cache_manifest_contract": {
            "schema": "stage22_semantic_cache_manifest_v1",
            "artifact_inventory": "nonempty; every physical cache artifact is SHA-256 verified and binds its deserialized FamilyInput content hash",
            "construction": "deterministic derivative only from the bound sources and code; source values, formulas, boundaries, universe, funding, and event keys cannot be caller-selected",
            "launch_gate": "build, independently replay, and atomically verify before any economic attempt",
        },
        "source_records": verified_sources,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    }
    atomic_write_json(output_root / "EXECUTION_INPUT_AUTHORITY.json", authority)
    return authority


def _synthetic_row(family: str, identifier: str) -> dict[str, Any]:
    config = normalize_config(family, baseline_config(family))
    return {
        "campaign_id": CAMPAIGN_ID,
        "executable_attempt_id": identifier,
        "family_id": family,
        "config": config,
        "canonical_economic_address_sha256": economic_address(family, config)[1],
        "execution_disposition": "execute_once",
        "duplicate_of_executable_attempt_id": None,
    }


def benchmark(
    root: Path,
    *,
    execution_authority: Mapping[str, Any] | None = None,
    authority_root: Path | None = None,
    state_name: str = "PRODUCTION_PATH_CAPACITY_CANARY_STATE",
    development_dispatches_per_address: int = 1,
) -> dict[str, Any]:
    """Measure four concurrent workers through cache decode, dispatch and atomic persistence."""
    state_root = root / state_name
    if execution_authority is None:
        source = state_root / "SYNTHETIC_SOURCE_AUTHORITY.json"
        atomic_write_json(source, {"schema": "stage22_synthetic_capacity_source_v1", "values": "deterministic_nonmarket_fixtures", "economic_outcomes_opened": False})
        source_record = {"role": "synthetic_capacity_source", "path": source.relative_to(root).as_posix(), "bytes": source.stat().st_size, "sha256": sha256_file(source)}
        authority = {
            "schema": "stage22_execution_input_authority_v1", "platform": "kraken_native_linear_pf",
            "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
            "source_manifest_sha256": source_record["sha256"], "pit_universe_sha256": canonical_hash({"synthetic": "PIT universe"}),
            "funding_manifest_sha256": canonical_hash({"synthetic": "funding"}), "rankable_funding_package_sha256": canonical_hash({"synthetic": "funding package"}),
            "cache_contract_sha256": canonical_hash(shared_semantic_cache_contract()), "fold_graph_sha256": canonical_hash({"synthetic": "fold graph"}),
            "cache_manifest_contract": {"schema": "stage22_semantic_cache_manifest_v1"}, "source_records": [source_record],
        }
        verified_authority_root = root
    else:
        authority = dict(execution_authority)
        if authority_root is None:
            raise ValueError("authority-bound capacity benchmark requires its repository root")
        verified_authority_root = authority_root
    parent = _synthetic_row("A1_COMPRESSION_V2", "capacity-parent-a1")
    rows = {family: _synthetic_row(family, f"capacity-{family}") for family in FAMILY_ORDER if family != "A2_PRIOR_HIGH_RS_CONTEXT_V1"}
    overlay = _synthetic_row("A2_PRIOR_HIGH_RS_CONTEXT_V1", "capacity-a2")
    template = canonical_hash({"capacity": "a2-parent-slot"})
    overlay.update({
        "execution_disposition": "execute_if_parent_available",
        "parent_binding_template_id": template,
        "parent_only_counterpart_id": canonical_hash({"template": template, "role": "parent_only"}),
        "overlay_counterpart_id": canonical_hash({"template": template, "role": "overlay"}),
    })
    rows["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = overlay
    registry = {row["executable_attempt_id"]: row for row in (*rows.values(), parent)}
    frames_by_family = {
        family: with_source_authority(frame_for_family(family, row["config"]), authority)
        for family, row in rows.items() if family != "A2_PRIOR_HIGH_RS_CONTEXT_V1"
    }
    parent_frame = with_source_authority(frame_for_family("A1_COMPRESSION_V2", parent["config"]), authority)
    frames_by_family["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = parent_frame
    unique_frames = {frame.content_sha256(): frame for frame in frames_by_family.values()}
    cache_root = state_root / "cache"
    cache_started = time.perf_counter()
    cache_manifest_path = build_semantic_cache(cache_root, authority, tuple(unique_frames.values()), authority_root=verified_authority_root, synthetic_only=True)
    cache_seconds = time.perf_counter() - cache_started
    cache_payload = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    path_by_hash = {record["frame_content_sha256"]: record["path"] for record in cache_payload["artifacts"]}
    cache_authority = CacheAuthority(cache_manifest_path, cache_root)
    campaign_manifest = {"execution_input_authority": authority}
    binding = {
        "parent_binding_template_id": template,
        "parent_executable_attempt_id": parent["executable_attempt_id"],
        "parent_only_counterpart_id": overlay["parent_only_counterpart_id"],
        "overlay_counterpart_id": overlay["overlay_counterpart_id"],
    }
    jobs = []
    for replicate in range(4):
        for family in FAMILY_ORDER:
            row = rows[family]; path = path_by_hash[frames_by_family[family].content_sha256()]
            job_id = f"capacity:{replicate}:{family}"
            base = synthetic_production_path_job(
                row,
                cache_authority=cache_authority,
                campaign_manifest=campaign_manifest,
                artifact_paths=[path],
                registry=registry,
                job_id=job_id,
                parent_binding=binding if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None,
                parent_artifact_paths=[path] if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else None,
            )
            def timed(base=base, family=family):
                started = time.perf_counter(); result = base(); result["benchmark_family"] = family; result["benchmark_elapsed_seconds"] = time.perf_counter() - started; return result
            jobs.append((job_id, timed))
    peak_rss = {"bytes": process_tree_rss()}
    def rss_sample() -> int:
        value = process_tree_rss(); peak_rss["bytes"] = max(peak_rss["bytes"], value); return value
    heartbeats: list[Mapping[str, Any]] = []
    class CanaryClock:
        value = 0.0
        def __call__(self) -> float:
            self.value += 0.2
            return self.value
    limits = ResourceLimits(max_workers=4, max_jobs_in_flight=4, heartbeat_seconds=1, monitor_interval_seconds=0.005)
    started = time.perf_counter()
    supervisor = LazySupervisor(
        state_root / "supervisor",
        limits,
        heartbeat=lambda payload: (heartbeats.append(dict(payload)) or True),
        real_unit_validator=lambda job_id, result: isinstance(result, Mapping) and result.get("registered_job_id") == job_id and result.get("synthetic_only") is True,
        rss_sampler=rss_sample,
        monotonic=CanaryClock(),
    ).run(iter(jobs))
    concurrent_seconds = time.perf_counter() - started
    if supervisor["status"] != "complete" or not supervisor.get("health_release") or not heartbeats:
        raise AssertionError("four-worker production-path capacity canary failed")
    io_path = state_root / "throughput-probe.bin"; io_bytes = 8 * 1024**2; payload = b"\0" * io_bytes
    io_started = time.perf_counter()
    with io_path.open("wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    io_seconds = max(time.perf_counter() - io_started, 1e-9); io_sha = sha256_file(io_path); io_path.unlink()
    family_results: dict[str, list[tuple[float, int]]] = {family: [] for family in FAMILY_ORDER}
    for marker in (state_root / "supervisor/markers").glob("*.json"):
        record = json.loads(marker.read_text(encoding="utf-8")); artifact = state_root / "supervisor" / record["artifact"]
        result = json.loads(artifact.read_text(encoding="utf-8"))["result"]
        family_results[result["benchmark_family"]].append((float(result["benchmark_elapsed_seconds"]), artifact.stat().st_size))
    families = []
    for family in FAMILY_ORDER:
        samples = family_results[family]
        if len(samples) != 4:
            raise AssertionError(f"capacity benchmark did not reconcile four {family} dispatches")
        families.append({
            "family": family,
            "dispatches": 4,
            "seconds_per_dispatch": sum(value for value, _ in samples) / len(samples),
            "output_bytes_per_dispatch": sum(value for _, value in samples) / len(samples),
        })
    usage = shutil.disk_usage(root)
    return {
        "schema": "stage22_four_worker_production_path_capacity_v1",
        "status": "pass",
        "workers": 4,
        "jobs_in_flight": 4,
        "production_dispatches": len(jobs),
        "development_dispatches_per_address": development_dispatches_per_address,
        "families": families,
        "cache_construction_seconds": cache_seconds,
        "cache_bytes": directory_size(cache_root),
        "concurrent_wall_seconds": concurrent_seconds,
        "peak_process_tree_rss_bytes": peak_rss["bytes"],
        "rss_limit_bytes": limits.max_rss_bytes,
        "persisted_output_bytes": directory_size(state_root / "supervisor"),
        "output_limit_bytes": limits.max_output_bytes,
        "disk_probe_bytes": io_bytes,
        "disk_probe_sha256": io_sha,
        "disk_write_seconds": io_seconds,
        "disk_write_bytes_per_second": io_bytes / io_seconds,
        "free_disk_bytes_after_probe": usage.free,
        "minimum_free_disk_bytes": max(limits.minimum_free_disk_bytes, int(usage.total * limits.minimum_free_disk_fraction)),
        "scheduled_heartbeat_count": len(heartbeats),
        "health_release": supervisor["health_release"],
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    }


def runtime_projection(root: Path, capacity: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = _registry_rows(root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl")
    execution_count = sum(row["execution_disposition"] != "multiplicity_only_duplicate" for row in registry)
    family_measurement = {row["family"]: row for row in capacity["families"]}
    rows = []
    projected_worker_seconds = 0.0; projected_output = 0.0
    multiplier = int(capacity["development_dispatches_per_address"])
    for family in FAMILY_ORDER:
        count = sum(row["family_id"] == family and row["execution_disposition"] != "multiplicity_only_duplicate" for row in registry)
        measured = family_measurement[family]
        family_multiplier = 1 if family == "KDA02B_SURVIVOR_ADJUDICATION_V1" else multiplier
        projected_worker_seconds += count * family_multiplier * float(measured["seconds_per_dispatch"])
        projected_output += count * family_multiplier * float(measured["output_bytes_per_dispatch"])
        rows.append({
            "family": family, "registered_execution_addresses": count, "real_dispatch_invocations": measured["dispatches"],
            "elapsed_seconds": f"{float(measured['seconds_per_dispatch']):.9f}", "output_bytes_per_dispatch": f"{float(measured['output_bytes_per_dispatch']):.3f}",
            "economic_outcomes_opened": "false",
        })
    limits = ResourceLimits()
    projection = {
        "schema": "stage22_runtime_projection_v3",
        "basis": {
            "measurement": "four concurrent workers through canonical cache decode, real dispatcher/accounting and atomic supervisor persistence; four samples per family",
            "capacity_measurement_sha256": canonical_hash(capacity),
            "registered_execution_addresses": execution_count,
            "projected_worker_seconds": projected_worker_seconds,
            "projected_four_worker_seconds": projected_worker_seconds / 4,
            "projected_output_bytes": projected_output,
            "measured_peak_process_tree_rss_bytes": capacity["peak_process_tree_rss_bytes"],
            "measured_disk_write_bytes_per_second": capacity["disk_write_bytes_per_second"],
            "logical_cpus": os.cpu_count(), "physical_memory_bytes": physical_memory_bytes(),
            "current_process_tree_rss_bytes": process_tree_rss(), "maxrss_ru_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "interpretation": "measured outcome-free production-path projection; renewable checkpoints retain no hard wall stop",
        "hard_wall_time": None,
        "limits": {
            "workers": limits.max_workers, "jobs_in_flight": limits.max_jobs_in_flight,
            "aggregate_process_tree_rss_bytes": limits.max_rss_bytes, "campaign_output_bytes": limits.max_output_bytes,
            "minimum_free_disk_bytes": limits.minimum_free_disk_bytes, "minimum_free_disk_fraction": limits.minimum_free_disk_fraction,
            "heartbeat_seconds": limits.heartbeat_seconds, "graceful_stop_seconds": limits.graceful_stop_seconds,
            "no_progress_seconds": limits.no_progress_seconds, "retry_delays_seconds": list(limits.retry_delays_seconds),
            "maximum_supervisor_restarts": limits.maximum_supervisor_restarts,
        },
        "safety_margins": {
            "rss_bytes": limits.max_rss_bytes - int(capacity["peak_process_tree_rss_bytes"]),
            "projected_output_bytes": limits.max_output_bytes - int(projected_output),
            "free_disk_bytes": int(capacity["free_disk_bytes_after_probe"]) - int(capacity["minimum_free_disk_bytes"]),
        },
        "economic_outcomes_opened": False,
    }
    if min(projection["safety_margins"].values()) <= 0:
        raise AssertionError("measured production-path resource margin is not positive")
    return rows, projection


def _development_dispatch_multiplier(paths: SourcePaths) -> int:
    with zipfile.ZipFile(paths.stage21_v1_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith("INNER_OUTER_FOLD_MAP.json")]
        if len(names) != 1:
            raise ValueError("Stage-21 V1 ZIP has no unique fold graph")
        graph = json.loads(archive.read(names[0]))
    value = sum(len(outer["inner_folds"]) for outer in graph["outer_folds"])
    if value <= 0:
        raise ValueError("fold graph has no development dispatches")
    return value


def _generated_inventory(root: Path) -> dict[str, Any]:
    excluded = {
        "INDEPENDENT_REVIEW_TARGET.json",
        "FINAL_CAMPAIGN_MANIFEST.json",
        "FINAL_HUMAN_APPROVAL_REQUEST.json",
        "FINAL_LAUNCH_TASK.md",
        "FINAL_PACKET_HASH_INVENTORY.json",
        "ARTIFACT_MANIFEST.json",
        "INDEPENDENT_PREOUTCOME_REVIEW.json",
    }
    records = [file_record(root, path) for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded]
    return {"artifacts": records, "sha256": sha256_file_from_records(records)}


def build_candidate(paths: SourcePaths, output_root: Path, repository_root: Path, implementation_commit: str) -> dict[str, Any]:
    actual_commit = subprocess.run(["git", "-C", str(repository_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual_commit != implementation_commit:
        raise ValueError("implementation commit is not repository HEAD")
    output_root.mkdir(parents=True, exist_ok=True)
    dispatch_multiplier = _development_dispatch_multiplier(paths)
    provisional = benchmark(output_root, state_name="PROVISIONAL_CAPACITY_CANARY_STATE", development_dispatches_per_address=dispatch_multiplier)
    compile_deterministic(paths, output_root, provisional)
    input_authority = _write_execution_input_authority(output_root, repository_root)
    capacity = benchmark(
        output_root,
        execution_authority=input_authority,
        authority_root=repository_root,
        state_name="AUTHORITY_BOUND_PRODUCTION_PATH_CANARY_STATE",
        development_dispatches_per_address=dispatch_multiplier,
    )
    atomic_write_json(output_root / "CAPACITY_BENCHMARK.json", capacity)
    atomic_write_json(output_root / "PRODUCTION_PATH_SYNTHETIC_CANARY.json", {**capacity, "canary_role": "authority-bound cache/decode/dispatcher/supervisor/heartbeat/health-release"})
    shutil.rmtree(output_root / "PROVISIONAL_CAPACITY_CANARY_STATE")
    compiled = compile_deterministic(paths, output_root, capacity)
    input_authority = _write_execution_input_authority(output_root, repository_root)
    validation = validate_compiled(output_root)
    replay = independent_replay(paths, output_root, capacity)
    aggregate = aggregate_materialized_probe()
    if not aggregate["pass"]:
        raise AssertionError("aggregate/materialized probe failed")
    atomic_write_json(output_root / "AGGREGATE_VS_MATERIALIZED_PROBE_AUDIT.json", aggregate)
    benchmark_rows, projection = runtime_projection(output_root, capacity)
    _write_csv(
        output_root / "ENGINE_BENCHMARK_SUMMARY.csv",
        ("family", "registered_execution_addresses", "real_dispatch_invocations", "elapsed_seconds", "output_bytes_per_dispatch", "economic_outcomes_opened"),
        benchmark_rows,
    )
    atomic_write_json(output_root / "RUNTIME_PROJECTION.json", projection)
    canary = synthetic_recovery_canary(output_root / "SYNTHETIC_RUNTIME_CANARY_STATE")
    if not canary["pass"]:
        raise AssertionError("runtime canary failed")
    atomic_write_json(output_root / "SYNTHETIC_RUNTIME_CANARY.json", canary)
    code = _code_inventory(repository_root)
    atomic_write_json(output_root / "CODE_HASH_INVENTORY.json", {"schema": "stage22_code_hash_inventory_v2", "implementation_commit": implementation_commit, **code})
    inventory = _generated_inventory(output_root)
    candidate = {
        "schema": "stage22_review_candidate_v2",
        "campaign_id": CAMPAIGN_ID,
        "implementation_commit": implementation_commit,
        "compiler_summary": compiled,
        "validation": validation,
        "independent_deterministic_replay": replay,
        "aggregate_materialized_probe_pass": aggregate["pass"],
        "runtime_canary_pass": canary["pass"],
        "execution_input_authority": input_authority,
        "review_bindings": {
            "implementation_commit": implementation_commit,
            "source_tree_sha256": code["source_tree_sha256"],
            "family_axis_schema_file_sha256": sha256_file(output_root / "FAMILY_AXIS_SCHEMA.json"),
            "strategy_registry_sha256": sha256_file(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"),
            "execution_registry_sha256": sha256_file(output_root / "FINAL_EXECUTION_REGISTRY.jsonl"),
            "control_registry_sha256": sha256_file(output_root / "FINAL_CONTROL_REGISTRY.jsonl"),
            "a2_counterpart_registry_sha256": sha256_file(output_root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl"),
            "generated_artifact_inventory_sha256": inventory["sha256"],
        },
        "reviewed_artifacts": inventory["artifacts"],
        "required_review_schema": "stage22_independent_preoutcome_review_v2",
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
        "status": "ready_for_independent_code_and_byte_review",
    }
    atomic_write_json(output_root / "INDEPENDENT_REVIEW_TARGET.json", candidate)
    return candidate


def _dependency_files(root: Path) -> list[Path]:
    excluded = {"FINAL_CAMPAIGN_MANIFEST.json", "FINAL_HUMAN_APPROVAL_REQUEST.json", "FINAL_LAUNCH_TASK.md", "FINAL_PACKET_HASH_INVENTORY.json", "ARTIFACT_MANIFEST.json"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name not in excluded)


def _require_bound_review(output_root: Path, review_path: Path, implementation_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    candidate_path = output_root / "INDEPENDENT_REVIEW_TARGET.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if review.get("schema") != candidate["required_review_schema"] or review.get("verdict") != "PASS" or review.get("blocking_findings") != 0:
        raise ValueError("independent review has not passed under the required schema")
    if review.get("economic_outcomes_opened") is not False or review.get("protected_rows_opened") != 0 or review.get("capitalcom_payload_opened") is not False:
        raise ValueError("independent review does not attest the protected/outcome boundary")
    expected = {**candidate["review_bindings"], "review_target_sha256": sha256_file(candidate_path)}
    if review.get("bindings") != expected or expected["implementation_commit"] != implementation_commit:
        raise ValueError("independent review is not bound to the exact implementation and generated bytes")
    return review, candidate


def finalize_packet(output_root: Path, repository_root: Path, implementation_commit: str, review_path: Path, inherited_manifest_path: Path) -> dict[str, Any]:
    actual_commit = subprocess.run(["git", "-C", str(repository_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual_commit != implementation_commit:
        raise ValueError("repository HEAD differs from the reviewed implementation commit")
    review, candidate = _require_bound_review(output_root, review_path, implementation_commit)
    validation = validate_compiled(output_root)
    live_code = _code_inventory(repository_root)
    if live_code["source_tree_sha256"] != candidate["review_bindings"]["source_tree_sha256"]:
        raise ValueError("live implementation differs from independently reviewed source bytes")
    recomputed_inventory = _generated_inventory(output_root)
    if recomputed_inventory["sha256"] != candidate["review_bindings"]["generated_artifact_inventory_sha256"] or recomputed_inventory["artifacts"] != candidate["reviewed_artifacts"]:
        raise ValueError("generated artifact inventory differs from the independently reviewed bytes")
    if sha256_file(inherited_manifest_path) != INHERITED_STAGE21_MANIFEST_SHA256:
        raise ValueError("inherited Stage-21 manifest physical hash mismatch")
    inherited = json.loads(inherited_manifest_path.read_text(encoding="utf-8"))
    dependency_records = [file_record(output_root, path) for path in _dependency_files(output_root)]
    registry = _registry_rows(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl")
    execution = _registry_rows(output_root / "FINAL_EXECUTION_REGISTRY.jsonl")
    controls = _registry_rows(output_root / "FINAL_CONTROL_REGISTRY.jsonl")
    counts = Counter(row["family_id"] for row in registry)
    input_authority = json.loads((output_root / "EXECUTION_INPUT_AUTHORITY.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": "stage22_final_campaign_manifest_v2",
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen_awaiting_one_exact_external_human_approval",
        "economic_run_authorized_by_manifest": False,
        "repository": {"implementation_commit": implementation_commit, "source_tree_sha256": live_code["source_tree_sha256"]},
        "platform": "Kraken native linear PF only",
        "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
        "protected_period": "[2026-01-01T00:00:00Z,+infinity)",
        "programme_exposure_class": "program_exposed_historical",
        "attempts": {
            "total_rows": len(registry),
            "execution_rows": len(execution),
            "unique_economic_addresses": len({row["canonical_economic_address_sha256"] for row in registry}),
            "by_family": {family: counts[family] for family in FAMILY_ORDER},
        },
        "controls": {"total_rows": len(controls), "unique_economic_addresses": len({row["economic_address_sha256"] for row in controls}), "conditional_on_frozen_parent_slot": True},
        "execution_input_authority": input_authority,
        "primary_hashes": {
            "family_axis_schema": sha256_file(output_root / "FAMILY_AXIS_SCHEMA.json"),
            "strategy_registry": sha256_file(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"),
            "execution_registry": sha256_file(output_root / "FINAL_EXECUTION_REGISTRY.jsonl"),
            "control_registry": sha256_file(output_root / "FINAL_CONTROL_REGISTRY.jsonl"),
            "a2_counterpart_registry": sha256_file(output_root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl"),
            "code_inventory": sha256_file(output_root / "CODE_HASH_INVENTORY.json"),
            "execution_input_authority": sha256_file(output_root / "EXECUTION_INPUT_AUTHORITY.json"),
            "selection_policy": sha256_file(output_root / "SAFE_PRUNING_POLICY.json"),
            "runtime_projection": sha256_file(output_root / "RUNTIME_PROJECTION.json"),
            "aggregate_materialized_probe": sha256_file(output_root / "AGGREGATE_VS_MATERIALIZED_PROBE_AUDIT.json"),
            "independent_review": sha256_file(review_path),
            "review_target": sha256_file(output_root / "INDEPENDENT_REVIEW_TARGET.json"),
        },
        "artifact_dependencies": dependency_records,
        "inherited_economic_dependency_sha256": inherited["economic_dependency_sha256"],
        "inherited_stage21_manifest_sha256": INHERITED_STAGE21_MANIFEST_SHA256,
        "selection_contract": {
            "plateau": "Gower radius 0.15; >=5 distinct cells; >=2 varied axes; >=60% base-positive; median base>0; median stress>-18bps; every inner vector retains explicit negative-infinity empties and >=75% nonempty",
            "beam": "maximum five per family/fold; deterministic lexicographic order; no manual selection",
            "A2": "exact pre-registered parent family/fold/beam slot; missing means unavailable_no_parent; paired counterpart frozen atomically",
            "outer_folds": 8,
            "materialization": "all beam survivors, anchors/nulls, frozen near-misses and deterministic failed audit sample",
            "safe_pruning_policy_sha256": sha256_file(output_root / "SAFE_PRUNING_POLICY.json"),
        },
        "runtime_contract": json.loads((output_root / "RUNTIME_PROJECTION.json").read_text(encoding="utf-8"))["limits"],
        "gates": {
            "typed_schema_complete": True,
            "registry_replay": "pass",
            "control_replay": "pass",
            "engine_semantic_coverage": "pass",
            "aggregate_materialized_probe": "pass",
            "runtime_recovery_canary": "pass",
            "independent_review": "pass",
            "protected_rows_opened": 0,
            "capitalcom_payload_opened": 0,
        },
        "prohibited": ["protected strategy outcomes", "Capital.com payloads", "new acquisition", "capture restart", "C17", "separate Phase 6 expansion", "account actions", "orders", "deployment", "live trading", "force push", "post-approval economic invention"],
        "stop_conditions": ["authority or dependency hash mismatch", "protected or Capital.com reader activation", "schema/registry/control drift", "PIT/funding/execution gate failure", "resource/canary/Telegram/preoutcome-review failure", "economic semantic change required"],
    }
    atomic_write_json(output_root / "FINAL_CAMPAIGN_MANIFEST.json", manifest)
    manifest_hash = sha256_file(output_root / "FINAL_CAMPAIGN_MANIFEST.json")
    approval = {
        "schema": "stage22_final_human_approval_request_v2",
        "campaign_id": CAMPAIGN_ID,
        "status": "awaiting_exact_external_human_approval",
        "final_campaign_manifest_sha256": manifest_hash,
        "bindings": manifest["primary_hashes"],
        "repository_implementation_commit": implementation_commit,
        "strategy_attempt_rows": len(registry),
        "execution_rows": len(execution),
        "control_rows": len(controls),
        "statements": {
            "no_post_approval_economic_invention_remains": True,
            "executor_exists_and_passed_semantic_coverage": True,
            "registry_and_controls_exist": True,
            "synthetic_and_aggregate_materialized_probes_passed": True,
            "runtime_projection_is_measured": True,
            "economic_outcomes_opened_in_stage22": False,
        },
        "authorization_requested": ["launch exact frozen campaign", "runtime-only repairs with semantic/hash preservation", "idempotent recovery", "terminal reconciliation and independent review", "non-force Git publication", "approved Drive handoff", "dynamic continuity publication"],
        "claim_limit": "all 2023-2025 results remain program_exposed_historical; not independent validation and not live-ready",
        "prohibited": manifest["prohibited"],
    }
    atomic_write_json(output_root / "FINAL_HUMAN_APPROVAL_REQUEST.json", approval)
    approval_hash = sha256_file(output_root / "FINAL_HUMAN_APPROVAL_REQUEST.json")
    launch_task = (
        "# Stage 22 final launch task\n\n"
        "No economics are authorized by this file alone. Exact external human approval must bind both hashes below.\n\n"
        f"- final campaign manifest SHA-256: `{manifest_hash}`\n"
        f"- final human approval request SHA-256: `{approval_hash}`\n"
        f"- implementation commit: `{implementation_commit}`\n"
        f"- registered strategy/adjudication attempts: `{len(registry)}` ({len(execution)} unique executions)\n"
        f"- conditional controls: `{len(controls)}`\n\n"
        "After exact approval, repeat every bound hash and safety gate, deterministically build and independently verify the bound semantic caches, run secure Telegram preflight, and launch under the detached supervisor. No post-approval economic translation or selection choice is permitted.\n"
    )
    atomic_write_bytes(output_root / "FINAL_LAUNCH_TASK.md", launch_task.encode("utf-8"))
    inventory_files = sorted(path for path in output_root.rglob("*") if path.is_file() and path.name not in {"FINAL_PACKET_HASH_INVENTORY.json", "ARTIFACT_MANIFEST.json"})
    inventory = {
        "schema": "stage22_final_packet_hash_inventory_v2",
        "campaign_id": CAMPAIGN_ID,
        "artifacts": [file_record(output_root, path) for path in inventory_files],
        "final_campaign_manifest_sha256": manifest_hash,
        "final_human_approval_request_sha256": approval_hash,
        "status": "pass",
    }
    atomic_write_json(output_root / "FINAL_PACKET_HASH_INVENTORY.json", inventory)
    artifact_files = sorted(path for path in output_root.rglob("*") if path.is_file() and path.name != "ARTIFACT_MANIFEST.json")
    atomic_write_json(output_root / "ARTIFACT_MANIFEST.json", {
        "schema": "stage22_artifact_manifest_v2",
        "campaign_id": CAMPAIGN_ID,
        "artifact_count": len(artifact_files),
        "artifacts": [file_record(output_root, path) for path in artifact_files],
        "status": "pass",
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    })
    return {"status": "pass", "final_manifest_sha256": manifest_hash, "final_approval_request_sha256": approval_hash, "validation": validation, "review": review["verdict"]}
