from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_bytes, atomic_write_json, canonical_hash, file_record, sha256_file
from .compiler import EXPECTED_FINAL_CONTROLS, SourcePaths, compile_deterministic
from .runtime import ResourceLimits, physical_memory_bytes, process_tree_rss, synthetic_recovery_canary
from .schema import CAMPAIGN_ID, FAMILY_ORDER
from .validators import (
    aggregate_materialized_probe,
    end_to_end_family_probe,
    independent_replay,
    validate_compiled,
)


CODE_PATHS = (
    "tools/build_stage22_core_liquid_campaign.py",
    "tools/core_liquid_campaign/__init__.py",
    "tools/core_liquid_campaign/canonical.py",
    "tools/core_liquid_campaign/schema.py",
    "tools/core_liquid_campaign/generator.py",
    "tools/core_liquid_campaign/budget.py",
    "tools/core_liquid_campaign/compiler.py",
    "tools/core_liquid_campaign/controls.py",
    "tools/core_liquid_campaign/engine_types.py",
    "tools/core_liquid_campaign/executor.py",
    "tools/core_liquid_campaign/accounting.py",
    "tools/core_liquid_campaign/selection.py",
    "tools/core_liquid_campaign/runtime.py",
    "tools/core_liquid_campaign/synthetic.py",
    "tools/core_liquid_campaign/validators.py",
    "tools/core_liquid_campaign/packet.py",
    "tools/core_liquid_campaign/family_engines/__init__.py",
    "tools/core_liquid_campaign/family_engines/common.py",
    "tools/core_liquid_campaign/family_engines/a4_tsmom.py",
    "tools/core_liquid_campaign/family_engines/a1_compression.py",
    "tools/core_liquid_campaign/family_engines/a2_context.py",
    "tools/core_liquid_campaign/family_engines/a3_starter_retest.py",
    "tools/core_liquid_campaign/family_engines/kda02b_adjudication.py",
    "unit_tests/test_core_liquid_campaign.py",
)


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
    files = []
    for relative in CODE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "source_tree_sha256": sha256_file_from_records(files)}


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
        "rankable_funding_package_sha256": "6c0969727c882ca6439c57c3b7d03367e1b2d26ee46823e56b983756d026ef64",
        "cache_contract_sha256": cache_contract_hash,
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


def benchmark(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = _registry_rows(root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl")
    rows: list[dict[str, Any]] = []
    elapsed_by_family: dict[str, float] = {}
    for family in FAMILY_ORDER:
        before_rss = process_tree_rss()
        started = time.perf_counter()
        result = end_to_end_family_probe(family)
        elapsed = time.perf_counter() - started
        after_rss = process_tree_rss()
        if result["status"] != "complete" or not result.get("synthetic_only"):
            raise AssertionError(f"real-dispatch synthetic benchmark failed: {family}")
        elapsed_by_family[family] = elapsed
        count = sum(row["family_id"] == family and row["execution_disposition"] != "multiplicity_only_duplicate" for row in registry)
        rows.append({
            "family": family,
            "registered_execution_addresses": count,
            "real_dispatch_invocations": 1,
            "ledger_rows": len(result["ledger"]),
            "observation_rows": len(result["observations"]),
            "elapsed_seconds": f"{elapsed:.9f}",
            "rss_before_bytes": before_rss,
            "rss_after_bytes": after_rss,
            "economic_outcomes_opened": "false",
        })
    limits = ResourceLimits()
    single_worker_seconds = sum(
        elapsed_by_family[family]
        * sum(row["family_id"] == family and row["execution_disposition"] != "multiplicity_only_duplicate" for row in registry)
        for family in FAMILY_ORDER
    )
    projection = {
        "schema": "stage22_runtime_projection_v2",
        "basis": {
            "measurement": "one complete synthetic raw-input->engine->accounting->aggregate dispatch per family",
            "registered_execution_addresses": sum(row["execution_disposition"] != "multiplicity_only_duplicate" for row in registry),
            "measured_single_worker_synthetic_seconds": single_worker_seconds,
            "measured_four_worker_synthetic_floor_seconds": single_worker_seconds / limits.max_workers,
            "registry_bytes": (root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl").stat().st_size,
            "raw_coordinate_bytes": (root / "RAW_SPACE_FILLING_COORDINATES.parquet").stat().st_size,
            "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": physical_memory_bytes(),
            "current_process_tree_rss_bytes": process_tree_rss(),
            "maxrss_ru_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "interpretation": "measured synthetic compute floor, not an economic wall-time estimate; the detached supervisor uses renewable checkpoints and bounded safety stops",
        "hard_wall_time": None,
        "limits": {
            "workers": limits.max_workers,
            "jobs_in_flight": limits.max_jobs_in_flight,
            "aggregate_process_tree_rss_bytes": limits.max_rss_bytes,
            "campaign_output_bytes": limits.max_output_bytes,
            "minimum_free_disk_bytes": limits.minimum_free_disk_bytes,
            "minimum_free_disk_fraction": limits.minimum_free_disk_fraction,
            "heartbeat_seconds": limits.heartbeat_seconds,
            "graceful_stop_seconds": limits.graceful_stop_seconds,
            "no_progress_seconds": limits.no_progress_seconds,
            "retry_delays_seconds": list(limits.retry_delays_seconds),
            "maximum_supervisor_restarts": limits.maximum_supervisor_restarts,
        },
        "economic_outcomes_opened": False,
    }
    return rows, projection


def _generated_inventory(root: Path) -> dict[str, Any]:
    excluded = {
        "INDEPENDENT_REVIEW_TARGET.json",
        "FINAL_CAMPAIGN_MANIFEST.json",
        "FINAL_HUMAN_APPROVAL_REQUEST.json",
        "FINAL_LAUNCH_TASK.md",
        "FINAL_PACKET_HASH_INVENTORY.json",
        "ARTIFACT_MANIFEST.json",
    }
    records = [file_record(root, path) for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded]
    return {"artifacts": records, "sha256": sha256_file_from_records(records)}


def build_candidate(paths: SourcePaths, output_root: Path, repository_root: Path, implementation_commit: str) -> dict[str, Any]:
    actual_commit = subprocess.run(["git", "-C", str(repository_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual_commit != implementation_commit:
        raise ValueError("implementation commit is not repository HEAD")
    compiled = compile_deterministic(paths, output_root)
    input_authority = _write_execution_input_authority(output_root, repository_root)
    validation = validate_compiled(output_root)
    replay = independent_replay(paths, output_root)
    aggregate = aggregate_materialized_probe()
    if not aggregate["pass"]:
        raise AssertionError("aggregate/materialized probe failed")
    atomic_write_json(output_root / "AGGREGATE_VS_MATERIALIZED_PROBE_AUDIT.json", aggregate)
    benchmark_rows, runtime_projection = benchmark(output_root)
    _write_csv(
        output_root / "ENGINE_BENCHMARK_SUMMARY.csv",
        ("family", "registered_execution_addresses", "real_dispatch_invocations", "ledger_rows", "observation_rows", "elapsed_seconds", "rss_before_bytes", "rss_after_bytes", "economic_outcomes_opened"),
        benchmark_rows,
    )
    atomic_write_json(output_root / "RUNTIME_PROJECTION.json", runtime_projection)
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
