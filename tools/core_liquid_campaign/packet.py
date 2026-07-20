from __future__ import annotations

import csv
import json
import os
import platform
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_bytes, atomic_write_json, file_record, sha256_file
from .compiler import EXPECTED_FINAL_ATTEMPTS, EXPECTED_FINAL_CONTROLS, SourcePaths, compile_deterministic
from .runtime import ResourceLimits, process_tree_rss, synthetic_recovery_canary
from .schema import CAMPAIGN_ID, FAMILY_ORDER, complexity, normalize_config, schema_hash
from .validators import aggregate_materialized_probe, engine_probe, independent_replay, validate_compiled


CODE_PATHS = (
    "tools/build_stage22_core_liquid_campaign.py",
    "tools/core_liquid_campaign/__init__.py",
    "tools/core_liquid_campaign/canonical.py",
    "tools/core_liquid_campaign/schema.py",
    "tools/core_liquid_campaign/generator.py",
    "tools/core_liquid_campaign/compiler.py",
    "tools/core_liquid_campaign/controls.py",
    "tools/core_liquid_campaign/executor.py",
    "tools/core_liquid_campaign/accounting.py",
    "tools/core_liquid_campaign/selection.py",
    "tools/core_liquid_campaign/runtime.py",
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


def _code_inventory(repository_root: Path) -> dict[str, Any]:
    files = []
    for relative in CODE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    combined = sha256_file_from_records(files)
    return {"files": files, "source_tree_sha256": combined}


def sha256_file_from_records(records: Sequence[Mapping[str, Any]]) -> str:
    import hashlib
    payload = "".join(f"{record['sha256']}  {record['path']}\n" for record in sorted(records, key=lambda item: item["path"])).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _registry_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def benchmark(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _registry_rows(root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl")
    results = []
    for family in FAMILY_ORDER:
        configs = [row["config"] for row in rows if row["family_id"] == family]
        started = time.perf_counter()
        checksum = 0.0
        for config in configs:
            normalized = normalize_config(family, config)
            checksum += complexity(family, normalized)
        elapsed = time.perf_counter() - started
        results.append({
            "family": family,
            "registered_attempts": len(configs),
            "outcome_free_normalizations": len(configs),
            "elapsed_seconds": f"{elapsed:.9f}",
            "normalizations_per_second": f"{len(configs) / elapsed:.3f}" if elapsed else "inf",
            "complexity_checksum": f"{checksum:.6f}",
            "economic_outcomes_opened": "false",
        })
    limits = ResourceLimits()
    old_attempts = 3521
    scale = EXPECTED_FINAL_ATTEMPTS / old_attempts
    projection = {
        "schema": "stage22_runtime_projection_v1",
        "basis": {
            "stage21_attempts": old_attempts,
            "stage21_cold_p50_hours": 48,
            "stage21_cold_p90_hours": 120,
            "stage21_prior_A1_peak_rss_bytes": 5110345728,
            "stage20_output_bytes": 1111965563,
            "stage22_attempts": EXPECTED_FINAL_ATTEMPTS,
            "linear_attempt_scale": scale,
            "current_logical_cpus": os.cpu_count(),
            "current_process_tree_rss_bytes": process_tree_rss(),
            "maxrss_ru_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "projection": {
            "cold_p50_hours_linear_upper_planning": round(48 * scale, 1),
            "cold_p90_hours_linear_upper_planning": round(120 * scale, 1),
            "interpretation": "capacity-planning bracket only; shared caches and aggregate-first execution may reduce it, but no economic timing run occurred",
            "hard_wall_time": None,
            "checkpoint_policy": "renewable until terminal completion or a safety/no-progress stop",
        },
        "limits": {
            "workers": limits.max_workers,
            "jobs_in_flight": limits.max_jobs_in_flight,
            "aggregate_process_tree_rss_bytes": limits.max_rss_bytes,
            "campaign_output_bytes": limits.max_output_bytes,
            "minimum_free_disk_bytes": limits.minimum_free_disk_bytes,
            "heartbeat_seconds": limits.heartbeat_seconds,
            "graceful_stop_seconds": limits.graceful_stop_seconds,
            "no_progress_seconds": limits.no_progress_seconds,
        },
        "economic_outcomes_opened": False,
    }
    return results, projection


def build_candidate(paths: SourcePaths, output_root: Path, repository_root: Path, implementation_commit: str) -> dict[str, Any]:
    compiled = compile_deterministic(paths, output_root)
    validation = validate_compiled(output_root)
    replay = independent_replay(paths, output_root)
    engine = engine_probe()
    aggregate = aggregate_materialized_probe()
    if not engine["pass"] or not aggregate["pass"]:
        raise AssertionError("engine or aggregate/materialized probe failed")
    atomic_write_json(output_root / "ENGINE_PROBE_AUDIT.json", engine)
    atomic_write_json(output_root / "AGGREGATE_VS_MATERIALIZED_PROBE_AUDIT.json", aggregate)
    benchmark_rows, runtime_projection = benchmark(output_root)
    _write_csv(output_root / "ENGINE_BENCHMARK_SUMMARY.csv", ("family", "registered_attempts", "outcome_free_normalizations", "elapsed_seconds", "normalizations_per_second", "complexity_checksum", "economic_outcomes_opened"), benchmark_rows)
    atomic_write_json(output_root / "RUNTIME_PROJECTION.json", runtime_projection)
    canary_root = output_root / "SYNTHETIC_RUNTIME_CANARY_STATE"
    canary = synthetic_recovery_canary(canary_root)
    if not canary["pass"]:
        raise AssertionError("runtime canary failed")
    atomic_write_json(output_root / "SYNTHETIC_RUNTIME_CANARY.json", canary)
    code = _code_inventory(repository_root)
    atomic_write_json(output_root / "CODE_HASH_INVENTORY.json", {"schema": "stage22_code_hash_inventory_v1", "implementation_commit": implementation_commit, **code})
    candidate = {
        "schema": "stage22_review_candidate_v1",
        "campaign_id": CAMPAIGN_ID,
        "implementation_commit": implementation_commit,
        "compiler_summary": compiled,
        "validation": validation,
        "independent_deterministic_replay": replay,
        "engine_probe_pass": engine["pass"],
        "aggregate_materialized_probe_pass": aggregate["pass"],
        "runtime_canary_pass": canary["pass"],
        "source_tree_sha256": code["source_tree_sha256"],
        "strategy_registry_sha256": sha256_file(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"),
        "control_registry_sha256": sha256_file(output_root / "FINAL_CONTROL_REGISTRY.jsonl"),
        "schema_sha256": schema_hash(),
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


def finalize_packet(
    output_root: Path,
    repository_root: Path,
    implementation_commit: str,
    review_path: Path,
    inherited_manifest_path: Path,
) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or review.get("blocking_findings") not in (0, [], None):
        raise ValueError("independent review has not passed")
    validation = validate_compiled(output_root)
    candidate = json.loads((output_root / "INDEPENDENT_REVIEW_TARGET.json").read_text(encoding="utf-8"))
    if candidate["implementation_commit"] != implementation_commit:
        raise ValueError("implementation commit differs from reviewed candidate")
    inherited = json.loads(inherited_manifest_path.read_text(encoding="utf-8"))
    dependency_records = [file_record(output_root, path) for path in _dependency_files(output_root)]
    code = json.loads((output_root / "CODE_HASH_INVENTORY.json").read_text(encoding="utf-8"))
    registry = _registry_rows(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl")
    controls = _registry_rows(output_root / "FINAL_CONTROL_REGISTRY.jsonl")
    counts = Counter(row["family_id"] for row in registry)
    manifest = {
        "schema": "stage22_final_campaign_manifest_v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen_awaiting_one_exact_external_human_approval",
        "economic_run_authorized_by_manifest": False,
        "repository": {"implementation_commit": implementation_commit, "source_tree_sha256": code["source_tree_sha256"]},
        "platform": "Kraken native linear PF only",
        "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
        "protected_period": "[2026-01-01T00:00:00Z,+infinity)",
        "programme_exposure_class": "program_exposed_historical",
        "attempts": {"total_rows": len(registry), "unique_economic_addresses": len({row["canonical_economic_address_sha256"] for row in registry}), "by_family": {family: counts[family] for family in FAMILY_ORDER}},
        "controls": {"total_rows": len(controls), "unique_economic_addresses": len({row["economic_address_sha256"] for row in controls}), "conditional_on_frozen_parent_slot": True},
        "primary_hashes": {
            "family_axis_schema": sha256_file(output_root / "FAMILY_AXIS_SCHEMA.json"),
            "strategy_registry": sha256_file(output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"),
            "control_registry": sha256_file(output_root / "FINAL_CONTROL_REGISTRY.jsonl"),
            "code_inventory": sha256_file(output_root / "CODE_HASH_INVENTORY.json"),
            "selection_policy": sha256_file(output_root / "SAFE_PRUNING_POLICY.json"),
            "runtime_projection": sha256_file(output_root / "RUNTIME_PROJECTION.json"),
            "aggregate_materialized_probe": sha256_file(output_root / "AGGREGATE_VS_MATERIALIZED_PROBE_AUDIT.json"),
            "independent_review": sha256_file(review_path),
        },
        "artifact_dependencies": dependency_records,
        "inherited_economic_dependency_sha256": inherited["economic_dependency_sha256"],
        "selection_contract": {
            "plateau": "Gower radius 0.15; >=5 distinct cells; >=2 varied axes; >=60% base-positive; median base>0; median stress>-18bps; every inner vector retains empties and >=75% nonempty",
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
        "schema": "stage22_final_human_approval_request_v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "awaiting_exact_external_human_approval",
        "final_campaign_manifest_sha256": manifest_hash,
        "bindings": manifest["primary_hashes"],
        "repository_implementation_commit": implementation_commit,
        "strategy_attempt_rows": EXPECTED_FINAL_ATTEMPTS,
        "control_rows": EXPECTED_FINAL_CONTROLS,
        "statements": {
            "no_post_approval_economic_invention_remains": True,
            "executor_exists_and_passed_semantic_coverage": True,
            "registry_and_controls_exist": True,
            "synthetic_and_aggregate_materialized_probes_passed": True,
            "runtime_projection_is_measured_and_inherits_explicit_prior_measurement": True,
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
        f"- registered strategy/adjudication attempts: `{EXPECTED_FINAL_ATTEMPTS}`\n"
        f"- conditional controls: `{EXPECTED_FINAL_CONTROLS}`\n\n"
        "After exact approval, repeat every bound hash and safety gate, run secure Telegram preflight and launch under the detached supervisor. No post-approval economic translation or selection choice is permitted.\n"
    )
    atomic_write_bytes(output_root / "FINAL_LAUNCH_TASK.md", launch_task.encode("utf-8"))
    inventory_files = sorted(path for path in output_root.rglob("*") if path.is_file() and path.name not in {"FINAL_PACKET_HASH_INVENTORY.json", "ARTIFACT_MANIFEST.json"})
    inventory = {
        "schema": "stage22_final_packet_hash_inventory_v1",
        "campaign_id": CAMPAIGN_ID,
        "artifacts": [file_record(output_root, path) for path in inventory_files],
        "final_campaign_manifest_sha256": manifest_hash,
        "final_human_approval_request_sha256": approval_hash,
        "status": "pass",
    }
    atomic_write_json(output_root / "FINAL_PACKET_HASH_INVENTORY.json", inventory)
    artifact_files = sorted(path for path in output_root.rglob("*") if path.is_file() and path.name != "ARTIFACT_MANIFEST.json")
    atomic_write_json(output_root / "ARTIFACT_MANIFEST.json", {
        "schema": "stage22_artifact_manifest_v1",
        "campaign_id": CAMPAIGN_ID,
        "artifact_count": len(artifact_files),
        "artifacts": [file_record(output_root, path) for path in artifact_files],
        "status": "pass",
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    })
    return {"status": "pass", "final_manifest_sha256": manifest_hash, "final_approval_request_sha256": approval_hash, "validation": validation}
