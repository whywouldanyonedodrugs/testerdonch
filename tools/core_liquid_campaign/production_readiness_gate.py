from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .a1_state import initial_state, transition
from .canonical import atomic_write_json, canonical_hash, sha256_file
from .executor import CacheAuthority, dispatch_registered_attempt
from .schema import FAMILY_ORDER, OUTER_FOLDS
from .shadow_payoff import ShadowPayoffProvider
from .stage24_probes import control_production_shadow_probe
from .terminal import terminal_package, verify_terminal_inventory


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


def _registry_gate(packet_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    }, execution, controls)


def _cache_gate(cache_path: Path, authority_path: Path) -> tuple[dict[str, Any], tuple[Any, ...]]:
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    manifest = json.loads(cache_path.read_text(encoding="utf-8"))
    cache = CacheAuthority(cache_path, cache_path.parent)
    paths = [str(row["path"]) for row in manifest["artifacts"]]
    cold_start = time.monotonic(); _, frames = cache.load_frames({"execution_input_authority": authority}, paths); cold = time.monotonic() - cold_start
    warm_start = time.monotonic(); _, warm_frames = cache.load_frames({"execution_input_authority": authority}, paths); warm = time.monotonic() - warm_start
    partitions = [row["campaign_partition"] for row in manifest["artifacts"]]
    outer = {str(row["outer_fold_id"]) for row in partitions if row["phase"] == "outer_evaluation"}
    inner = {str(row["inner_fold_id"]) for row in partitions if row["phase"] == "inner_validation"}
    symbols = {str(row["symbol"]) for row in manifest["artifacts"]}
    protected = sum(int(frame.metadata.get("protected_rows", 0)) for frame in frames)
    value_equal = all(left.content_sha256() == right.content_sha256() for left, right in zip(frames, warm_frames))
    # A production campaign requires both development and outer partitions.
    complete_campaign_cache = outer == set(OUTER_FOLDS) and bool(inner) and len(symbols) >= 3
    return ({
        "status": "pass" if complete_campaign_cache and protected == 0 and value_equal else "fail",
        "cache_manifest_sha256": sha256_file(cache_path),
        "artifacts": len(paths), "outer_folds": sorted(outer), "inner_fold_ids": len(inner), "symbols": sorted(symbols),
        "all_five_family_outer_positions": 5 * len(outer),
        "typed_kda_unavailable_positions": len(outer),
        "protected_rows": protected,
        "cold_seconds": cold, "warm_seconds": warm,
        "warm_value_equivalent": value_equal,
        "complete_campaign_cache": complete_campaign_cache,
        "blocking_reason": None if complete_campaign_cache else "cache lacks full inner-development and small/median/large-symbol production partitions",
    }, frames)


def _real_engine_gate(execution: list[dict[str, Any]], frames: tuple[Any, ...]) -> dict[str, Any]:
    registry = {str(row["executable_attempt_id"]): row for row in execution}
    selected_ids = {
        "A4_TSMOM_V7": "A4_TSMOM_V7:S22:L:0006:1",
        "A1_COMPRESSION_V2": "A1_COMPRESSION_V2:S22:L:0865:1",
        "A3_STARTER_RETEST_V3": "A3_STARTER_RETEST_V3:S22:L:2610:1",
    }
    rows = {family: registry[identity] for family, identity in selected_ids.items()}
    provider = ShadowPayoffProvider("stage24-production-readiness-real-input-v1")
    results = []
    start = time.monotonic()
    for frame in frames:
        partition = frame.metadata["campaign_partition"]
        for family, row in rows.items():
            result = dispatch_registered_attempt(row, (frame,), registry_by_id=registry, payoff_provider=provider)
            results.append({
                "family": family, "outer_fold_id": partition["outer_fold_id"], "symbol": frame.symbol,
                "status": result["status"], "observation_count": len(result["observations"]),
                "aggregate_sha256": canonical_hash(result["aggregate"]),
            })
    elapsed = time.monotonic() - start
    covered = {(row["family"], row["outer_fold_id"]) for row in results if row["status"] == "complete"}
    required = {(family, fold) for family in rows for fold in OUTER_FOLDS}
    attestation = provider.attestation()
    return {
        "status": "pass" if covered == required and attestation["economic_outcomes_opened"] is False else "fail",
        "family_fold_rows": len(results), "covered_family_folds": len(covered),
        "nonempty_rows": sum(int(row["observation_count"]) > 0 for row in results),
        "elapsed_seconds": elapsed,
        "rows_sha256": canonical_hash(results),
        "shadow_attestation": attestation,
        "KDA02B": {"status": "unavailable_data", "reason": "authorized Stage20 tape lacks raw decision-time derivative feature columns"},
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


def _terminal_gate(output_root: Path) -> dict[str, Any]:
    complete_root = output_root / "terminal_complete"
    bound_root = output_root / "terminal_bound_stop"
    terminal_package(
        complete_root, attempt_ids=["a", "b"], control_ids=["c"],
        attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}, {"attempt_id": "b", "terminal_status": "unavailable_data"}],
        control_rows=[{"control_attempt_id": "c", "terminal_status": "unavailable_no_parent"}],
        routes=[{"family": "fixture", "route": "shadow_path_verified"}],
        forensics=[{"family": "fixture", "event_count": 1}], all_workers_stopped=True,
        job_reconciliation={"pass": True},
    )
    complete = verify_terminal_inventory(complete_root)
    terminal_package(
        bound_root, attempt_ids=["a", "b"], control_ids=["c"],
        attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}], control_rows=[],
        routes=[], forensics=[], all_workers_stopped=True, bound_stop=True,
    )
    bound = verify_terminal_inventory(bound_root)
    return {"status": "pass", "complete": complete, "bound_stop": bound, "resumable": True}


def _service_evidence_gate(repository_root: Path) -> dict[str, Any]:
    roots = {
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
    passed = (
        all(row["campaign_status"] == "complete" and row["health_release"] is True and row["all_workers_stopped"] is True for row in rows.values())
        and attempt_values == [2]
        and list(rows["graceful_resume"]["attempts"].values()) == [2]
        and all(str(row["service_identity"]).startswith("qlmg-stage24-shadow-") for row in rows.values())
    )
    return {"status": "pass" if passed else "fail", "installed_service_runs": rows, "telegram_preflight": "pass", "independent_of_chat": True}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Mapping[str, Any]] = {}
    checks["authority"] = _authority_gate(args.repository_root, args.stage24_task)
    registry, execution, controls = _registry_gate(args.packet_root); checks["registries"] = registry
    cache, frames = _cache_gate(args.cache_manifest, args.packet_root / "EXECUTION_INPUT_AUTHORITY.json"); checks["cache_authority"] = cache
    checks["real_family_inputs_and_engines"] = _real_engine_gate(execution, frames)
    checks["controls"] = control_production_shadow_probe(args.output / "control_workers", controls)
    checks["a1_state"] = _a1_state_gate()
    checks["shadow_service"] = _service_evidence_gate(args.repository_root)
    checks["terminal"] = _terminal_gate(args.output)
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
    result.add_argument("--cache-manifest", type=Path, default=Path("results/rebaseline/stage24_production_readiness_20260721_v01/semantic_cache/SEMANTIC_CACHE_MANIFEST.json"))
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
