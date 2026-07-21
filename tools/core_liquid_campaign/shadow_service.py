from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .executor import CacheAuthority, dispatch_registered_attempt
from .family_engines.common import require_utc
from .runtime import LazySupervisor, ResourceLimits
from .shadow_payoff import ShadowPayoffProvider
from .terminal import forensic_summary, terminal_package, verify_terminal_inventory


class ShadowAuthorizationError(PermissionError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return require_utc(value).isoformat()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class ShadowAuthorization:
    """Exact Stage-24 no-outcome authorization; never authorizes economic payoff reads."""

    def __init__(self, spec_path: Path) -> None:
        self.spec_path = spec_path

    @staticmethod
    def _bound_file(record: Mapping[str, Any], *, repository_root: Path) -> Path:
        raw = Path(str(record.get("path", "")))
        path = raw if raw.is_absolute() else repository_root / raw
        if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise ShadowAuthorizationError(f"shadow authority file mismatch: {record.get('role')}")
        return path

    def require(self) -> dict[str, Any]:
        if not self.spec_path.is_file():
            raise ShadowAuthorizationError("shadow service specification is absent")
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        if spec.get("schema") != "stage24_shadow_service_spec_v1" or spec.get("mode") != "shadow_no_outcome":
            raise ShadowAuthorizationError("shadow service mode/schema differs")
        if spec.get("economic_outcomes_authorized") is not False or spec.get("protected_outcomes_authorized") is not False or spec.get("capitalcom_payload_access") is not False:
            raise ShadowAuthorizationError("shadow service specification broadens outcome authority")
        repository_root = Path(str(spec["repository_root"]))
        authority_record = spec.get("stage24_task")
        if not isinstance(authority_record, Mapping):
            raise ShadowAuthorizationError("Stage 24 task binding is absent")
        self._bound_file(authority_record, repository_root=repository_root)
        if authority_record.get("sha256") != "9e546e9376408f97bc3bc2ef2862c06e746864eacbd9a5a70f0e71680eeeccdf":
            raise ShadowAuthorizationError("Stage 24 task hash differs")
        for record in spec.get("bound_files", ()):
            self._bound_file(record, repository_root=repository_root)
        actual_commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reviewed_commit = str(spec.get("reviewed_commit", ""))
        if subprocess.run(
            ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", reviewed_commit, actual_commit],
            check=False,
        ).returncode != 0:
            raise ShadowAuthorizationError("live worktree is not the reviewed commit or its descendant")
        row = spec.get("registered_attempt")
        if not isinstance(row, Mapping) or canonical_hash(row) != spec.get("registered_attempt_sha256"):
            raise ShadowAuthorizationError("shadow registered attempt binding differs")
        return spec


def run_shadow_service(spec_path: Path) -> dict[str, Any]:
    spec = ShadowAuthorization(spec_path).require()
    from tools.run_stage22_core_liquid_campaign import TelegramTransport

    telegram = TelegramTransport()
    if telegram.preflight() is not True:
        raise ShadowAuthorizationError("secure Telegram preflight failed")
    run_root = Path(str(spec["run_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "SHADOW_CAMPAIGN_STATE.json"
    state = {
        "schema": "stage24_shadow_campaign_state_v1",
        "status": "running",
        "service_identity": spec["service_identity"],
        "identity_bindings_sha256": canonical_hash(spec["identity_bindings"]),
        "economic_outcomes_opened": False,
    }
    atomic_write_json(state_path, state)
    authority = json.loads(Path(str(spec["execution_input_authority_path"])).read_text(encoding="utf-8"))
    cache_path = Path(str(spec["cache_manifest_path"]))
    cache = CacheAuthority(cache_path, cache_path.parent)
    artifact_path = str(spec["cache_artifact_path"])
    cache.load_frames({"execution_input_authority": authority}, [artifact_path])
    row = dict(spec["registered_attempt"])
    registry = {str(row["executable_attempt_id"]): row}
    job_id = str(spec["registered_job_id"])

    def job() -> dict[str, Any]:
        _, frames = cache.load_frames({"execution_input_authority": authority}, [artifact_path])
        worker_hold = float(spec.get("worker_hold_seconds", 0.0))
        if worker_hold > 0:
            time.sleep(worker_hold)
        provider = ShadowPayoffProvider(str(spec["synthetic_provider_version"]))
        result = dispatch_registered_attempt(
            row,
            frames,
            registry_by_id=registry,
            payoff_provider=provider,
        )
        return {
            **_jsonable(result),
            "registered_job_id": job_id,
            "registered_attempt_id": row["executable_attempt_id"],
            "cache_manifest_sha256": sha256_file(cache_path),
            "shadow_attestation": provider.attestation(),
        }

    bindings = dict(spec["identity_bindings"])

    def validator(candidate_job_id: str, result: Any) -> bool:
        if candidate_job_id != job_id or not isinstance(result, Mapping):
            return False
        attestation = result.get("shadow_attestation", {})
        if result.get("status") != "complete" or result.get("registered_attempt_id") != row["executable_attempt_id"]:
            return False
        if not result.get("observations") or result.get("cache_manifest_sha256") != sha256_file(cache_path):
            return False
        if attestation.get("economic_outcomes_opened") is not False or attestation.get("real_post_entry_rows_opened") != 0:
            return False
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        return persisted.get("status") == "running" and persisted.get("identity_bindings_sha256") == canonical_hash(bindings)

    limits = ResourceLimits(
        max_workers=int(spec["workers"]),
        max_jobs_in_flight=int(spec["workers"]),
        max_rss_bytes=10 * 1024**3,
        max_output_bytes=24 * 1024**3,
        minimum_free_disk_bytes=8 * 1024**3,
        heartbeat_seconds=int(spec.get("heartbeat_seconds", 1800)),
        graceful_stop_seconds=300,
        wall_time_seconds=None,
    )
    supervisor_root = run_root / "production_shadow_unit"
    supervisor = LazySupervisor(
        supervisor_root,
        limits,
        heartbeat=telegram.heartbeat,
        real_unit_validator=validator,
        identity_bindings=bindings,
    )
    supervisor_state = supervisor.run(iter([(job_id, job)]), require_health_release=True)
    if supervisor_state.get("status") != "complete" or supervisor_state.get("health_release") is not True:
        state.update({
            "status": str(supervisor_state.get("status", "global_resumable_bound_stop")),
            "health_release": False,
            "all_workers_stopped": supervisor_state.get("all_workers_stopped") is True,
            "resumable": True,
        })
        atomic_write_json(state_path, state)
        return state
    marker = next((supervisor_root / "markers").glob("*.json"))
    marker_record = json.loads(marker.read_text(encoding="utf-8"))
    artifact = supervisor_root / marker_record["artifact"]
    result = json.loads(artifact.read_text(encoding="utf-8"))["result"]
    observations = list(result.get("observations", ()))
    terminal_root = run_root / "terminal"
    if (terminal_root / "TERMINAL_ARTIFACT_INVENTORY.json").is_file():
        terminal_verification = verify_terminal_inventory(terminal_root)
    else:
        terminal_package(
            terminal_root,
            attempt_ids=[str(row["executable_attempt_id"])],
            control_ids=[],
            attempt_rows=[{"attempt_id": row["executable_attempt_id"], "terminal_status": "completed"}],
            control_rows=[],
            routes=[{"family": row["family_id"], "route": "shadow_path_verified_no_economic_claim"}],
            forensics=[{"family": row["family_id"], **forensic_summary(observations), "shadow_only": True}],
            all_workers_stopped=True,
            job_reconciliation={"pass": True, "expected": 1, "observed": 1},
        )
        terminal_verification = verify_terminal_inventory(terminal_root)
    state.update({
        "status": "complete",
        "first_real_registered_unit_reconciled": True,
        "scheduled_heartbeat_delivered": int(supervisor_state.get("heartbeat_success_count", 0)) >= 1,
        "health_release": True,
        "terminal_inventory": terminal_verification,
    })
    atomic_write_json(state_path, state)
    hold = float(spec.get("hold_after_health_seconds", 0.0))
    if hold > 0:
        time.sleep(hold)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the exact Stage 24 production shadow service")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_shadow_service(args.spec), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ShadowAuthorization", "ShadowAuthorizationError", "run_shadow_service"]
