from __future__ import annotations

import errno
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file


class ResourceGateError(RuntimeError):
    pass


def _default_pid_list() -> list[int]:
    return sorted(int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit())


def _default_status_reader(pid: int) -> str:
    return Path(f"/proc/{pid}/status").read_text(encoding="utf-8")


def _status_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields


def _status_identity(text: str) -> tuple[int, int]:
    fields = _status_fields(text)
    try:
        return int(fields["Pid"].split()[0]), int(fields["PPid"].split()[0])
    except (KeyError, ValueError, IndexError) as exc:
        raise ResourceGateError("unparseable /proc PID/PPID status") from exc


def _status_rss(text: str) -> int:
    fields = _status_fields(text)
    try:
        return int(fields["VmRSS"].split()[0]) * 1024
    except (KeyError, ValueError, IndexError) as exc:
        raise ResourceGateError("sampled tree process has no parseable VmRSS") from exc


def _status_values(text: str) -> tuple[int, int, int]:
    pid, ppid = _status_identity(text)
    return pid, ppid, _status_rss(text)


def _vanished(exc: BaseException) -> bool:
    return isinstance(exc, (FileNotFoundError, ProcessLookupError)) or (isinstance(exc, OSError) and exc.errno == errno.ESRCH)


def process_tree_rss(root_pid: int | None = None, *, pid_list: Callable[[], Sequence[int]] = _default_pid_list, status_reader: Callable[[int], str] = _default_status_reader) -> int:
    root = os.getpid() if root_pid is None else int(root_pid)
    try:
        root_status = _status_values(status_reader(root))
    except BaseException as exc:
        if _vanished(exc):
            raise ResourceGateError("supervisor/root process cannot be sampled") from exc
        raise
    if root_status[0] != root:
        raise ResourceGateError("supervisor/root PID mismatch")
    identities: dict[int, tuple[int, int]] = {root: (root_status[0], root_status[1])}
    status_text: dict[int, str] = {}
    for pid in pid_list():
        if pid == root:
            continue
        try:
            text = status_reader(int(pid)); parsed = _status_identity(text)
        except BaseException as exc:
            if _vanished(exc):
                continue
            raise
        identities[parsed[0]] = parsed; status_text[parsed[0]] = text
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for pid, (_, ppid) in identities.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid); changed = True
    total = root_status[2] + sum(_status_rss(status_text[pid]) for pid in descendants - {root} if pid in status_text)
    if total <= 0:
        raise ResourceGateError("aggregate process-tree RSS is not positive")
    return total


def physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


@dataclass(frozen=True)
class ResourceLimits:
    max_workers: int = 4
    max_jobs_in_flight: int = 4
    max_rss_bytes: int = int(physical_memory_bytes() * 0.75)
    max_output_bytes: int = 24 * 1024**3
    minimum_free_disk_bytes: int = 8 * 1024**3
    minimum_free_disk_fraction: float = 0.10
    heartbeat_seconds: int = 1800
    graceful_stop_seconds: int = 300
    no_progress_seconds: int = 7200
    retry_delays_seconds: tuple[int, ...] = (60, 300, 900)
    maximum_supervisor_restarts: int = 3
    monitor_interval_seconds: float = 1.0
    wall_time_seconds: int | None = None

    def validate(self) -> None:
        if not 1 <= self.max_workers <= min(4, os.cpu_count() or 1) or not 1 <= self.max_jobs_in_flight <= 4:
            raise ResourceGateError("workers/jobs-in-flight exceed the frozen CPU boundary")
        if self.max_jobs_in_flight < self.max_workers:
            raise ResourceGateError("jobs-in-flight cannot be lower than workers")
        if self.max_rss_bytes > int(physical_memory_bytes() * 0.75):
            raise ResourceGateError("RSS limit exceeds 75% physical memory")
        if self.retry_delays_seconds != (60, 300, 900):
            raise ResourceGateError("worker retry delays differ from the frozen contract")
        if self.maximum_supervisor_restarts != 3 or self.no_progress_seconds != 7200:
            raise ResourceGateError("stall/restart contract differs from the frozen contract")
        if self.wall_time_seconds is not None:
            raise ResourceGateError("Stage 22 runtime uses renewable checkpoints, not a hard wall stop")


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def resource_preflight(output_root: Path, limits: ResourceLimits, *, rss_sampler: Callable[[], int] = process_tree_rss) -> dict[str, int]:
    limits.validate(); output_root.mkdir(parents=True, exist_ok=True)
    rss = rss_sampler(); output = directory_size(output_root); usage = shutil.disk_usage(output_root)
    required_free = max(limits.minimum_free_disk_bytes, int(usage.total * limits.minimum_free_disk_fraction))
    if rss > limits.max_rss_bytes:
        raise ResourceGateError("RSS limit exceeded")
    if output > limits.max_output_bytes:
        raise ResourceGateError("output limit exceeded")
    if usage.free < required_free:
        raise ResourceGateError("minimum free disk gate failed")
    return {"rss_bytes": rss, "output_bytes": output, "free_disk_bytes": usage.free, "required_free_disk_bytes": required_free}


class LazySupervisor:
    """Bounded generation supervisor with atomic markers and idempotent replay."""

    def __init__(
        self,
        run_root: Path,
        limits: ResourceLimits,
        *,
        executor_factory: Callable[[int], Executor] = lambda workers: ThreadPoolExecutor(max_workers=workers),
        heartbeat: Callable[[Mapping[str, Any]], None] | None = None,
        rss_sampler: Callable[[], int] = process_tree_rss,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_root = run_root; self.limits = limits; self.executor_factory = executor_factory
        self.heartbeat = heartbeat or (lambda _: None); self.rss_sampler = rss_sampler; self.monotonic = monotonic
        self.state_path = run_root / "SUPERVISOR_STATE.json"
        self.marker_root = run_root / "markers"; self.artifact_root = run_root / "artifacts"

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema": "stage22_supervisor_state_v2", "completed": {}, "failed": {}, "attempts": {}, "generation": 0, "status": "new", "heartbeat_count": 0, "first_real_unit_reconciled": False, "health_release": False, "consecutive_supervisor_restarts": 0}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save(self, state: Mapping[str, Any]) -> None:
        updated = dict(state); updated["generation"] = int(updated.get("generation", 0)) + 1
        if isinstance(state, dict):
            state["generation"] = updated["generation"]
        atomic_write_json(self.state_path, updated)

    def _paths(self, job_id: str) -> tuple[Path, Path]:
        stem = canonical_hash({"job_id": job_id})
        return self.artifact_root / f"{stem}.json", self.marker_root / f"{stem}.json"

    def _commit_result(self, job_id: str, result: Any) -> dict[str, Any]:
        artifact, marker = self._paths(job_id)
        atomic_write_json(artifact, {"job_id": job_id, "result": result})
        record = {"job_id": job_id, "artifact": artifact.relative_to(self.run_root).as_posix(), "artifact_sha256": sha256_file(artifact), "status": "complete"}
        atomic_write_json(marker, record)
        return record

    def _reconcile(self, job_ids: set[str]) -> dict[str, Any]:
        completed: dict[str, Any] = {}
        if not self.marker_root.exists():
            return completed
        for marker in sorted(self.marker_root.glob("*.json")):
            record = json.loads(marker.read_text(encoding="utf-8")); job_id = str(record.get("job_id"))
            if job_id not in job_ids:
                raise ResourceGateError("marker belongs to an unregistered job")
            artifact = self.run_root / str(record["artifact"])
            if not artifact.is_file() or sha256_file(artifact) != record.get("artifact_sha256"):
                raise ResourceGateError("committed marker/artifact reconciliation failed")
            completed[job_id] = record
        return completed

    def _graceful_shutdown(self, executor: Executor, in_flight: Mapping[Future[Any], tuple[str, Callable[[], Any]]], state: dict[str, Any], status: str) -> dict[str, Any]:
        for future in in_flight:
            future.cancel()
        deadline = self.monotonic() + self.limits.graceful_stop_seconds
        while any(not future.done() for future in in_flight) and self.monotonic() < deadline:
            wait(tuple(in_flight), timeout=min(self.limits.monitor_interval_seconds, 0.05), return_when=FIRST_COMPLETED)
        state["status"] = status if all(future.done() for future in in_flight) else "forced_after_grace_timeout"
        self._save(state)
        executor.shutdown(wait=False, cancel_futures=True)
        return state

    def run(self, jobs: Iterable[tuple[str, Callable[[], Any]]], *, stop_after_completions: int | None = None) -> dict[str, Any]:
        self.limits.validate()
        job_list = list(jobs)
        if len({job_id for job_id, _ in job_list}) != len(job_list):
            raise ResourceGateError("duplicate registered job ID")
        callable_by_id = dict(job_list); job_ids = set(callable_by_id)
        state = self._load_state(); completed = self._reconcile(job_ids)
        failed: dict[str, Any] = dict(state.get("failed", {})); attempts = {str(key): int(value) for key, value in state.get("attempts", {}).items()}
        pending = [job_id for job_id, _ in job_list if job_id not in completed]; retry_ready: dict[str, float] = {}
        in_flight: dict[Future[Any], tuple[str, Callable[[], Any]]] = {}
        now = self.monotonic(); last_progress = now; last_heartbeat = now
        state.update({"status": "running", "limits": asdict(self.limits), "completed": completed, "failed": failed, "attempts": attempts})
        self._save(state)
        executor = self.executor_factory(self.limits.max_workers)
        try:
            while in_flight or pending or retry_ready:
                resource_preflight(self.run_root, self.limits, rss_sampler=self.rss_sampler)
                now = self.monotonic()
                for job_id in sorted([key for key, ready in retry_ready.items() if ready <= now]):
                    pending.append(job_id); del retry_ready[job_id]
                while pending and len(in_flight) < self.limits.max_jobs_in_flight:
                    resource_preflight(self.run_root, self.limits, rss_sampler=self.rss_sampler)
                    job_id = pending.pop(0)
                    if job_id in completed:
                        continue
                    attempts[job_id] = attempts.get(job_id, 0) + 1
                    in_flight[executor.submit(callable_by_id[job_id])] = (job_id, callable_by_id[job_id])
                done, _ = wait(tuple(in_flight), timeout=self.limits.monitor_interval_seconds, return_when=FIRST_COMPLETED) if in_flight else (set(), set())
                for future in done:
                    job_id, _ = in_flight.pop(future)
                    try:
                        completed[job_id] = self._commit_result(job_id, future.result()); failed.pop(job_id, None)
                    except Exception as exc:
                        if attempts[job_id] <= len(self.limits.retry_delays_seconds):
                            retry_ready[job_id] = self.monotonic() + self.limits.retry_delays_seconds[attempts[job_id] - 1]
                        else:
                            failed[job_id] = {"error_type": type(exc).__name__, "error": str(exc), "attempts": attempts[job_id], "status": "family_job_exhausted"}
                    last_progress = self.monotonic()
                    state.update({"completed": completed, "failed": failed, "attempts": attempts})
                    self._save(state)
                    if stop_after_completions is not None and len(completed) >= stop_after_completions:
                        return self._graceful_shutdown(executor, in_flight, state, "graceful_bound_stop")
                now = self.monotonic()
                if now - last_heartbeat >= self.limits.heartbeat_seconds:
                    payload = {"completed": len(completed), "failed": len(failed), "in_flight": len(in_flight), "generation": state.get("generation")}
                    self.heartbeat(payload); state["heartbeat_count"] = int(state.get("heartbeat_count", 0)) + 1; last_heartbeat = now
                    self._save(state)
                state["first_real_unit_reconciled"] = bool(completed)
                state["health_release"] = bool(completed) and int(state.get("heartbeat_count", 0)) >= 1
                if now - last_progress > self.limits.no_progress_seconds and (pending or retry_ready or in_flight):
                    restarts = int(state.get("consecutive_supervisor_restarts", 0)) + 1
                    state["consecutive_supervisor_restarts"] = restarts
                    status = "restart_required_from_atomic_generation" if restarts <= self.limits.maximum_supervisor_restarts else "global_resumable_bound_stop_stalled"
                    return self._graceful_shutdown(executor, in_flight, state, status)
            state.update({"completed": completed, "failed": failed, "attempts": attempts, "status": "complete" if not failed else "complete_with_family_job_exhaustion", "first_real_unit_reconciled": bool(completed), "health_release": bool(completed) and int(state.get("heartbeat_count", 0)) >= 1})
            self._save(state); executor.shutdown(wait=True)
            return state
        except Exception:
            self._graceful_shutdown(executor, in_flight, state, "resource_or_common_failure_bound_stop")
            raise


def detached_service_spec(repository_root: Path, run_root: Path, manifest_sha256: str, workers: int) -> dict[str, Any]:
    if workers > 4 or not all(character in "0123456789abcdef" for character in manifest_sha256) or len(manifest_sha256) != 64:
        raise ResourceGateError("invalid detached service binding")
    service_id = f"qlmg-stage22-{manifest_sha256[:12]}"
    return {
        "service_id": service_id,
        "working_directory": str(repository_root),
        "run_root": str(run_root),
        "workers": workers,
        "restart": "on-failure",
        "restart_limit": 3,
        "independent_of_chat": True,
        "environment": {"PYTHONPATH": None, "invocation": "installed package or repository-root python -m only"},
    }


def synthetic_recovery_canary(root: Path) -> dict[str, Any]:
    class Clock:
        value = 0.0
        def __call__(self) -> float:
            self.value += 1000.0
            return self.value
    clock = Clock(); heartbeats: list[Mapping[str, Any]] = []; calls = {"flaky": 0}
    def flaky() -> int:
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise RuntimeError("synthetic worker death")
        return 7
    jobs = [("stable-1", lambda: 1), ("flaky", flaky), ("stable-2", lambda: 2)]
    limits = ResourceLimits(max_workers=2, max_jobs_in_flight=2, max_output_bytes=1024**3, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0, monitor_interval_seconds=0.001)
    first = LazySupervisor(root, limits, heartbeat=heartbeats.append, monotonic=clock).run(jobs, stop_after_completions=1)
    resumed = LazySupervisor(root, limits, heartbeat=heartbeats.append, monotonic=clock).run(jobs)
    replay = LazySupervisor(root, limits, heartbeat=heartbeats.append, monotonic=clock).run(jobs)
    markers = list((root / "markers").glob("*.json")); artifacts = list((root / "artifacts").glob("*.json"))
    excursion_root = root / "resource_excursion"
    rss_calls = {"count": 0}
    def excursion_rss() -> int:
        rss_calls["count"] += 1
        return limits.max_rss_bytes + 1 if rss_calls["count"] == 3 else 1
    excursion_stopped = False
    try:
        LazySupervisor(excursion_root, limits, rss_sampler=excursion_rss).run([("slow", lambda: (time.sleep(0.02), 3)[1])])
    except ResourceGateError:
        excursion_stopped = True
    excursion_resumed = LazySupervisor(excursion_root, limits, rss_sampler=lambda: 1).run([("slow", lambda: 3)])
    return {
        "first_status": first["status"],
        "resumed_status": resumed["status"],
        "replay_status": replay["status"],
        "completed": len(replay["completed"]),
        "markers_reconciled": len(markers),
        "artifacts_reconciled": len(artifacts),
        "worker_death_retried": calls["flaky"] == 2,
        "heartbeat_delivered": bool(heartbeats),
        "health_release": bool(replay.get("health_release")),
        "continuous_resource_excursion_stopped": excursion_stopped,
        "resource_excursion_idempotently_resumed": excursion_resumed["status"] == "complete",
        "idempotent": resumed["completed"] == replay["completed"],
        "pass": first["status"] == "graceful_bound_stop" and resumed["status"] == "complete" and replay["status"] == "complete" and len(replay["completed"]) == 3 and len(markers) == 3 and len(artifacts) == 3 and calls["flaky"] == 2 and bool(heartbeats) and bool(replay.get("health_release")) and excursion_stopped and excursion_resumed["status"] == "complete",
    }


__all__ = ["LazySupervisor", "ResourceGateError", "ResourceLimits", "detached_service_spec", "directory_size", "physical_memory_bytes", "process_tree_rss", "resource_preflight", "synthetic_recovery_canary"]
