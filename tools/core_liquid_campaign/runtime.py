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

from .canonical import atomic_write_json


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


def process_tree_rss(
    root_pid: int | None = None,
    *,
    pid_list: Callable[[], Sequence[int]] = _default_pid_list,
    status_reader: Callable[[int], str] = _default_status_reader,
) -> int:
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
            text = status_reader(int(pid))
            parsed = _status_identity(text)
        except BaseException as exc:
            if _vanished(exc):
                continue
            raise
        identities[parsed[0]] = parsed
        status_text[parsed[0]] = text
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for pid, (_, ppid) in identities.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    total = root_status[2]
    for pid in descendants - {root}:
        if pid in status_text:
            total += _status_rss(status_text[pid])
    if total <= 0:
        raise ResourceGateError("aggregate process-tree RSS is not positive")
    return total


@dataclass(frozen=True)
class ResourceLimits:
    max_workers: int = 4
    max_jobs_in_flight: int = 4
    max_rss_bytes: int = 10 * 1024**3
    max_output_bytes: int = 24 * 1024**3
    minimum_free_disk_bytes: int = 8 * 1024**3
    heartbeat_seconds: int = 1800
    graceful_stop_seconds: int = 300
    no_progress_seconds: int = 3600
    wall_time_seconds: int | None = None

    def validate(self) -> None:
        if not 1 <= self.max_workers <= 4 or not 1 <= self.max_jobs_in_flight <= 4:
            raise ResourceGateError("workers/jobs-in-flight must be within [1,4]")
        if self.max_jobs_in_flight < self.max_workers:
            raise ResourceGateError("jobs-in-flight cannot be lower than workers")
        if self.wall_time_seconds is not None:
            raise ResourceGateError("Stage 22 runtime uses renewable checkpoints, not a hard wall stop")


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def resource_preflight(output_root: Path, limits: ResourceLimits, *, rss_sampler: Callable[[], int] = process_tree_rss) -> dict[str, int]:
    limits.validate()
    output_root.mkdir(parents=True, exist_ok=True)
    rss = rss_sampler()
    output = directory_size(output_root)
    free = shutil.disk_usage(output_root).free
    if rss > limits.max_rss_bytes:
        raise ResourceGateError("RSS limit exceeded")
    if output > limits.max_output_bytes:
        raise ResourceGateError("output limit exceeded")
    if free < limits.minimum_free_disk_bytes:
        raise ResourceGateError("minimum free disk gate failed")
    return {"rss_bytes": rss, "output_bytes": output, "free_disk_bytes": free}


class LazySupervisor:
    def __init__(
        self,
        run_root: Path,
        limits: ResourceLimits,
        *,
        executor_factory: Callable[[int], Executor] = lambda workers: ThreadPoolExecutor(max_workers=workers),
        heartbeat: Callable[[Mapping[str, Any]], None] | None = None,
        rss_sampler: Callable[[], int] = process_tree_rss,
    ) -> None:
        self.run_root = run_root
        self.limits = limits
        self.executor_factory = executor_factory
        self.heartbeat = heartbeat or (lambda _: None)
        self.rss_sampler = rss_sampler
        self.state_path = run_root / "SUPERVISOR_STATE.json"

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema": "stage22_supervisor_state_v1", "completed": {}, "failed": {}, "status": "new"}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save(self, state: Mapping[str, Any]) -> None:
        atomic_write_json(self.state_path, dict(state))

    def run(self, jobs: Iterable[tuple[str, Callable[[], Any]]], *, stop_after_completions: int | None = None) -> dict[str, Any]:
        state = self._load_state()
        completed: dict[str, Any] = dict(state.get("completed", {}))
        failed: dict[str, Any] = dict(state.get("failed", {}))
        iterator = iter(jobs)
        in_flight: dict[Future[Any], tuple[str, Callable[[], Any]]] = {}
        attempts: dict[str, int] = {str(key): int(value) for key, value in state.get("attempts", {}).items()}
        last_progress = time.monotonic()
        last_heartbeat = time.monotonic()
        state.update({"status": "running", "limits": asdict(self.limits)})
        self._save(state)
        with self.executor_factory(self.limits.max_workers) as executor:
            exhausted = False
            while in_flight or not exhausted:
                while not exhausted and len(in_flight) < self.limits.max_jobs_in_flight:
                    resource_preflight(self.run_root, self.limits, rss_sampler=self.rss_sampler)
                    try:
                        job_id, callable_job = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    if job_id in completed:
                        continue
                    if job_id in {value[0] for value in in_flight.values()}:
                        raise ResourceGateError(f"duplicate in-flight job: {job_id}")
                    attempts[job_id] = attempts.get(job_id, 0) + 1
                    in_flight[executor.submit(callable_job)] = (job_id, callable_job)
                if not in_flight:
                    continue
                done, _ = wait(in_flight, timeout=0.1, return_when=FIRST_COMPLETED)
                if not done:
                    if time.monotonic() - last_progress > self.limits.no_progress_seconds:
                        raise ResourceGateError("no-progress stop")
                for future in done:
                    job_id, callable_job = in_flight.pop(future)
                    try:
                        completed[job_id] = future.result()
                        failed.pop(job_id, None)
                    except Exception as exc:
                        if attempts[job_id] < 2:
                            resource_preflight(self.run_root, self.limits, rss_sampler=self.rss_sampler)
                            attempts[job_id] += 1
                            in_flight[executor.submit(callable_job)] = (job_id, callable_job)
                        else:
                            failed[job_id] = {"error_type": type(exc).__name__, "error": str(exc), "attempts": attempts[job_id]}
                    last_progress = time.monotonic()
                    state.update({"completed": completed, "failed": failed, "attempts": attempts, "status": "running"})
                    self._save(state)
                    if stop_after_completions is not None and len(completed) >= stop_after_completions:
                        state["status"] = "graceful_bound_stop"
                        self._save(state)
                        for pending in in_flight:
                            pending.cancel()
                        return state
                if time.monotonic() - last_heartbeat >= self.limits.heartbeat_seconds:
                    self.heartbeat({"completed": len(completed), "failed": len(failed), "in_flight": len(in_flight)})
                    last_heartbeat = time.monotonic()
        state.update({"completed": completed, "failed": failed, "attempts": attempts, "status": "complete" if not failed else "complete_with_failed_jobs"})
        self._save(state)
        return state


def synthetic_recovery_canary(root: Path) -> dict[str, Any]:
    limits = ResourceLimits(max_workers=2, max_jobs_in_flight=2, max_rss_bytes=10 * 1024**3, max_output_bytes=1024**3, minimum_free_disk_bytes=1)
    jobs = [(f"job-{index}", lambda value=index: value * value) for index in range(4)]
    first = LazySupervisor(root, limits).run(jobs, stop_after_completions=1)
    resumed = LazySupervisor(root, limits).run(jobs)
    replay = LazySupervisor(root, limits).run(jobs)
    return {
        "first_status": first["status"],
        "resumed_status": resumed["status"],
        "replay_status": replay["status"],
        "completed": len(replay["completed"]),
        "idempotent": resumed["completed"] == replay["completed"],
        "pass": first["status"] == "graceful_bound_stop" and resumed["status"] == "complete" and replay["status"] == "complete" and len(replay["completed"]) == 4,
    }
