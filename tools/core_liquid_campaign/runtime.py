from __future__ import annotations

import errno
import json
import multiprocessing as mp
import os
import signal
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .family_engines.common import EngineInputError


class ResourceGateError(RuntimeError):
    pass


def _default_pid_list() -> list[int]:
    return sorted(int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit())


def _default_status_reader(pid: int) -> str:
    return Path(f"/proc/{pid}/status").read_text(encoding="utf-8")


def _default_statm_reader(pid: int) -> str:
    return Path(f"/proc/{pid}/statm").read_text(encoding="utf-8")


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


class _MissingVmRSS(ResourceGateError):
    pass


def _status_rss(text: str) -> int:
    fields = _status_fields(text)
    if fields.get("State", "")[:1] in {"Z", "X"}:
        # A dead descendant can remain in /proc after its address space has
        # gone.  This is an exact zero, not an I/O or parsing failure; the live
        # supervisor/root sample remains mandatory.
        return 0
    if "VmRSS" not in fields:
        raise _MissingVmRSS("sampled tree process has no VmRSS")
    try:
        return int(fields["VmRSS"].split()[0]) * 1024
    except (ValueError, IndexError) as exc:
        raise ResourceGateError("sampled tree process has no parseable VmRSS") from exc


def _statm_rss(text: str) -> int:
    fields = text.split()
    try:
        pages = int(fields[1])
    except (ValueError, IndexError) as exc:
        raise ResourceGateError("sampled tree process has no parseable statm resident pages") from exc
    if pages < 0:
        raise ResourceGateError("sampled tree process has negative statm resident pages")
    return pages * int(os.sysconf("SC_PAGE_SIZE"))


def _rss_with_statm_fallback(pid: int, status_text: str, statm_reader: Callable[[int], str]) -> int:
    try:
        return _status_rss(status_text)
    except _MissingVmRSS:
        return _statm_rss(statm_reader(pid))


def _vanished(exc: BaseException) -> bool:
    return isinstance(exc, (FileNotFoundError, ProcessLookupError)) or (isinstance(exc, OSError) and exc.errno == errno.ESRCH)


def process_tree_rss(
    root_pid: int | None = None,
    *,
    pid_list: Callable[[], Sequence[int]] = _default_pid_list,
    status_reader: Callable[[int], str] = _default_status_reader,
    statm_reader: Callable[[int], str] = _default_statm_reader,
) -> int:
    root = os.getpid() if root_pid is None else int(root_pid)
    try:
        root_text = status_reader(root)
        root_identity = _status_identity(root_text)
        root_rss = _rss_with_statm_fallback(root, root_text, statm_reader)
    except BaseException as exc:
        if _vanished(exc):
            raise ResourceGateError("supervisor/root process cannot be sampled") from exc
        raise
    if root_identity[0] != root:
        raise ResourceGateError("supervisor/root PID mismatch")
    if root_rss <= 0:
        raise ResourceGateError("supervisor/root RSS sample is not positive")
    identities: dict[int, tuple[int, int]] = {root: root_identity}
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
    total = root_rss
    for pid in sorted(descendants - {root}):
        if pid not in status_text:
            continue
        try:
            total += _status_rss(status_text[pid])
        except _MissingVmRSS:
            # A just-forked or just-exited process can expose identity before
            # VmRSS. Re-sample that exact PID once, then read the kernel's
            # resident-page field. Persistent malformed data remains
            # fail-closed, while ESRCH is the permitted vanished entry.
            try:
                retry = status_reader(pid)
            except BaseException as exc:
                if _vanished(exc):
                    continue
                raise
            retry_pid, retry_ppid = _status_identity(retry)
            if retry_pid != pid:
                raise ResourceGateError("descendant PID changed during RSS re-sample")
            if retry_ppid not in descendants:
                continue
            try:
                total += _rss_with_statm_fallback(pid, retry, statm_reader)
            except BaseException as exc:
                if _vanished(exc):
                    continue
                raise
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


def _process_target(send: Any, task: Callable[[], Any], result_path: Path, job_id: str) -> None:
    """Execute one job and persist its potentially large result before IPC.

    The pipe carries only a small acknowledgement.  This removes the classic
    child-send/parent-wait deadlock for production-sized ledgers.
    """
    # Forked workers must not inherit the supervisor's graceful-stop handler:
    # terminate() must end the worker so it cannot commit after a bound stop.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        result = task()
        atomic_write_json(result_path, {"job_id": job_id, "result": result})
        send.send(("committed", sha256_file(result_path), result_path.stat().st_size))
    except EngineInputError as exc:
        result = {
            "status": "unavailable_data",
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "registered_job_id": job_id,
            "registered_attempt_id": job_id.rsplit(":", 1)[-1],
            "aggregate": {},
            "observation_count": 0,
            "event_ids": [],
            "day_base_net_bps": {},
            "materialization": "explicit_empty_unavailable_observation",
        }
        atomic_write_json(result_path, {"job_id": job_id, "result": result})
        send.send(("unavailable", sha256_file(result_path), result_path.stat().st_size))
    except BaseException as exc:
        try:
            send.send(("error", type(exc).__name__, str(exc)))
        except BaseException:
            pass
    finally:
        send.close()


@dataclass
class _Worker:
    job_id: str
    task: Callable[[], Any]
    process: Any
    receiver: Any
    result_path: Path


class LazySupervisor:
    """Bounded lazy fork-worker supervisor with verified process termination."""

    def __init__(
        self,
        run_root: Path,
        limits: ResourceLimits,
        *,
        heartbeat: Callable[[Mapping[str, Any]], bool] | None = None,
        real_unit_validator: Callable[[str, Any], bool] | None = None,
        rss_sampler: Callable[[], int] = process_tree_rss,
        monotonic: Callable[[], float] = time.monotonic,
        identity_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_root = run_root; self.limits = limits
        self.heartbeat = heartbeat
        self.real_unit_validator = real_unit_validator or (lambda _job_id, _result: False)
        self.rss_sampler = rss_sampler; self.monotonic = monotonic
        self.identity_bindings = dict(identity_bindings or {})
        self.state_path = run_root / "SUPERVISOR_STATE.json"
        self.marker_root = run_root / "markers"; self.artifact_root = run_root / "artifacts"
        self.staging_root = run_root / "staging"
        self._ctx = mp.get_context("fork")

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema": "stage23_supervisor_state_v4", "completed": {}, "failed": {}, "attempts": {}, "generation": 0, "status": "new", "heartbeat_count": 0, "heartbeat_success_count": 0, "first_real_unit_reconciled": False, "health_release": False, "consecutive_supervisor_restarts": 0}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save(self, state: Mapping[str, Any]) -> None:
        updated = dict(state); updated["generation"] = int(updated.get("generation", 0)) + 1
        if isinstance(state, dict):
            state["generation"] = updated["generation"]
        atomic_write_json(self.state_path, updated)

    def _paths(self, job_id: str) -> tuple[Path, Path]:
        stem = canonical_hash({"job_id": job_id})
        return self.artifact_root / f"{stem}.json", self.marker_root / f"{stem}.json"

    def _commit_staged_result(self, job_id: str, staged: Path, expected_sha256: str, real_unit: bool) -> dict[str, Any]:
        artifact, marker = self._paths(job_id)
        if not staged.is_file() or sha256_file(staged) != expected_sha256:
            raise ResourceGateError("worker result acknowledgement does not match staged bytes")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, artifact)
        record = {"job_id": job_id, "artifact": artifact.relative_to(self.run_root).as_posix(), "artifact_sha256": sha256_file(artifact), "status": "complete", "reconciled_real_registered_unit": bool(real_unit)}
        atomic_write_json(marker, record)
        return record

    def _reconcile_all(self) -> dict[str, Any]:
        completed: dict[str, Any] = {}
        if not self.marker_root.exists():
            return completed
        for marker in sorted(self.marker_root.glob("*.json")):
            record = json.loads(marker.read_text(encoding="utf-8")); job_id = str(record.get("job_id"))
            artifact = self.run_root / str(record.get("artifact"))
            if not job_id or job_id in completed or not artifact.is_file() or sha256_file(artifact) != record.get("artifact_sha256"):
                raise ResourceGateError("committed marker/artifact reconciliation failed")
            completed[job_id] = record
        return completed

    def _start(self, job_id: str, task: Callable[[], Any]) -> _Worker:
        receiver, sender = self._ctx.Pipe(duplex=False)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        result_path = self.staging_root / f"{canonical_hash({'job_id': job_id})}.json"
        result_path.unlink(missing_ok=True)
        process = self._ctx.Process(target=_process_target, args=(sender, task, result_path, job_id), name=f"stage23-{canonical_hash(job_id)[:10]}")
        process.start(); sender.close()
        return _Worker(job_id, task, process, receiver, result_path)

    def _stop_workers(self, workers: Mapping[str, _Worker]) -> None:
        deadline = time.monotonic() + self.limits.graceful_stop_seconds
        for worker in workers.values():
            if worker.process.is_alive():
                worker.process.terminate()
        for worker in workers.values():
            remaining = max(0.0, deadline - time.monotonic())
            worker.process.join(remaining)
        for worker in workers.values():
            if worker.process.is_alive():
                worker.process.kill(); worker.process.join()
            worker.receiver.close()
        if any(worker.process.is_alive() for worker in workers.values()):
            raise ResourceGateError("worker process survived the graceful/forced bound stop")

    def _bound_stop(self, workers: dict[str, _Worker], state: dict[str, Any], status: str) -> dict[str, Any]:
        self._stop_workers(workers); workers.clear()
        state["status"] = status; state["all_workers_stopped"] = True
        self._save(state)
        return state

    def run(self, jobs: Iterable[tuple[str, Callable[[], Any]]], *, stop_after_completions: int | None = None, require_health_release: bool = False) -> dict[str, Any]:
        self.limits.validate()
        iterator = iter(jobs)
        exhausted = False
        seen: set[str] = set()
        completed = self._reconcile_all()
        state = self._load_state(); failed = dict(state.get("failed", {})); attempts = {str(key): int(value) for key, value in state.get("attempts", {}).items()}
        recorded_bindings = state.get("identity_bindings")
        if recorded_bindings is not None and recorded_bindings != self.identity_bindings:
            raise ResourceGateError("resume identity bindings do not match the persisted generation")
        pending: deque[tuple[str, Callable[[], Any]]] = deque(); retry_ready: dict[str, tuple[float, Callable[[], Any]]] = {}
        workers: dict[str, _Worker] = {}
        now = self.monotonic(); last_progress = now; last_heartbeat = now
        stop_signal: list[int | None] = [None]
        prior_handlers: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                prior_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda received, _frame: stop_signal.__setitem__(0, received))
        state.update({"status": "running", "limits": asdict(self.limits), "completed": completed, "failed": failed, "attempts": attempts, "all_workers_stopped": False, "service_identity": os.environ.get("SYSTEMD_UNIT") or os.environ.get("TMUX") or "foreground_canary", "launcher_pid": os.getppid(), "supervisor_pid": os.getpid(), "run_root": str(self.run_root), "identity_bindings": self.identity_bindings})
        self._save(state)
        try:
            while workers or pending or retry_ready or not exhausted:
                if stop_signal[0] is not None:
                    state["exact_stop_signal"] = int(stop_signal[0])
                    return self._bound_stop(workers, state, "global_resumable_bound_stop_signal")
                resource = resource_preflight(self.run_root, self.limits, rss_sampler=self.rss_sampler)
                state["peak_process_tree_rss_bytes"] = max(int(state.get("peak_process_tree_rss_bytes", 0)), int(resource["rss_bytes"]))
                state["peak_output_bytes"] = max(int(state.get("peak_output_bytes", 0)), int(resource["output_bytes"]))
                now = self.monotonic()
                for job_id in sorted([key for key, (ready, _) in retry_ready.items() if ready <= now]):
                    _, task = retry_ready.pop(job_id); pending.append((job_id, task))
                while len(workers) + len(pending) < self.limits.max_jobs_in_flight and not exhausted:
                    try:
                        job_id, task = next(iterator)
                    except StopIteration:
                        exhausted = True; break
                    if job_id in seen:
                        raise ResourceGateError("duplicate registered job ID")
                    seen.add(job_id)
                    if job_id not in completed:
                        pending.append((job_id, task))
                while pending and len(workers) < self.limits.max_workers:
                    resource_preflight(self.run_root, self.limits, rss_sampler=self.rss_sampler)
                    job_id, task = pending.popleft(); attempts[job_id] = attempts.get(job_id, 0) + 1
                    workers[job_id] = self._start(job_id, task)
                state.update({"queue_count": len(pending) + len(retry_ready), "in_flight_count": len(workers), "completed_count": len(completed), "worker_pids": sorted(worker.process.pid for worker in workers.values()), "resource_snapshot": resource})
                progressed = False
                for job_id in sorted(list(workers)):
                    worker = workers[job_id]
                    # Drain the tiny acknowledgement while the worker is live;
                    # the large payload is already fsynced/renamed in staging.
                    if not worker.receiver.poll() and worker.process.is_alive():
                        continue
                    message = worker.receiver.recv() if worker.receiver.poll() else ("error", "WorkerProcessExit", f"exitcode={worker.process.exitcode}")
                    worker.process.join()
                    worker.receiver.close(); del workers[job_id]
                    if message[0] in {"committed", "unavailable"}:
                        envelope = json.loads(worker.result_path.read_text(encoding="utf-8"))
                        if envelope.get("job_id") != job_id:
                            raise ResourceGateError("worker staged a result for a different job")
                        result = envelope["result"]
                        real_unit = bool(self.real_unit_validator(job_id, result))
                        completed[job_id] = self._commit_staged_result(job_id, worker.result_path, str(message[1]), real_unit); failed.pop(job_id, None)
                        state["last_committed_marker"] = completed[job_id]
                    elif attempts[job_id] <= len(self.limits.retry_delays_seconds):
                        retry_ready[job_id] = (self.monotonic() + self.limits.retry_delays_seconds[attempts[job_id] - 1], worker.task)
                    else:
                        failed[job_id] = {"error_type": message[1], "error": message[2], "attempts": attempts[job_id], "status": "family_job_exhausted"}
                    progressed = True; last_progress = self.monotonic()
                    state.update({"completed": completed, "failed": failed, "attempts": attempts})
                    self._save(state)
                    if stop_after_completions is not None and len(completed) >= stop_after_completions:
                        return self._bound_stop(workers, state, "graceful_bound_stop")
                now = self.monotonic()
                if now - last_heartbeat >= self.limits.heartbeat_seconds:
                    if self.heartbeat is None:
                        raise ResourceGateError("scheduled heartbeat transport is not configured")
                    payload = {"health": "running", "stage_state": state.get("status"), "completed": len(completed), "failed": len(failed), "in_flight": len(workers), "generation": state.get("generation")}
                    delivered = self.heartbeat(payload)
                    state["heartbeat_count"] = int(state.get("heartbeat_count", 0)) + 1
                    if delivered is True:
                        state["heartbeat_success_count"] = int(state.get("heartbeat_success_count", 0)) + 1
                        state["last_successful_heartbeat_monotonic"] = now
                    last_heartbeat = now; self._save(state)
                successful_heartbeat = state.get("last_successful_heartbeat_monotonic")
                if successful_heartbeat is not None and now - float(successful_heartbeat) > 3900:
                    return self._bound_stop(workers, state, "global_resumable_bound_stop_heartbeat_stale")
                state["first_real_unit_reconciled"] = any(record.get("reconciled_real_registered_unit") is True for record in completed.values())
                state["health_release"] = state["first_real_unit_reconciled"] and int(state.get("heartbeat_success_count", 0)) >= 1
                if now - last_progress > self.limits.no_progress_seconds and (pending or retry_ready or workers or not exhausted):
                    restarts = int(state.get("consecutive_supervisor_restarts", 0)) + 1; state["consecutive_supervisor_restarts"] = restarts
                    status = "restart_required_from_atomic_generation" if restarts <= self.limits.maximum_supervisor_restarts else "global_resumable_bound_stop_stalled"
                    return self._bound_stop(workers, state, status)
                if not progressed:
                    time.sleep(min(self.limits.monitor_interval_seconds, 0.05))
            unknown_markers = set(completed) - seen
            if unknown_markers:
                raise ResourceGateError("reconciled marker belongs to a job absent from the complete lazy registry")
            while require_health_release and not (any(record.get("reconciled_real_registered_unit") is True for record in completed.values()) and int(state.get("heartbeat_success_count", 0)) >= 1):
                if self.heartbeat is None:
                    raise ResourceGateError("health release requires scheduled heartbeat transport")
                now = self.monotonic()
                if now - last_heartbeat >= self.limits.heartbeat_seconds:
                    payload = {"health": "awaiting_release", "stage_state": "work_complete", "completed": len(completed), "failed": len(failed), "in_flight": 0, "generation": state.get("generation")}
                    delivered = self.heartbeat(payload)
                    state["heartbeat_count"] = int(state.get("heartbeat_count", 0)) + 1
                    if delivered is True:
                        state["heartbeat_success_count"] = int(state.get("heartbeat_success_count", 0)) + 1
                        state["last_successful_heartbeat_monotonic"] = now
                    last_heartbeat = now; self._save(state)
                if not any(record.get("reconciled_real_registered_unit") is True for record in completed.values()):
                    raise ResourceGateError("health release requires one reconciled real registered unit")
                if int(state.get("heartbeat_success_count", 0)) < 1:
                    time.sleep(min(self.limits.monitor_interval_seconds, 0.05))
            state.update({"completed": completed, "failed": failed, "attempts": attempts, "status": "complete" if not failed else "complete_with_family_job_exhaustion", "all_workers_stopped": True, "queue_count": 0, "in_flight_count": 0, "completed_count": len(completed), "retry_counts": attempts, "supervisor_pid": os.getpid(), "worker_pids": []})
            state["first_real_unit_reconciled"] = any(record.get("reconciled_real_registered_unit") is True for record in completed.values())
            state["health_release"] = state["first_real_unit_reconciled"] and int(state.get("heartbeat_success_count", 0)) >= 1
            self._save(state)
            return state
        except Exception:
            self._bound_stop(workers, state, "resource_or_common_failure_bound_stop")
            raise
        finally:
            for signum, handler in prior_handlers.items():
                signal.signal(signum, handler)


def detached_service_spec(repository_root: Path, run_root: Path, manifest_sha256: str, workers: int, *, manifest: Path, approval_request: Path, external_approval: Path, cache_manifest: Path) -> dict[str, Any]:
    if workers > 4 or not all(character in "0123456789abcdef" for character in manifest_sha256) or len(manifest_sha256) != 64:
        raise ResourceGateError("invalid detached service binding")
    service_id = f"qlmg-stage22-{manifest_sha256[:12]}"
    command = [
        str(repository_root / ".venv/bin/python"), "-m", "tools.run_stage22_core_liquid_campaign", "run",
        "--manifest", str(manifest), "--approval-request", str(approval_request),
        "--external-approval", str(external_approval), "--cache-manifest", str(cache_manifest),
        "--run-root", str(run_root), "--workers", str(workers),
    ]
    unit = "\n".join([
        "[Unit]", "Description=QLMG Stage 23 reviewed campaign", "After=network-online.target",
        "StartLimitIntervalSec=3600", "StartLimitBurst=3",
        "[Service]", "Type=simple", f"WorkingDirectory={repository_root}",
        "ExecStart=" + " ".join(command), "Restart=on-failure", "RestartSec=60",
        "KillMode=control-group", "TimeoutStopSec=300", "NoNewPrivileges=true",
        "[Install]", "WantedBy=default.target", "",
    ])
    return {"service_id": service_id, "working_directory": str(repository_root), "run_root": str(run_root), "workers": workers, "restart": "on-failure", "restart_limit": 3, "independent_of_chat": True, "exec_start": command, "systemd_user_unit": unit, "environment": {"PYTHONPATH": None, "invocation": "reviewed worktree .venv/bin/python -m tools.run_stage22_core_liquid_campaign"}}


def install_detached_service(spec: Mapping[str, Any], unit_root: Path) -> Path:
    """Install and start an exact reviewed systemd-user service."""
    service_id = str(spec["service_id"])
    if not service_id.startswith("qlmg-stage22-") or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in service_id):
        raise ResourceGateError("unsafe detached service identity")
    unit_root.mkdir(parents=True, exist_ok=True)
    unit_path = unit_root / f"{service_id}.service"
    unit_path.write_text(str(spec["systemd_user_unit"]), encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", unit_path.name], check=True)
    active = subprocess.run(["systemctl", "--user", "is-active", unit_path.name], check=False, capture_output=True, text=True)
    if active.stdout.strip() != "active":
        raise ResourceGateError("detached systemd-user service did not become active")
    return unit_path


def launch_detached_supervisor(spec: Mapping[str, Any], unit_root: Path) -> dict[str, Any]:
    """Start the reviewed command independently through systemd or tmux."""
    systemd_available = shutil.which("systemctl") is not None and subprocess.run(
        ["systemctl", "--user", "show-environment"], check=False, capture_output=True,
    ).returncode == 0
    if systemd_available:
        unit = install_detached_service(spec, unit_root)
        return {"mechanism": "systemd_user", "service_identity": unit.name, "active": True, "independent_of_chat": True}
    if shutil.which("tmux") is None:
        raise ResourceGateError("neither systemd --user nor tmux is available for detached supervision")
    service_id = str(spec["service_id"])
    subprocess.run(["tmux", "new-session", "-d", "-s", service_id, *[str(value) for value in spec["exec_start"]]], check=True)
    active = subprocess.run(["tmux", "has-session", "-t", service_id], check=False).returncode == 0
    if not active:
        raise ResourceGateError("detached tmux supervisor did not remain active")
    return {"mechanism": "tmux", "service_identity": service_id, "active": True, "independent_of_chat": True}


def synthetic_recovery_canary(root: Path) -> dict[str, Any]:
    class Clock:
        value = 0.0
        def __call__(self) -> float:
            self.value += 1000.0
            return self.value
    clock = Clock(); heartbeats: list[Mapping[str, Any]] = []; large = "x" * (2 * 1024 * 1024)
    retry_flag = root / "retry.flag"
    def flaky() -> int:
        if not retry_flag.exists():
            retry_flag.write_text("first worker exited\n", encoding="utf-8")
            raise RuntimeError("synthetic worker death")
        return {"registered_attempt_id": "flaky", "value": large}
    jobs = [("stable-1", lambda: {"registered_attempt_id": "stable-1", "value": large}), ("flaky", flaky), ("stable-2", lambda: {"registered_attempt_id": "stable-2", "value": large})]
    limits = ResourceLimits(max_workers=2, max_jobs_in_flight=2, max_output_bytes=1024**3, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0, heartbeat_seconds=1, monitor_interval_seconds=0.001)
    heartbeat = lambda payload: (heartbeats.append(payload) or True)
    validator = lambda job_id, result: isinstance(result, Mapping) and result.get("registered_attempt_id") == job_id
    first = LazySupervisor(root, limits, heartbeat=heartbeat, real_unit_validator=validator, monotonic=clock).run(iter(jobs), stop_after_completions=1)
    resumed = LazySupervisor(root, limits, heartbeat=heartbeat, real_unit_validator=validator, monotonic=clock).run(iter(jobs))
    replay = LazySupervisor(root, limits, heartbeat=heartbeat, real_unit_validator=validator, monotonic=clock).run(iter(jobs))
    markers = list((root / "markers").glob("*.json")); artifacts = list((root / "artifacts").glob("*.json"))
    excursion_root = root / "resource_excursion"; rss_calls = {"count": 0}; late_write = excursion_root / "late-write"
    def excursion_rss() -> int:
        rss_calls["count"] += 1
        return limits.max_rss_bytes + 1 if rss_calls["count"] == 3 else 1
    def slow() -> int:
        time.sleep(1.0); late_write.write_text("unsafe\n", encoding="utf-8"); return 3
    excursion_stopped = False
    try:
        LazySupervisor(excursion_root, limits, heartbeat=heartbeat, rss_sampler=excursion_rss).run(iter([("slow", slow)]))
    except ResourceGateError:
        excursion_stopped = True
    time.sleep(0.05)
    no_late_write = not late_write.exists()
    excursion_resumed = LazySupervisor(excursion_root, limits, heartbeat=heartbeat).run(iter([("slow", lambda: 3)]))
    return {
        "first_status": first["status"], "resumed_status": resumed["status"], "replay_status": replay["status"],
        "completed": len(replay["completed"]), "markers_reconciled": len(markers), "artifacts_reconciled": len(artifacts),
        "worker_death_retried": retry_flag.exists(), "heartbeat_delivered": bool(heartbeats), "health_release": bool(replay.get("health_release")),
        "continuous_resource_excursion_stopped": excursion_stopped, "all_stopped_before_bound_stop": no_late_write,
        "resource_excursion_idempotently_resumed": excursion_resumed["status"] == "complete", "idempotent": resumed["completed"] == replay["completed"],
        "large_result_bytes": len(large),
        "pass": first["status"] == "graceful_bound_stop" and resumed["status"] == "complete" and replay["status"] == "complete" and len(replay["completed"]) == 3 and len(markers) == 3 and len(artifacts) == 3 and retry_flag.exists() and bool(heartbeats) and bool(replay.get("health_release")) and excursion_stopped and no_late_write and excursion_resumed["status"] == "complete",
    }


__all__ = ["LazySupervisor", "ResourceGateError", "ResourceLimits", "detached_service_spec", "directory_size", "install_detached_service", "launch_detached_supervisor", "physical_memory_bytes", "process_tree_rss", "resource_preflight", "synthetic_recovery_canary"]
