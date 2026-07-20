#!/usr/bin/env python3
"""Outcome-free transactional supervisor benchmark for the frozen registry."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import resource
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def process_tree_rss(root_pid: int) -> int:
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit(): continue
        try:
            lines = (entry / "status").read_text().splitlines()
            parent = int(next(line.split()[1] for line in lines if line.startswith("PPid:")))
            rss_lines = [line for line in lines if line.startswith("VmRSS:")]
            if not rss_lines:
                if int(entry.name) == root_pid: raise RuntimeError("supervisor/root RSS sample unavailable")
                continue
            rss = int(rss_lines[0].split()[1]) * 1024
            table[int(entry.name)] = (parent, rss)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            if exc.errno == errno.ESRCH: continue
            raise
    if root_pid not in table: raise RuntimeError("supervisor/root RSS sample unavailable")
    selected, frontier = {root_pid}, {root_pid}
    while frontier:
        children = {pid for pid, (parent, _) in table.items() if parent in frontier and pid not in selected}
        selected.update(children); frontier = children
    total = sum(table[pid][1] for pid in selected)
    if total <= 0: raise RuntimeError("supervisor/root RSS sample unavailable")
    return total


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def job(row: dict, root: Path) -> dict:
    job_id = hashlib.sha256(canonical({"synthetic": True, "attempt_id": row["attempt_id"], "address": row["canonical_economic_address_sha256"]})).hexdigest()
    marker = root / "markers" / f"{job_id}.json"
    if marker.is_file():
        prior = json.loads(marker.read_text()); prior["resumed"] = True; prior["marker_bytes"] = marker.stat().st_size; prior["created"] = False; return prior
    result = {"status": "pass", "synthetic": True, "job_id": job_id, "attempt_id": row["attempt_id"],
              "input_address": row["canonical_economic_address_sha256"], "outcome_reader_opened": False,
              "protected_rows_opened": 0, "Capitalcom_payload_opened": False}
    atomic_json(marker, result); result["marker_bytes"] = marker.stat().st_size; result["created"] = True; return result


def run(registry: Path, root: Path, workers: int, wall_seconds: int,
        rss_limit: int, output_limit: int, heartbeat_seconds: int,
        stop_after: int = 0) -> dict:
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    started = time.monotonic(); next_heartbeat = started + heartbeat_seconds
    state_path = root / "CAMPAIGN_STATE.json"
    prior = json.loads(state_path.read_text()) if state_path.is_file() else {}
    state = {"schema": "synthetic_supervisor_state_v1", "status": "running", "generation": int(prior.get("generation", 0)) + 1,
             "registered": len(rows), "complete": 0, "outcome_reader_opened": False,
             "protected_rows_opened": 0, "Capitalcom_payload_opened": False, "max_in_flight": 0}
    atomic_json(state_path, state)
    iterator = iter(rows); pending = {}; complete = []; max_rss = max_output = 0; heartbeats = 0; exhausted = False
    tracked_output = directory_bytes(root); bound_stop_requested = False

    def limits() -> None:
        nonlocal max_rss, max_output, tracked_output
        elapsed = time.monotonic() - started; rss = process_tree_rss(os.getpid()); output = tracked_output
        max_rss = max(max_rss, rss); max_output = max(max_output, output)
        if elapsed >= wall_seconds: raise RuntimeError("wall_time_limit")
        if rss >= rss_limit: raise RuntimeError("RSS_limit")
        if output >= output_limit: raise RuntimeError("output_limit")
        if shutil.disk_usage(root).free < output_limit: raise RuntimeError("free_disk_limit")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                limits()
                try: row = next(iterator)
                except StopIteration: exhausted = True; break
                future = pool.submit(job, row, root); pending[future] = row
                state["max_in_flight"] = max(state["max_in_flight"], len(pending))
            if not pending: continue
            done, _ = wait(tuple(pending), timeout=0.05, return_when=FIRST_COMPLETED)
            if not done: limits(); continue
            for future in done:
                pending.pop(future); result = future.result()
                if result["outcome_reader_opened"] or result["protected_rows_opened"]: raise RuntimeError("outcome firewall")
                if result.get("created"): tracked_output += int(result["marker_bytes"])
                complete.append(result)
            if stop_after and len(complete) >= stop_after:
                exhausted = True; bound_stop_requested = True
            now = time.monotonic()
            if now >= next_heartbeat:
                heartbeats += 1; atomic_json(root / f"heartbeat_{heartbeats:04d}.json", {"complete": len(complete), "partial_rankings": False})
                next_heartbeat += heartbeat_seconds
            state.update({"complete": len(complete), "generation": state["generation"] + 1}); atomic_json(state_path, state)
    tracked_output = directory_bytes(root); limits()
    terminal_status = "bound_stop" if bound_stop_requested else "terminal_complete"
    state.update({"status": terminal_status, "complete": len(complete), "generation": state["generation"] + 1}); atomic_json(state_path, state)
    elapsed = time.monotonic() - started
    return {"schema": "no_outcome_runtime_benchmark_v1", "status": terminal_status,
            "registry_sha256": sha256_file(registry), "registered_jobs": len(rows),
            "completed_jobs": len(complete), "workers": workers, "max_in_flight": state["max_in_flight"],
            "elapsed_seconds": elapsed, "jobs_per_second": len(complete) / elapsed, "peak_sampled_process_tree_rss_bytes": max_rss,
            "peak_output_bytes": max_output, "heartbeats": heartbeats, "state_generation": state["generation"],
            "idempotent_resume_markers": sum(bool(row.get("resumed")) for row in complete),
            "outcome_reader_opened": False, "protected_rows_opened": 0, "Capitalcom_payload_opened": False,
            "maxrss_ru_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--wall-seconds", type=int, default=120)
    parser.add_argument("--rss-limit", type=int, default=2 * 1024**3); parser.add_argument("--output-limit", type=int, default=1024**3)
    parser.add_argument("--heartbeat-seconds", type=int, default=1)
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args(); result = run(args.registry, args.run_root, args.workers, args.wall_seconds, args.rss_limit, args.output_limit, args.heartbeat_seconds, args.stop_after)
    atomic_json(args.output, result); print(json.dumps(result, sort_keys=True))
