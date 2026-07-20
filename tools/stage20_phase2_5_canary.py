#!/usr/bin/env python3
"""Outcome-free end-to-end canary for the Stage 20 campaign supervisor."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qlmg_stage19_funding import Stage19FundingEngine
from tools.qlmg_stage20_campaign import (
    FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256, STAGE19_REL, Stage20Error,
    atomic_json, file_sha256, score_executions, stable_hash, utc_now,
)
from tools.qlmg_stage20_launch_gates import validate_source_manifest, write_gate
from tools.run_stage20_economic_campaign import (
    _atomic_parquet, _cell_metrics, execute_jobs, verify_outer_freeze,
)


class CanaryNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def send(self, title: str, body: str = "") -> bool:
        self.messages.append((title, body))
        return True


def _synthetic_supervisor_worker(bundle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage19 = ROOT / STAGE19_REL
    contract = json.loads((stage19 / "FUNDING_COST_AND_COVERAGE_CONTRACT.json").read_text())
    funding = Stage19FundingEngine(
        FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256,
        stage19 / "FUNDING_GAP_ALLOWANCE_TABLE.csv", contract["gap_allowance_table_sha256"],
    )
    results = []
    for job in bundle:
        job_id = stable_hash(job)
        marker = Path(job["work_root"]) / "jobs" / f"{job_id}.json"
        if marker.is_file():
            prior = json.loads(marker.read_text())
            if all(Path(row["path"]).is_file() and file_sha256(Path(row["path"])) == row["sha256"]
                   for row in prior.get("files", [])):
                results.append({**prior, "resumed": True})
                continue
        entry = pd.Timestamp("2024-01-02T00:00:00Z") + pd.Timedelta(hours=job["index"])
        executions = pd.DataFrame([{
            "cell_id": f"SYNTH:{job['index']}", "symbol": "PF_XBTUSD", "side": "long",
            "entry_ts": entry, "exit_ts": entry + pd.Timedelta(hours=1),
            "entry_open": 40000.0, "exit_open": 40010.0,
        }])
        scored = score_executions(executions, funding)
        target = Path(job["work_root"]) / "scored" / f"{job_id}.parquet"
        _atomic_parquet(scored, target)
        result = {
            "status": "pass", "synthetic": True, "job_id": job_id,
            "registered_cell_ids": [f"SYNTH:{job['index']}"],
            "files": [{"path": str(target), "bytes": target.stat().st_size,
                       "sha256": file_sha256(target), "rows": 1}],
            "funding_fields_present": all(column in scored for column in (
                "funding_adverse_exact_bps", "funding_base_gap_cost_bps",
                "funding_start_alignment_bps", "funding_end_alignment_bps")),
            "protected_rows_opened": 0, "Capitalcom_payload_opened": False,
        }
        atomic_json(marker, result)
        results.append(result)
    return results


def _args(work_root: Path) -> argparse.Namespace:
    return argparse.Namespace(run_root=work_root)


def _state() -> dict[str, Any]:
    return {"status": "synthetic_canary", "generation": 1, "jobs_complete": 0,
            "health_release_status": "pending", "first_reconciled_real_cell": None,
            "first_scheduled_heartbeat_delivered": False,
            "scheduler_accepting_submissions": True}


def _run_in(work_root: Path) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    notifier = CanaryNotifier()
    state = _state()
    atomic_json(work_root / "CAMPAIGN_STATE.json", state)
    jobs = [[{"index": index, "work_root": str(work_root)}] for index in range(4)]
    limits = {"max_workers": 2, "max_wall_seconds": 120, "max_rss_bytes": 5 * 1024**3,
              "max_output_bytes": 100 * 1024**2, "heartbeat_seconds": .01}
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        execute_jobs(pool, jobs, _args(work_root), state, notifier, started, started,
                     results, worker=_synthetic_supervisor_worker, limits=limits)
    first_state = json.loads((work_root / "CAMPAIGN_STATE.json").read_text())

    recovered: list[dict[str, Any]] = []
    recovery_state = _state()
    with ProcessPoolExecutor(max_workers=2) as pool:
        execute_jobs(pool, jobs, _args(work_root), recovery_state, notifier, time.monotonic(),
                     time.monotonic() + 60, recovered,
                     worker=_synthetic_supervisor_worker, limits=limits)

    stop_root = work_root / "bound_stop"
    stop_root.mkdir(exist_ok=True)
    atomic_json(stop_root / "occupied.json", {"bytes": "already above bound"})
    stopped = False
    stop_state = _state()
    stop_limits = dict(limits, max_output_bytes=1)
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            execute_jobs(pool, jobs, _args(stop_root), stop_state, notifier, time.monotonic(),
                         time.monotonic() + 60, [], worker=_synthetic_supervisor_worker,
                         limits=stop_limits)
    except Stage20Error as exc:
        stopped = str(exc) == "campaign_output_limit"
    persisted_stop = json.loads((stop_root / "CAMPAIGN_STATE.json").read_text())

    freeze_path = work_root / "freeze.json"
    selected = ["SYNTH:0"]
    freeze_sha = stable_hash({"fold": "SYNTH:F1", "cells": selected})
    atomic_json(freeze_path, {"selected_cell_ids": selected, "freeze_sha256": freeze_sha})
    freeze_verified = verify_outer_freeze({
        "required_freeze_path": str(freeze_path), "required_freeze_sha256": freeze_sha,
        "selected_cell_ids": selected,
    })["selected_cell_ids"] == selected

    partial = pd.read_parquet(results[0]["files"][0]["path"])
    partial["source_model_id"] = "I_1"
    observed_cell = str(partial.iloc[0].cell_id)
    metrics = _cell_metrics(partial, [{"cell_id": observed_cell, "family": "SYNTH",
                                      "canonical_translation_id": "SYNTH:T0", "complexity": 1}],
                            ["I_1", "I_EMPTY"], 20, 86400)[0]
    empty_preserved = (metrics["unavailable_inner_fold_count"] == 1
                       and np.isnan(metrics["p20_inner_fold_base_net_mean_bps"]))
    passed = all((
        len(results) == 4, first_state.get("scheduler_max_in_flight") == 2,
        first_state.get("first_scheduled_heartbeat_delivered") is True,
        all(row.get("funding_fields_present") for row in results),
        len(recovered) == 4 and all(row.get("resumed") for row in recovered),
        stopped, persisted_stop.get("scheduler_accepting_submissions") is False,
        persisted_stop.get("bundles_submitted", 0) == 0, freeze_verified,
        empty_preserved, not (work_root / "HEALTH_RELEASE.json").exists(),
    ))
    result = {
        "status": "pass" if passed else "fail", "created_at_utc": utc_now(),
        "synthetic_only": True, "real_economic_outcomes_computed": 0,
        "protected_rows_opened": 0, "Capitalcom_payload_opened": False,
        "real_supervisor_execute_jobs": True, "real_funding_integration": True,
        "bounded_lazy_scheduler": first_state.get("scheduler_max_in_flight") == 2,
        "scheduled_heartbeat_exercised": first_state.get("first_scheduled_heartbeat_delivered") is True,
        "synthetic_health_release_refused": not (work_root / "HEALTH_RELEASE.json").exists(),
        "graceful_pre_submission_bound_stop": stopped,
        "persisted_stop_state": persisted_stop.get("scheduler_accepting_submissions") is False,
        "idempotent_recovery": len(recovered) == 4 and all(row.get("resumed") for row in recovered),
        "outer_freeze_guard": freeze_verified, "empty_inner_fold_preserved": empty_preserved,
    }
    atomic_json(work_root / "CANARY_EVIDENCE.json", result)
    return result


def run(work_root: Path | None = None) -> dict[str, Any]:
    if work_root is not None:
        return _run_in(work_root)
    with tempfile.TemporaryDirectory() as directory:
        return _run_in(Path(directory))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, required=True)
    args = parser.parse_args()
    validate_source_manifest(args.source_manifest)
    result = run(args.work_root)
    if result["status"] != "pass":
        print(json.dumps(result, sort_keys=True))
        return 2
    gate = write_gate(
        args.output, "synthetic_supervisor_canary",
        [args.approval, args.source_manifest,
         args.event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json",
         args.work_root / "CANARY_EVIDENCE.json", args.work_root / "CAMPAIGN_STATE.json",
         args.work_root / "HEARTBEAT.json", args.work_root / "bound_stop" / "CAMPAIGN_STATE.json"],
        result,
    )
    print(json.dumps({"status": gate["status"], "gate_id": gate["gate_id"],
                      "binding_sha256": gate["binding_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
