#!/usr/bin/env python3
"""Reviewed one-use Stage 20 resume orchestration; not a campaign semantic change."""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.qlmg_stage20_launch_gates as launch_gates

RESUME_REPOSITORY_AUTHORITY = "7e7c7b47a7693b7923097f01c22b4b7287b6d971"
ORIGINAL_DEADLINE = dt.datetime.fromisoformat("2026-07-20T18:49:50.022014+00:00")
AUDIT_RESERVE_SECONDS = 120
EXPECTED_MARKERS = 6459
EXPECTED_MARKER_INVENTORY = "5a94431f604471597b8b6a74e9f980b20183d6d94e3660a68e14fb88787e8f79"
EXPECTED_ARTIFACT_CLAIMS = 3193
EXPECTED_ARTIFACT_INVENTORY = "06346a213ff8d2bf364696d08a0254a9003f6ee753612956505a8b7d6ed80407"

# The external approval remains bound to its original approved start commit.
# Only resume/review gates and current Git authority use the post-stop commit.
launch_gates.START_COMMIT = RESUME_REPOSITORY_AUTHORITY

from tools import run_stage20_economic_campaign as campaign
from tools.qlmg_stage20_campaign import atomic_json, file_sha256, stable_hash, utc_now
from tools.telegram_notify import TelegramNotifier


def reconcile_existing(run_root: Path) -> dict[str, Any]:
    markers = sorted((run_root / "state" / "jobs").glob("*.json"))
    records, artifacts = [], []
    for marker in markers:
        record = json.loads(marker.read_text())
        if (record.get("status") != "pass" or record.get("job_id") != marker.stem
                or record.get("protected_rows_opened") != 0
                or record.get("Capitalcom_payload_opened") is not False):
            raise campaign.Stage20Error(f"invalid reusable marker: {marker}")
        campaign.verify_job_result(record)
        records.append(record)
        artifacts.extend(record.get("files", []))
    marker_inventory = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in markers
    ]
    summary = {
        "marker_count": len(markers),
        "marker_inventory_sha256": stable_hash(marker_inventory),
        "artifact_claim_count": len(artifacts),
        "artifact_claims_sha256": stable_hash(sorted(artifacts, key=lambda row: row["path"])),
        "protected_rows_opened": 0,
        "Capitalcom_payload_opened": False,
    }
    expected = (
        summary["marker_count"] == EXPECTED_MARKERS
        and summary["marker_inventory_sha256"] == EXPECTED_MARKER_INVENTORY
        and summary["artifact_claim_count"] == EXPECTED_ARTIFACT_CLAIMS
        and summary["artifact_claims_sha256"] == EXPECTED_ARTIFACT_INVENTORY
    )
    if not expected:
        raise campaign.Stage20Error("reusable marker/artifact inventory drift")
    return summary


def install_missing_only_scheduler(event_root: Path, run_root: Path) -> None:
    original_execute = campaign.execute_jobs
    event_hashes: dict[str, str] = {}

    def reusable(job: dict[str, Any]) -> dict[str, Any] | None:
        job_id = stable_hash(job)
        marker = run_root / "state" / "jobs" / f"{job_id}.json"
        if not marker.is_file():
            return None
        prior = json.loads(marker.read_text())
        event_path = event_root / "events" / f"{job['symbol']}.parquet"
        event_hashes.setdefault(job["symbol"], file_sha256(event_path))
        if prior.get("event_sha256") != event_hashes[job["symbol"]]:
            raise campaign.Stage20Error(f"reusable marker event hash drift: {marker}")
        campaign.verify_job_result(prior)
        return {**prior, "resumed": True}

    def missing_only_execute(pool, jobs, args, state, notifier, started, next_heartbeat,
                             results, **kwargs):
        missing = []
        reused = []
        for bundle in jobs:
            pending = []
            for job in bundle:
                prior = reusable(job)
                if prior is None:
                    pending.append(job)
                else:
                    reused.append(prior)
            if pending:
                missing.append(pending)
        results.extend(reused)
        state["reused_completed_jobs"] = state.get("reused_completed_jobs", 0) + len(reused)
        state["missing_jobs_submitted"] = state.get("missing_jobs_submitted", 0)
        if state.get("first_reconciled_real_cell") is None:
            first = next((row for row in reused if row.get("registered_cell_ids") and row.get("files")), None)
            if first:
                state["first_reconciled_real_cell"] = {
                    "cell_id": first["registered_cell_ids"][0], "job_id": first["job_id"],
                    "files": first["files"], "reconciled_at_utc": utc_now(), "resumed": True,
                }
        state["jobs_complete"] = len(results)
        state["registered_cells_with_completed_job"] = len({
            cell_id for row in results for cell_id in row.get("registered_cell_ids", [])
        })
        state["generation"] += 1
        atomic_json(args.run_root / "CAMPAIGN_STATE.json", state)
        if time.monotonic() >= next_heartbeat:
            snapshot = campaign.operational_snapshot(args, started, campaign.DEFAULT_LIMITS)
            campaign.emit_heartbeat(args, state, notifier, snapshot)
            next_heartbeat += campaign.DEFAULT_LIMITS["heartbeat_seconds"]
        state["missing_jobs_submitted"] += sum(len(bundle) for bundle in missing)
        return original_execute(
            pool, missing, args, state, notifier, started, next_heartbeat, results, **kwargs
        )

    campaign.execute_jobs = missing_only_execute


def main() -> int:
    args = campaign.parser().parse_args()
    reconciliation = reconcile_existing(args.run_root)
    remaining = int((ORIGINAL_DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds())
    remaining -= AUDIT_RESERVE_SECONDS
    if remaining <= 0:
        raise campaign.Stage20Error("original campaign wall-time deadline exhausted")
    campaign.MAX_WALL_SECONDS = remaining
    campaign.DEFAULT_LIMITS = dict(campaign.DEFAULT_LIMITS, max_wall_seconds=remaining)
    campaign.HEARTBEAT_SECONDS = 0
    install_missing_only_scheduler(args.event_root, args.run_root)
    atomic_json(args.run_root / "RESUME_RUNTIME_BOUNDARY.json", {
        "status": "pass", "verified_at_utc": utc_now(),
        "repository_authority": RESUME_REPOSITORY_AUTHORITY,
        "original_deadline_utc": ORIGINAL_DEADLINE.isoformat().replace("+00:00", "Z"),
        "audit_reserve_seconds": AUDIT_RESERVE_SECONDS,
        "maximum_resume_runtime_seconds": remaining,
        "first_overdue_heartbeat_due_immediately": True,
        "existing_reconciliation": reconciliation,
        "completed_jobs_submitted_again": False,
    })
    return campaign.run(args)


if __name__ == "__main__":
    namespace = None
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            namespace = campaign.parser().parse_args()
            state_path = namespace.run_root / "CAMPAIGN_STATE.json"
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            state.update({
                "status": "global_stop", "reason": f"{type(exc).__name__}: {exc}",
                "stopped_at_utc": utc_now(), "resumable_state_preserved": True,
                "scheduler_accepting_submissions": False,
                "generation": int(state.get("generation", 0)) + 1,
            })
            atomic_json(state_path, state)
            notifier = TelegramNotifier.from_args(namespace, run_label="stage20-stage19-resume")
            state["Telegram_global_stop_delivered"] = notifier.send(
                "GLOBAL STOP", f"reason_type={type(exc).__name__}"
            )
            atomic_json(state_path, state)
        except Exception:
            pass
        print(json.dumps({"status": "global_stop", "reason": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
