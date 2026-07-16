#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.telegram_notify import TelegramNotifier, load_telegram_env_files


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    root = Path(args.run_root)
    load_telegram_env_files()

    class TelegramArgs:
        tg_bot_token = ""
        tg_chat_id = ""
        tg_auto_chat = False
        disable_telegram = False
        telegram_dry_run = False

    notifier = TelegramNotifier.from_args(TelegramArgs(), run_label="prior-high-v2-train-scan")
    watcher_dir = root / "watcher"
    watcher_dir.mkdir(parents=True, exist_ok=True)
    last_reported = -1
    last_progress_ts = time.time()
    warned_stall = False
    warned_rss = False
    if notifier.enabled:
        notifier.send("Prior-high v2 scan watcher attached", f"PID={args.pid}\nRun root={root}\nCompleted shards={len(list((root / 'aggregate_shards').glob('*/shard_manifest.json')))} / 48")
    while Path(f"/proc/{args.pid}").exists():
        manifests = list((root / "aggregate_shards").glob("*/shard_manifest.json"))
        completed = len(manifests)
        rss = rss_bytes(args.pid)
        if completed != last_reported:
            last_progress_ts = time.time()
            warned_stall = False
            if notifier.enabled and (completed == 2 or completed % 5 == 0 or completed >= 48):
                notifier.send("Prior-high v2 scan progress", f"Completed shards={completed}/48\nRSS={rss / 1024**3:.2f} GiB\nRun root={root}")
            last_reported = completed
        if rss >= 6 * 1024**3 and not warned_rss:
            warned_rss = True
            if notifier.enabled:
                notifier.send("Prior-high v2 scan RSS warning", f"RSS={rss / 1024**3:.2f} GiB\nPID={args.pid}\nRun root={root}")
        if time.time() - last_progress_ts >= 3600 and not warned_stall:
            warned_stall = True
            if notifier.enabled:
                notifier.send("Prior-high v2 scan no-progress warning", f"No finalized shard for 60 minutes\nPID={args.pid}\nRSS={rss / 1024**3:.2f} GiB\nRun root={root}")
        status = {
            "pid": args.pid, "run_root": str(root), "telegram_enabled": notifier.enabled,
            "completed_shards": completed, "planned_shards": 48, "rss_bytes": rss,
            "seconds_since_progress": time.time() - last_progress_ts, "status": "running", "ts_utc": utc_now(),
        }
        (watcher_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        time.sleep(max(30, args.interval_seconds))
    decision = root / "decision_summary.json"
    if decision.exists():
        payload = json.loads(decision.read_text())
        title = "Prior-high v2 train scan complete" if payload.get("status") == "complete" else "Prior-high v2 train scan ended"
        body = f"Status={payload.get('status')}\nShards={payload.get('shards_completed')}/48\nDefinitions={payload.get('definitions_scored')}\nEvents={payload.get('events_scored')}\nRun root={root}"
    else:
        title = "Prior-high v2 train scan failed or interrupted"
        body = f"Main PID exited without decision_summary.json\nCompleted shards={len(list((root / 'aggregate_shards').glob('*/shard_manifest.json')))}/48\nRun root={root}"
    if notifier.enabled:
        notifier.send(title, body)
    (watcher_dir / "status.json").write_text(json.dumps({"pid": args.pid, "run_root": str(root), "telegram_enabled": notifier.enabled, "status": "process_exited", "ts_utc": utc_now()}, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
