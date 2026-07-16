#!/usr/bin/env python3
"""Non-invasive Telegram/status watcher for the detached DFRL screen."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.telegram_notify import TelegramNotifier, load_telegram_env_files


def process_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return len(fields) > 2 and fields[2] != "Z"
    except Exception:
        return False


def rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval-seconds", type=int, default=180)
    args = parser.parse_args()
    root = Path(args.run_root)
    load_telegram_env_files()

    class TelegramArgs:
        tg_bot_token = ""
        tg_chat_id = ""
        tg_auto_chat = False

    notifier = TelegramNotifier.from_args(TelegramArgs(), run_label="kraken-delayed-flush-reclaim-long")
    state_dir = root / "watcher"
    last_key = ""
    last_change = time.time()
    last_bucket = -1
    last_stage = ""
    stalled_sent = False
    rss_alerts: set[int] = set()
    if notifier.enabled:
        notifier.send("Delayed flush reclaim long screen started", f"PID={args.pid}\nRun root={root}")
    while process_running(args.pid):
        progress = {}
        try:
            progress = json.loads((root / "watch_status.json").read_text())
        except Exception:
            pass
        stage = str(progress.get("stage", "startup"))
        completed = int(progress.get("symbols_completed", 0) or 0)
        key = f"{stage}:{completed}:{progress.get('selected_keys', 0)}"
        if key != last_key:
            last_key = key
            last_change = time.time()
            stalled_sent = False
            bucket = completed // 30
            if notifier.enabled and (stage != last_stage or bucket > last_bucket):
                notifier.send(
                    "Delayed flush reclaim long progress",
                    f"Stage={stage}\nSymbols={completed}\nSelected keys={progress.get('selected_keys', 0)}\n"
                    f"RSS={rss(args.pid)/1024**3:.2f} GiB\nElapsed={progress.get('elapsed_seconds', 0):.0f}s",
                )
            last_bucket = max(last_bucket, bucket)
            last_stage = stage
        current_rss = rss(args.pid)
        for threshold in (6, 8):
            if current_rss >= threshold * 1024**3 and threshold not in rss_alerts:
                rss_alerts.add(threshold)
                if notifier.enabled:
                    notifier.send("Delayed flush reclaim long RSS alert", f"RSS crossed {threshold} GiB\nPID={args.pid}\nRun root={root}")
        if time.time() - last_change >= 3600 and not stalled_sent:
            stalled_sent = True
            if notifier.enabled:
                notifier.send("Delayed flush reclaim long stalled", f"No heartbeat progress for 60 minutes\nPID={args.pid}\nRun root={root}")
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "status.json").write_text(json.dumps({
            "pid": args.pid, "telegram_enabled": notifier.enabled, "rss_bytes": current_rss,
            "seconds_since_progress": time.time() - last_change, "progress": progress,
            "updated_ts": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n")
        time.sleep(max(30, args.interval_seconds))
    decision = root / "decision_summary.json"
    if decision.exists():
        payload = json.loads(decision.read_text())
        title = "Delayed flush reclaim long screen complete" if payload.get("status") == "complete" else "Delayed flush reclaim long screen blocked"
        body = f"Status={payload.get('status')}\nDecision={payload.get('final_decision')}\nDefinitions={payload.get('definitions_evaluated')}\nEvents={payload.get('canonical_event_rows')}\nRun root={root}"
    else:
        title = "Delayed flush reclaim long screen failed or interrupted"
        body = f"PID exited without decision summary\nRun root={root}"
    if notifier.enabled:
        notifier.send(title, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
