#!/usr/bin/env python3
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

    notifier = TelegramNotifier.from_args(TelegramArgs(), run_label="kraken-backside-blowoff-short")
    state_dir = root / "watcher"
    state_dir.mkdir(parents=True, exist_ok=True)
    last_key = ""
    last_change = time.time()
    last_bucket = -1
    last_stage = ""
    stalled_sent = False
    rss_alerts_sent: set[str] = set()
    if notifier.enabled:
        notifier.send("Backside blowoff short screen started", f"PID={args.pid}\nRun root={root}")
    while process_running(args.pid):
        progress = {}
        try:
            progress = json.loads((root / "watch_status.json").read_text())
        except Exception:
            pass
        stage = str(progress.get("stage", "startup"))
        completed = int(progress.get("symbols_completed", progress.get("definitions_completed", 0)) or 0)
        planned = int(progress.get("symbols_planned", progress.get("definitions_planned", 0)) or 0)
        rows = int(progress.get("raw_signals", progress.get("accepted_rows", progress.get("accepted_trades", 0))) or 0)
        key = f"{stage}:{completed}:{rows}"
        if key != last_key:
            last_key = key
            last_change = time.time()
            stalled_sent = False
            bucket = completed // 30
            if notifier.enabled and (stage != last_stage or bucket > last_bucket):
                notifier.send(
                    "Backside blowoff short screen progress",
                    f"Stage={stage}\nProgress={completed}/{planned}\nRows={rows}\n"
                    f"RSS={rss(args.pid)/1024**3:.2f} GiB\nElapsed={progress.get('elapsed_seconds', 0):.0f}s",
                )
            last_bucket = max(last_bucket, bucket)
            last_stage = stage
        if time.time() - last_change >= 3600 and not stalled_sent:
            stalled_sent = True
            if notifier.enabled:
                notifier.send("Backside blowoff short screen stalled", f"No heartbeat progress for 60 minutes\nPID={args.pid}\nRun root={root}")
        current_rss = rss(args.pid)
        for label, threshold in (("6_gib", 6 * 1024**3), ("8_gib", 8 * 1024**3)):
            if current_rss >= threshold and label not in rss_alerts_sent:
                rss_alerts_sent.add(label)
                if notifier.enabled:
                    notifier.send("Backside repaired screen RSS alert", f"RSS={current_rss/1024**3:.2f} GiB\nThreshold={label}\nRun root={root}")
        (state_dir / "status.json").write_text(json.dumps({
            "pid": args.pid, "telegram_enabled": notifier.enabled, "rss_bytes": rss(args.pid),
            "seconds_since_progress": time.time() - last_change, "progress": progress,
            "updated_ts": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n")
        time.sleep(max(30, args.interval_seconds))
    decision = root / "decision_summary.json"
    if decision.exists():
        payload = json.loads(decision.read_text())
        title = "Backside blowoff short screen complete" if payload.get("status") == "complete" else "Backside blowoff short screen blocked"
        body = f"Status={payload.get('status')}\nClassification={payload.get('final_classification')}\nDefinitions={payload.get('definitions_evaluated')}\nEvents={payload.get('accepted_trades')}\nRun root={root}"
    else:
        title = "Backside blowoff short screen failed or interrupted"
        body = f"PID exited without decision summary\nRun root={root}"
    if notifier.enabled:
        notifier.send(title, body)
    (state_dir / "status.json").write_text(json.dumps({
        "pid": args.pid, "telegram_enabled": notifier.enabled, "process_running": False,
        "decision_summary_present": decision.exists(), "updated_ts": datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
