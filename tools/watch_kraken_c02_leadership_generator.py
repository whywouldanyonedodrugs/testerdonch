#!/usr/bin/env python3
"""Detached progress and Telegram watcher for the outcome-free C02 build."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.telegram_notify import TelegramNotifier, load_telegram_env_files


def running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    load_telegram_env_files()

    class TelegramArgs:
        tg_bot_token = ""
        tg_chat_id = ""
        tg_auto_chat = False

    notifier = TelegramNotifier.from_args(TelegramArgs(), run_label="C02 leadership generator Stage 3B")
    last_bucket = -1
    if notifier.enabled:
        notifier.send("C02 Stage 3B started", f"Outcome-free generator\nPID={args.pid}\nArchive={args.run_root}")
    while running(args.pid):
        try:
            status = json.loads((args.run_root / "watch_status.json").read_text())
        except Exception:
            status = {}
        completed = int(status.get("symbols_completed", 0))
        bucket = completed // 25
        if notifier.enabled and bucket > last_bucket:
            notifier.send("C02 Stage 3B progress", f"Symbols={completed}/{status.get('symbols_total', '?')}\nImpulses={status.get('impulse_events', 0)}\nFailures={status.get('failure_events', 0)}\nElapsed={status.get('elapsed_seconds', 0):.0f}s")
        last_bucket = max(last_bucket, bucket)
        time.sleep(max(30, args.interval_seconds))
    summary_path = args.run_root / "C02_GENERATION_SUMMARY.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        title = "C02 Stage 3B complete"
        body = f"Events={summary.get('events')}\nFailures={summary.get('failures')}\nAlignment={summary.get('alignment_status')}\nArchive={args.run_root}"
    else:
        title = "C02 Stage 3B failed or interrupted"
        body = f"PID exited without generation summary\nArchive={args.run_root}"
    if notifier.enabled:
        notifier.send(title, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
