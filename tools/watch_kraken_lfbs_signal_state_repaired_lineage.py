#!/usr/bin/env python3
"""Non-invasive Telegram watcher for the three-stage repaired LFBS lineage."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.telegram_notify import TelegramNotifier, load_telegram_env_files


def running(pid: int) -> bool:
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


def read_progress(roots: list[Path]) -> tuple[Path, dict]:
    for root in reversed(roots):
        path = root / "watch_status.json"
        if path.exists():
            try:
                return root, json.loads(path.read_text())
            except Exception:
                pass
    return roots[0], {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--presample-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    roots = [args.screen_root, args.presample_root, args.canonical_root]
    load_telegram_env_files()

    class TelegramArgs:
        tg_bot_token = ""
        tg_chat_id = ""
        tg_auto_chat = False

    notifier = TelegramNotifier.from_args(TelegramArgs(), run_label="kraken-lfbs-signal-state-repaired-lineage")
    state_dir = args.screen_root / "watcher"
    state_dir.mkdir(parents=True, exist_ok=True)
    last_key = ""
    last_change = time.time()
    last_stage = ""
    last_bucket = -1
    stalled = False
    rss_alerts: set[int] = set()
    if notifier.enabled:
        notifier.send("LFBS repaired lineage started", f"PID={args.pid}\nScreen={args.screen_root}\nPresample={args.presample_root}\nCanonical={args.canonical_root}")
    while running(args.pid):
        active_root, progress = read_progress(roots)
        stage = str(progress.get("stage", "startup"))
        completed = int(progress.get("symbols_completed", 0) or 0)
        key = json.dumps({"root": str(active_root), "progress": progress}, sort_keys=True)
        if key != last_key:
            last_key = key
            last_change = time.time()
            stalled = False
            bucket = completed // 30
            if notifier.enabled and (stage != last_stage or bucket > last_bucket or progress.get("status") == "complete"):
                notifier.send(
                    "LFBS repaired lineage progress",
                    f"Root={active_root.name}\nStage={stage}\nSymbols={completed}/{progress.get('symbols_planned', '?')}\n"
                    f"Raw={progress.get('raw_signals', 0)}\nEligible={progress.get('eligible_definition_rows', 0)}\n"
                    f"Accepted={progress.get('accepted_rows', progress.get('accepted_trades', 0))}\nSkips={progress.get('overlap_skips', progress.get('actual_overlap_skips', 0))}\n"
                    f"RSS={rss(args.pid)/1024**3:.2f} GiB\nElapsed={progress.get('elapsed_seconds', 0):.0f}s",
                )
            last_stage = stage
            last_bucket = max(last_bucket, bucket)
        current_rss = rss(args.pid)
        for threshold in (6, 8):
            if current_rss >= threshold * 1024**3 and threshold not in rss_alerts:
                rss_alerts.add(threshold)
                if notifier.enabled:
                    notifier.send("LFBS repaired lineage RSS alert", f"RSS crossed {threshold} GiB\nPID={args.pid}\nRoot={active_root}")
        if time.time() - last_change >= 3600 and not stalled:
            stalled = True
            if notifier.enabled:
                notifier.send("LFBS repaired lineage stalled", f"No heartbeat change for 60 minutes\nPID={args.pid}\nRoot={active_root}\nStage={stage}")
        (state_dir / "status.json").write_text(json.dumps({
            "pid": args.pid, "telegram_enabled": notifier.enabled, "active_root": str(active_root),
            "rss_bytes": current_rss, "seconds_since_progress": time.time()-last_change,
            "progress": progress, "updated_ts": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n")
        time.sleep(max(30, args.interval_seconds))
    decision_path = args.canonical_root / "decision_summary.json"
    if decision_path.exists():
        payload = json.loads(decision_path.read_text())
        title = "LFBS repaired lineage complete" if payload.get("status") == "complete" else "LFBS repaired lineage blocked"
        body = f"Status={payload.get('status')}\nClassification={payload.get('classification')}\nEvents={payload.get('canonical_events')}\nRoot={args.canonical_root}"
    else:
        title = "LFBS repaired lineage failed or interrupted"
        body = f"PID exited without canonical decision\nScreen={args.screen_root}"
    if notifier.enabled:
        notifier.send(title, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
