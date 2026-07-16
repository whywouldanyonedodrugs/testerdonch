#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.telegram_notify import TelegramNotifier, load_telegram_env_files


def rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): return int(line.split()[1]) * 1024
    except Exception: pass
    return 0


def process_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return len(fields) > 2 and fields[2] != "Z"
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--pid", type=int, required=True); p.add_argument("--run-root", required=True); p.add_argument("--interval-seconds", type=int, default=120); args = p.parse_args()
    root = Path(args.run_root); load_telegram_env_files()
    class T: tg_bot_token=""; tg_chat_id=""; tg_auto_chat=False
    notifier = TelegramNotifier.from_args(T(), run_label="kraken-liquid-failed-breakout-short")
    watcher = root / "watcher"; watcher.mkdir(parents=True, exist_ok=True); last_key=""; last_change=time.time(); warned=False; last_symbol_bucket=-1; last_shard=-1; last_stage=""
    if notifier.enabled: notifier.send("Failed-breakout short screen watcher attached", f"PID={args.pid}\nRun root={root}")
    while process_running(args.pid):
        progress = {}
        try: progress=json.loads((root/"watch_status.json").read_text())
        except Exception: pass
        key=f"{progress.get('stage')}:{progress.get('symbols_completed')}:{progress.get('shards_completed')}"
        if key != last_key:
            last_key=key; last_change=time.time(); warned=False
            symbol=int(progress.get('symbols_completed',0) or 0)
            finalized_shards = len(list((root / "aggregate_shards").glob("*/shard_manifest.json")))
            shard=max(int(progress.get('shards_completed',0) or 0), finalized_shards)
            stage=str(progress.get('stage','')); bucket=symbol//30
            notify_progress = stage != last_stage or bucket > last_symbol_bucket or shard > last_shard
            if notifier.enabled and notify_progress: notifier.send("Failed-breakout short screen progress", f"Stage={stage}\nSymbols={symbol}/301\nShards={shard}/8\nSelected={progress.get('selected_keys',0)}\nEvents={progress.get('event_rows',0)}\nRSS={rss(args.pid)/1024**3:.2f} GiB")
            last_symbol_bucket=max(last_symbol_bucket,bucket); last_shard=max(last_shard,shard); last_stage=stage
        if time.time()-last_change >= 3600 and not warned:
            warned=True
            if notifier.enabled: notifier.send("Failed-breakout short screen stalled", f"No progress for 60 minutes\nPID={args.pid}\nRun root={root}")
        status={"pid":args.pid,"telegram_enabled":notifier.enabled,"rss_bytes":rss(args.pid),"seconds_since_progress":time.time()-last_change,"finalized_outcome_shards":len(list((root / "aggregate_shards").glob("*/shard_manifest.json"))),"progress":progress,"updated_ts":datetime.now(timezone.utc).isoformat()}; (watcher/"status.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
        time.sleep(max(30,args.interval_seconds))
    decision=root/"decision_summary.json"
    if decision.exists(): payload=json.loads(decision.read_text()); title="Failed-breakout short screen complete" if payload.get("status")=="complete" else "Failed-breakout short screen ended"; body=f"Status={payload.get('status')}\nDefinitions={payload.get('definitions_evaluated')}\nEvents={payload.get('events')}\nRun root={root}"
    else: title="Failed-breakout short screen failed or interrupted"; body=f"PID exited without decision summary\nRun root={root}"
    if notifier.enabled: notifier.send(title,body)
    return 0


if __name__ == "__main__": raise SystemExit(main())
