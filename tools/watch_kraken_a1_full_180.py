#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
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
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    load_telegram_env_files()
    class A:
        tg_bot_token = ""; tg_chat_id = ""; tg_auto_chat = False; disable_telegram = False; telegram_dry_run = False
    notifier = TelegramNotifier.from_args(A(), run_label="a1-full-180-watcher")
    started = time.time()
    warned_6 = False
    while Path(f"/proc/{args.pid}").exists():
        now = time.time()
        rss = rss_bytes(args.pid)
        manifests = list((root / "aggregate_shards").glob("*/shard_manifest.json"))
        heartbeat_files = list((root / "performance").glob("*heartbeat*.json"))
        newest_shard = max([path.stat().st_mtime for path in manifests], default=started)
        newest_heartbeat = max([path.stat().st_mtime for path in heartbeat_files], default=started)
        payload = {"pid": args.pid, "run_root": str(root), "rss_bytes": rss, "verified_shard_files": len(manifests), "seconds_since_shard_output": now - newest_shard, "seconds_since_heartbeat": now - newest_heartbeat, "status": "running", "ts_utc": utc_now()}
        (root / "watcher").mkdir(parents=True, exist_ok=True)
        (root / "watcher/status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if rss >= 6 * 1024**3 and not warned_6:
            warned_6 = True
            if notifier.enabled: notifier.send("A1 full scan RSS warning", f"RSS={rss/1024**3:.2f} GiB; root={root}")
        stop_reason = ""
        if rss >= int(7.8 * 1024**3):
            stop_reason = f"RSS hard guard reached {rss/1024**3:.2f} GiB"
        elif now - newest_shard >= 3600 and now - newest_heartbeat >= 3600:
            stop_reason = "no shard output and no heartbeat for 60 minutes"
        if stop_reason:
            if notifier.enabled: notifier.send("A1 full scan watcher stopping process", f"{stop_reason}; pid={args.pid}; root={root}")
            try: os.kill(args.pid, signal.SIGINT)
            except ProcessLookupError: pass
            payload.update({"status": "stop_requested", "stop_reason": stop_reason, "ts_utc": utc_now()})
            (root / "watcher/status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            return 2
        time.sleep(300)
    (root / "watcher/status.json").write_text(json.dumps({"pid": args.pid, "run_root": str(root), "status": "process_exited", "ts_utc": utc_now()}, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
