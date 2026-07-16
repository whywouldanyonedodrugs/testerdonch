#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.telegram_notify import TelegramNotifier, load_telegram_env_files


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def process_state(pid: int) -> tuple[bool, int, str]:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return False, 0, "exited"
    rss, state = 0, "unknown"
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("State:"):
            state = line.split()[1]
    return state != "Z", rss, state


def stage(root: Path) -> str:
    if (root / "decision_summary.json").exists(): return "decision_complete"
    if (root / "controls/symmetric_control_forensics.csv").exists(): return "symmetric_forensics_complete"
    if (root / "controls/control_outcomes.parquet").exists(): return "control_outcomes_complete"
    if (root / "controls/control_key_manifest.csv").exists(): return "control_keys_frozen"
    if (root / "controls/eligible_control_pool_manifest.csv").exists(): return "eligible_pool_complete"
    if (root / "audit/candidate_ledger_reuse_audit.csv").exists(): return "eligible_pool_build"
    return "startup"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    root = Path(args.run_root)
    load_telegram_env_files()
    class A:
        tg_bot_token = ""; tg_chat_id = ""; tg_auto_chat = False; disable_telegram = False; telegram_dry_run = False
    notifier = TelegramNotifier.from_args(A(), run_label="prior-high-control-repair")
    watcher = root / "watcher"; watcher.mkdir(parents=True, exist_ok=True)
    current_stage = stage(root); last_change = time.time(); warned_stall = False; warned_rss = False
    if notifier.enabled:
        notifier.send("Prior-high control repair watcher attached", f"PID={args.pid}\nStage={current_stage}\nRun root={root}")
    while True:
        alive, rss, state = process_state(args.pid)
        observed = stage(root)
        if observed != current_stage:
            current_stage = observed; last_change = time.time(); warned_stall = False
            if notifier.enabled:
                notifier.send("Prior-high control repair progress", f"Stage={current_stage}\nRSS={rss/1024**3:.2f} GiB\nRun root={root}")
        if rss >= 6 * 1024**3 and not warned_rss:
            warned_rss = True
            if notifier.enabled: notifier.send("Prior-high control repair RSS warning", f"RSS={rss/1024**3:.2f} GiB\nPID={args.pid}\nRun root={root}")
        if time.time() - last_change >= 3600 and not warned_stall:
            warned_stall = True
            if notifier.enabled: notifier.send("Prior-high control repair no-progress warning", f"No stage transition for 60 minutes\nStage={current_stage}\nRSS={rss/1024**3:.2f} GiB\nRun root={root}")
        payload = {"pid": args.pid, "run_root": str(root), "telegram_enabled": notifier.enabled, "stage": current_stage, "rss_bytes": rss, "process_state": state, "status": "running" if alive else "exited", "seconds_since_stage_transition": time.time() - last_change, "ts_utc": now()}
        (watcher / "status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if not alive:
            decision = root / "decision_summary.json"
            if decision.exists():
                result = json.loads(decision.read_text()); title = "Prior-high control repair complete" if result.get("status") == "complete" else "Prior-high control repair blocked"
                body = f"Status={result.get('status')}\nControls={result.get('control_outcome_rows')}\nAdvanced={result.get('advanced_candidates')}\nRun root={root}"
            else:
                title = "Prior-high control repair failed or interrupted"; body = f"PID exited without decision_summary.json\nLast stage={current_stage}\nRun root={root}"
            if notifier.enabled: notifier.send(title, body)
            return 0 if decision.exists() else 2
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__": raise SystemExit(main())
