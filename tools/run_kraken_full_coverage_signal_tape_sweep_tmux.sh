#!/usr/bin/env bash
set -euo pipefail

LAUNCH=0
SESSION="kraken_full_coverage_signal_tape"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch-tmux)
      LAUNCH=1
      ARGS+=("$1")
      shift
      ;;
    --tmux-session-name)
      SESSION="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$LAUNCH" != "1" ]]; then
  echo "refusing to launch tmux without --launch-tmux" >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 3
fi

SUPERVISOR="$(mktemp /tmp/kraken_fullcov_supervisor.XXXXXX.sh)"
cat >"$SUPERVISOR" <<'SUPERVISOR_SH'
#!/usr/bin/env bash
set -uo pipefail
cd /opt/testerdonch
START_EPOCH="$(date +%s)"
set +e
./.venv/bin/python tools/run_kraken_full_coverage_signal_tape_sweep.py "$@"
EC="$?"
set -e
if [[ "$EC" != "0" ]]; then
  latest="$(find results/rebaseline -maxdepth 1 -type d -name 'phase_kraken_full_coverage_signal_tape_sweep_20260702_v1*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -n "$latest" && -d "$latest" ]]; then
    OOM_RECENT="false"
    if [[ "$EC" == "137" ]]; then
      OOM_RECENT="true"
    elif dmesg -T 2>/dev/null | tail -120 | grep -Eiq 'oom-kill|Out of memory: Killed process'; then
      OOM_RECENT="true"
    fi
    ./.venv/bin/python - "$latest" "$EC" "$OOM_RECENT" <<'PY'
from __future__ import annotations
import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
exit_code = int(sys.argv[2])
oom_recent = sys.argv[3].lower() == "true"
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
reason = "oom_killed" if oom_recent or exit_code == 137 else "process_exited_nonzero"

def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

watch = read_json(root / "watch_status.json", {})
stage = watch.get("stage", "unknown")
free = shutil.disk_usage(root).free / (1024 ** 3)
payload = {
    "status": "interrupted",
    "reason": reason,
    "exit_code": exit_code,
    "stage": stage,
    "run_root": str(root),
    "host": socket.gethostname(),
    "free_disk_gb": free,
    "ts_utc": ts,
}
write_json(root / "interruptions/interruption_status.json", payload)
write_json(root / "watch_status.json", {"run_root": str(root), "status": "interrupted", "stage": reason, "ts_utc": ts})
write_json(root / "notifications/final_completion_marker.json", payload)
if not (root / "decision_summary.json").exists():
    write_json(root / "decision_summary.json", payload)
(root / "interruptions").mkdir(parents=True, exist_ok=True)
(root / "interruptions/interruption_report.md").write_text(
    "# Interruption Report\n\n"
    f"Status: `interrupted`\n\nReason: `{reason}`\n\nExit code: `{exit_code}`\n\n"
    f"Last recorded stage: `{stage}`\n\nFree disk GB: `{free:.2f}`\n\n"
    "This marker was written by the tmux supervisor after the child process exited.\n"
)

sent = False
err = ""
try:
    from tools.telegram_notify import TelegramNotifier
    class Args:
        tg_bot_token = ""
        tg_chat_id = ""
        tg_auto_chat = False
    notifier = TelegramNotifier.from_args(Args(), run_label="kraken-full-coverage-supervisor")
    sent = notifier.send(
        "Kraken full-coverage sweep interrupted",
        f"reason={reason}\nexit_code={exit_code}\nstage={stage}\nrun_root={root}\nfree_disk_gb={free:.2f}",
    )
except Exception as exc:
    err = f"{type(exc).__name__}: {exc}"
log = {
    "ts_utc": ts,
    "title": "Kraken full-coverage sweep interrupted",
    "body": f"reason={reason} exit_code={exit_code} stage={stage} run_root={root}",
    "level": "error",
    "sent": bool(sent),
    "status": "supervisor",
    "error": err,
}
(root / "notifications").mkdir(parents=True, exist_ok=True)
with (root / "notifications/telegram_events.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(log, sort_keys=True) + "\n")
PY
  fi
fi
exit "$EC"
SUPERVISOR_SH
chmod +x "$SUPERVISOR"
printf -v QUOTED_ARGS ' %q' "${ARGS[@]}"
CMD="cd /opt/testerdonch && $SUPERVISOR${QUOTED_ARGS}"
tmux new-session -d -s "$SESSION" "$CMD"
echo "started tmux session: $SESSION"
echo "watch: tmux attach -t $SESSION"
echo "status: find latest run under results/rebaseline/phase_kraken_full_coverage_signal_tape_sweep_20260702_v1* and cat watch_status.json"
echo "log: tail -f <run_root>/notifications/telegram_events.jsonl"
