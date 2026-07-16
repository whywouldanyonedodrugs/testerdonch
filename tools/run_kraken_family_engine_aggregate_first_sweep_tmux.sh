#!/usr/bin/env bash
set -euo pipefail

LAUNCH=0
SESSION="kraken_family_engine_aggregate_sweep"
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

SUPERVISOR="$(mktemp /tmp/kraken_family_engine_supervisor.XXXXXX.sh)"
cat >"$SUPERVISOR" <<'SUPERVISOR_SH'
#!/usr/bin/env bash
set -uo pipefail
cd /opt/testerdonch
set +e
./.venv/bin/python tools/run_kraken_family_engine_aggregate_first_sweep.py "$@"
EC="$?"
set -e
if [[ "$EC" != "0" ]]; then
    latest="$(find results/rebaseline -maxdepth 1 -type d \( -name 'phase_kraken_uncapped_tier1_two_family_sweep_repaired_memorysafe_20260706_v1*' -o -name 'phase_kraken_uncapped_tier1_two_family_sweep_repaired_20260705_v2*' -o -name 'phase_kraken_engine_wave_v0_tranche_tsmom_retest_p1_canary_20260704_v1*' -o -name 'phase_kraken_engine_wave_v0_tranche_tsmom_retest_p1_20260704_v1*' -o -name 'phase_kraken_engine_wave_v0_tranche_tsmom_retest_20260703_v1*' -o -name 'phase_kraken_engine_wave_v0_tranche_20260703_v1*' -o -name 'phase_kraken_aggfirst_vectorized_priority_repair_20260703_v2*' -o -name 'phase_kraken_aggfirst_vectorized_priority_repair_20260703_v1*' -o -name 'phase_kraken_family_engine_aggregate_first_repair_20260702_v1*' -o -name 'phase_kraken_family_engine_aggregate_first_sweep_20260702_v1*' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -n "$latest" && -d "$latest" ]]; then
    ./.venv/bin/python - "$latest" "$EC" <<'PY'
from __future__ import annotations
import json
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1]); exit_code = int(sys.argv[2])
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
payload = {
    "status": "interrupted",
    "reason": "process_exited_nonzero",
    "exit_code": exit_code,
    "run_root": str(root),
    "host": socket.gethostname(),
    "free_disk_gb": shutil.disk_usage(root).free / (1024 ** 3),
    "ts_utc": ts,
}
for rel in ["interruptions/interruption_status.json", "notifications/final_completion_marker.json", "decision_summary.json"]:
    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
(root / "watch_status.json").write_text(json.dumps({"run_root": str(root), "status": "interrupted", "stage": "tmux_supervisor", "ts_utc": ts}, indent=2, sort_keys=True) + "\n")
try:
    from tools.telegram_notify import TelegramNotifier
    class Args:
        tg_bot_token = ""; tg_chat_id = ""; tg_auto_chat = False
    TelegramNotifier.from_args(Args(), run_label="kraken-family-engine-supervisor").send(
        "Kraken aggregate-first sweep interrupted",
        f"exit_code={exit_code}\nrun_root={root}\nfree_disk_gb={payload['free_disk_gb']:.2f}",
    )
except Exception:
    pass
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
echo "status: find latest run under results/rebaseline/phase_kraken_engine_wave_v0_tranche_tsmom_retest_p1_20260704_v1* and cat watch_status.json"
echo "log: tail -f <run_root>/notifications/telegram_events.jsonl"
