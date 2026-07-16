#!/usr/bin/env bash
set -euo pipefail
SESSION="qlmg_d1_validation"
RUN_ROOT="/opt/testerdonch/results/rebaseline/phase_qlmg_d1_narrow_validation_20260624_v1"
SKIP_SMOKE=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session|--tmux-session-name) SESSION="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/notifications"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
SMOKE_ROOT="${RUN_ROOT}_smoke"
CMD="cd /opt/testerdonch && set -o pipefail; echo '{\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\",\"status\":\"started\",\"run_root\":\"$RUN_ROOT\"}' > '$RUN_ROOT/watch_status.json';"
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  CMD+=" ./.venv/bin/python tools/run_qlmg_d1_narrow_validation.py --stage all --smoke --max-symbols 5 --start 2025-01-01 --end 2025-02-15 --run-root '$SMOKE_ROOT' 2>&1 | tee '$RUN_ROOT/logs/smoke_run.log';"
fi
CMD+=" ./.venv/bin/python tools/run_qlmg_d1_narrow_validation.py --stage all --resume --run-root '$RUN_ROOT' ${EXTRA_ARGS[*]} 2>&1 | tee '$RUN_ROOT/logs/full_run.log'; STATUS=\${PIPESTATUS[0]}; echo '{\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\",\"status\":\"finished\",\"exit_code\":'\$STATUS',\"run_root\":\"$RUN_ROOT\"}' > '$RUN_ROOT/watch_status.json'; exit \$STATUS"
tmux new-session -d -s "$SESSION" "$CMD"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: tail -f $RUN_ROOT/logs/full_run.log"
echo "Status: watch -n 30 'cat $RUN_ROOT/watch_status.json'"
