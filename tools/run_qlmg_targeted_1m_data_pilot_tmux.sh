#!/usr/bin/env bash
set -euo pipefail

SESSION="qlmg_1m_pilot"
RUN_ROOT="/opt/testerdonch/results/rebaseline/phase_qlmg_targeted_1m_data_pilot_20260624_v1"
SKIP_SMOKE=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux-session-name|--session)
      SESSION="$2"; shift 2 ;;
    --run-root)
      RUN_ROOT="$2"; ARGS+=("--run-root" "$2"); shift 2 ;;
    --skip-smoke)
      SKIP_SMOKE=1; shift ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmux" "$RUN_ROOT/notifications"
LOG="$RUN_ROOT/logs/full_run.log"
STATUS="$RUN_ROOT/watch_status.json"

cat > "$RUN_ROOT/tmux/watch_commands.md" <<EOF
# Watch Commands

- \`tmux attach -t $SESSION\`
- \`tail -f $LOG\`
- \`watch -n 30 'cat $STATUS'\`
- \`tail -f $RUN_ROOT/notifications/telegram_events.jsonl\`
- \`df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h\`
EOF

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

CMD="cd /opt/testerdonch && set -euo pipefail; "
CMD+="echo '{\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\",\"status\":\"started\",\"run_root\":\"$RUN_ROOT\"}' > '$STATUS'; "
if [[ "$SKIP_SMOKE" != "1" ]]; then
  CMD+="./.venv/bin/python tools/run_qlmg_targeted_1m_data_pilot.py --stage all --smoke --max-symbols 2 --no-download --run-root '${RUN_ROOT}_smoke' 2>&1 | tee -a '$LOG'; "
fi
CMD+="./.venv/bin/python tools/run_qlmg_targeted_1m_data_pilot.py ${ARGS[*]} --run-root '$RUN_ROOT' 2>&1 | tee -a '$LOG'; "
CMD+="ec=\${PIPESTATUS[0]}; if [[ \$ec -eq 0 ]]; then st=finished; else st=failed; fi; echo '{\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\",\"status\":\"'\$st'\",\"exit_code\":'\$ec',\"run_root\":\"$RUN_ROOT\"}' > '$STATUS'; exit \$ec"

tmux new-session -d -s "$SESSION" "$CMD"
echo "Started tmux session: $SESSION"
echo "Run root: $RUN_ROOT"
echo "Watch: tail -f $LOG"
