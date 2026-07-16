#!/usr/bin/env bash
set -euo pipefail

SESSION="qlmg_f1_g1_short_unblock"
RUN_ROOT="results/rebaseline/phase_qlmg_f1_g1_short_unblock_20260625_v1"
SKIP_SMOKE=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux-session-name)
      SESSION="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --run-root)
      RUN_ROOT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --skip-smoke)
      SKIP_SMOKE=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmux"
LOG="$RUN_ROOT/logs/full_run.log"
WATCH="$RUN_ROOT/watch_status.json"

cat > "$RUN_ROOT/tmux/watch_commands.md" <<EOF
# Watch Commands

\`\`\`bash
tmux attach -t $SESSION
tail -f $RUN_ROOT/logs/full_run.log
watch -n 30 'cat $RUN_ROOT/watch_status.json'
tail -f $RUN_ROOT/notifications/telegram_events.jsonl
df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h
\`\`\`
EOF

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

SMOKE_CMD="./.venv/bin/python tools/run_qlmg_f1_g1_short_unblock.py --stage all --smoke --max-symbols 5 --start 2025-01-01 --end 2025-02-15 --sweep-budget 20 --run-root ${RUN_ROOT}_smoke"
FULL_CMD="./.venv/bin/python tools/run_qlmg_f1_g1_short_unblock.py --stage all --resume --run-root $RUN_ROOT ${ARGS[*]}"

if [[ $SKIP_SMOKE -eq 1 ]]; then
  CMD="set -euo pipefail; echo '{\"status\":\"starting_full\"}' > '$WATCH'; $FULL_CMD 2>&1 | tee -a '$LOG'"
else
  CMD="set -euo pipefail; echo '{\"status\":\"starting_smoke\"}' > '$WATCH'; $SMOKE_CMD 2>&1 | tee -a '$LOG'; echo '{\"status\":\"starting_full\"}' > '$WATCH'; $FULL_CMD 2>&1 | tee -a '$LOG'"
fi

tmux new-session -d -s "$SESSION" "cd /opt/testerdonch && $CMD"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: tail -f $LOG"
