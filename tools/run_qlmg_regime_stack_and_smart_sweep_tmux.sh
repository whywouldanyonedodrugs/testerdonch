#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RUN_ROOT="$ROOT_DIR/results/rebaseline/phase_qlmg_regime_stack_and_smart_sweep_20260625_v1"
if [[ -n "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$RUN_ROOT"
elif [[ -e "$BASE_RUN_ROOT" ]]; then
  RUN_ROOT="${BASE_RUN_ROOT}_$(date -u +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="$BASE_RUN_ROOT"
fi
SESSION="qlmg_regime_sweep"
SKIP_SMOKE=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux-session-name)
      SESSION="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --skip-smoke)
      SKIP_SMOKE=1; shift ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmux" "$RUN_ROOT/notifications"
LOG="$RUN_ROOT/logs/full_run.log"
cat > "$RUN_ROOT/tmux/watch_commands.md" <<EOF
# Watch Commands

- \`tmux attach -t $SESSION\`
- \`tail -f $RUN_ROOT/logs/full_run.log\`
- \`watch -n 30 'cat $RUN_ROOT/watch_status.json'\`
- \`tail -f $RUN_ROOT/notifications/telegram_events.jsonl\`
- \`df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h\`
EOF
cat > "$RUN_ROOT/watch_status.json" <<EOF
{"status":"launching","run_root":"$RUN_ROOT","ts_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
CMD="cd '$ROOT_DIR'; set -euo pipefail; echo '[tmux] run root: $RUN_ROOT' | tee -a '$LOG'; "
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  CMD+="./.venv/bin/python tools/run_qlmg_regime_stack_and_smart_sweep.py --stage all --smoke --max-symbols 5 --start 2025-01-01 --end 2025-02-15 --sweep-budget 20 --local-refine-budget 6 --run-root '${RUN_ROOT}_smoke' 2>&1 | tee -a '$LOG'; "
fi
CMD+="./.venv/bin/python tools/run_qlmg_regime_stack_and_smart_sweep.py --run-root '$RUN_ROOT' ${ARGS[*]} 2>&1 | tee -a '$LOG'; STATUS=\${PIPESTATUS[0]}; if [[ \$STATUS -eq 0 ]]; then echo '{\"status\":\"complete\",\"run_root\":\"$RUN_ROOT\",\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\"}' > '$RUN_ROOT/watch_status.json'; else echo '{\"status\":\"failed\",\"run_root\":\"$RUN_ROOT\",\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\"}' > '$RUN_ROOT/watch_status.json'; fi; exit \$STATUS"

tmux new-session -d -s "$SESSION" "$CMD"
echo "Launched tmux session: $SESSION"
echo "Run root: $RUN_ROOT"
echo "Watch: tail -f $LOG"
