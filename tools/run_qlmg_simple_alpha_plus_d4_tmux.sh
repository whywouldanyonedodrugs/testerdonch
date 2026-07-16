#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RUN_ROOT="$ROOT_DIR/results/rebaseline/phase_qlmg_simple_alpha_plus_d4_20260626_v1"
if [[ -n "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$RUN_ROOT"
elif [[ -e "$BASE_RUN_ROOT" ]]; then
  RUN_ROOT="${BASE_RUN_ROOT}_$(date -u +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="$BASE_RUN_ROOT"
fi
SESSION="qlmg_simple_alpha_d4"
SKIP_SMOKE=0
LAUNCH_TMUX=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux-session-name)
      SESSION="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --skip-smoke)
      SKIP_SMOKE=1; shift ;;
    --launch-tmux)
      LAUNCH_TMUX=1; ARGS+=("$1"); shift ;;
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
{"status":"prepared","run_root":"$RUN_ROOT","ts_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","tmux_session":"$SESSION"}
EOF
SMOKE_CMD="cd '$ROOT_DIR' && ./.venv/bin/python tools/run_qlmg_simple_alpha_plus_d4.py --stage all --smoke --max-symbols 5 --start 2025-01-01 --end 2025-02-15 --disable-telegram --discovery-budget 40 --refine-budget 8 --nulls-per-event 1 --d4-carry-forward-required --run-root '${RUN_ROOT}_smoke'"
FULL_CMD="cd '$ROOT_DIR' && ./.venv/bin/python tools/run_qlmg_simple_alpha_plus_d4.py --run-root '$RUN_ROOT' ${ARGS[*]}"
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  echo "[wrapper] running smoke first" | tee -a "$LOG"
  bash -lc "$SMOKE_CMD" 2>&1 | tee -a "$LOG"
fi
if [[ "$LAUNCH_TMUX" -ne 1 ]]; then
  echo "Smoke complete. Full tmux launch not started because --launch-tmux was not provided." | tee -a "$LOG"
  echo "Full command:" | tee -a "$LOG"
  echo "bash tools/run_qlmg_simple_alpha_plus_d4_tmux.sh --tmux-session-name $SESSION --stage all --resume --discovery-budget 2400 --refine-budget 360 --nulls-per-event 3 --use-targeted-1m-if-overlap --require-telegram --d4-carry-forward-required --build-depth-plan --seed 20260626 --launch-tmux" | tee -a "$LOG"
  cat > "$RUN_ROOT/watch_status.json" <<EOF
{"status":"smoke_complete_full_not_launched","run_root":"$RUN_ROOT","ts_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","tmux_session":"$SESSION"}
EOF
  exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
CMD="set -euo pipefail; echo '[tmux] run root: $RUN_ROOT' | tee -a '$LOG'; $FULL_CMD 2>&1 | tee -a '$LOG'; STATUS=\${PIPESTATUS[0]}; if [[ \$STATUS -eq 0 ]]; then echo '{\"status\":\"complete\",\"run_root\":\"$RUN_ROOT\",\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\"}' > '$RUN_ROOT/watch_status.json'; else echo '{\"status\":\"failed\",\"run_root\":\"$RUN_ROOT\",\"ts_utc\":\"'\$(date -u +%Y-%m-%dT%H:%M:%SZ)'\"}' > '$RUN_ROOT/watch_status.json'; fi; exit \$STATUS"
tmux new-session -d -s "$SESSION" "$CMD"
echo "Launched tmux session: $SESSION"
echo "Run root: $RUN_ROOT"
echo "Watch: tail -f $LOG"
