#!/usr/bin/env bash
set -euo pipefail

SESSION="qlmg_d4_survival"
RUN_ROOT=""
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
    *)
      ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  BASE="/opt/testerdonch/results/rebaseline/phase_qlmg_d4_survivability_redesign_20260625_v1"
  if [[ -d "$BASE" ]]; then
    RUN_ROOT="${BASE}_$(date -u +%Y%m%d_%H%M%S)"
  else
    RUN_ROOT="$BASE"
  fi
  ARGS+=("--run-root" "$RUN_ROOT")
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmux" "$RUN_ROOT/notifications"
cat > "$RUN_ROOT/tmux/watch_commands.md" <<EOF
# Watch Commands

tmux attach -t $SESSION

tail -f $RUN_ROOT/logs/full_run.log

watch -n 30 'cat $RUN_ROOT/watch_status.json'

tail -f $RUN_ROOT/notifications/telegram_events.jsonl

df -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h
EOF

FULL_ARGS_Q=""
for arg in "${ARGS[@]}"; do
  FULL_ARGS_Q+=" $(printf '%q' "$arg")"
done

ENTRYPOINT="$RUN_ROOT/logs/tmux_entrypoint.sh"
cat > "$ENTRYPOINT" <<EOF
#!/usr/bin/env bash
set -o pipefail
cd /opt/testerdonch || exit 1
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/notifications"
echo '{"status":"starting","run_root":"$RUN_ROOT","session":"$SESSION"}' > "$RUN_ROOT/watch_status.json"
if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  ./.venv/bin/python tools/run_qlmg_d4_survivability_redesign.py --stage all --smoke --max-symbols 5 --start 2025-01-01 --end 2025-02-15 --disable-telegram --run-root "$RUN_ROOT/smoke" 2>&1 | tee "$RUN_ROOT/logs/smoke.log"
  smoke_rc=\${PIPESTATUS[0]}
  if [[ \$smoke_rc -ne 0 ]]; then
    echo '{"status":"failed_smoke","run_root":"$RUN_ROOT","session":"$SESSION","exit_code":'\$smoke_rc'}' > "$RUN_ROOT/watch_status.json"
    exit \$smoke_rc
  fi
fi
./.venv/bin/python tools/run_qlmg_d4_survivability_redesign.py$FULL_ARGS_Q 2>&1 | tee "$RUN_ROOT/logs/full_run.log"
rc=\${PIPESTATUS[0]}
if [[ \$rc -eq 0 ]]; then
  echo '{"status":"complete","run_root":"$RUN_ROOT","session":"$SESSION"}' > "$RUN_ROOT/watch_status.json"
else
  echo '{"status":"failed","run_root":"$RUN_ROOT","session":"$SESSION","exit_code":'\$rc'}' > "$RUN_ROOT/watch_status.json"
fi
exit \$rc
EOF
chmod +x "$ENTRYPOINT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "attach: tmux attach -t $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "bash '$ENTRYPOINT'"
echo "tmux session: $SESSION"
echo "run root: $RUN_ROOT"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $RUN_ROOT/logs/full_run.log"
echo "smoke log: tail -f $RUN_ROOT/logs/smoke.log"
