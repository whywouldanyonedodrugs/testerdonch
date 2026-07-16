#!/usr/bin/env bash
set -euo pipefail

LAUNCH=0
SESSION="kraken_sweep_readiness"
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

printf -v QUOTED_ARGS ' %q' "${ARGS[@]}"
CMD="cd /opt/testerdonch && ./.venv/bin/python tools/run_kraken_hypothesis_sweep_readiness.py${QUOTED_ARGS}"
tmux new-session -d -s "$SESSION" "$CMD"
echo "started tmux session: $SESSION"
echo "watch: tmux attach -t $SESSION"
echo "status: find latest run under results/rebaseline/phase_kraken_hypothesis_sweep_readiness_20260701_v1* and tail notifications/telegram_events.jsonl"
