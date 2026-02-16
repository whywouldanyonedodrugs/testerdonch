#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [extra jt014 args...]"
  echo "Example: $0 jt014_full_20260215 --jobs 3 --preset-mode grid --risk-off-sl-values 1.0,1.25,1.5 --risk-off-tp-values 3,4,6 --risk-off-time-values 24,48,72"
  exit 2
fi

RUN_ID="$1"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="jt014_${RUN_ID}"
OUT_DIR="${ROOT_DIR}/results/jt014_regime_exits/${RUN_ID}"
LOG_PATH="${OUT_DIR}/_runner.log"

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[jt014-tmux] session already exists: ${SESSION}"
  echo "[jt014-tmux] attach: tmux attach -t ${SESSION}"
  echo "[jt014-tmux] tail:   tail -f ${LOG_PATH}"
  exit 0
fi

CMD=(
  "${ROOT_DIR}/.venv/bin/python"
  "${ROOT_DIR}/tools/run_jt014_regime_conditioned_exits.py"
  "--start" "2023-01-01"
  "--end" "latest"
  "--run-id" "${RUN_ID}"
  "--jobs" "3"
  "--preset-mode" "compact"
  "--variant-retries" "2"
  "--no-policy-block-when-down"
  "--policy-probe-mult" "0.25"
  "--tg-auto-chat"
)

if [[ -n "${DONCH_TG_BOT_TOKEN:-}" ]]; then
  CMD+=("--tg-bot-token" "${DONCH_TG_BOT_TOKEN}")
fi
if [[ -n "${DONCH_TG_CHAT_ID:-}" ]]; then
  CMD+=("--tg-chat-id" "${DONCH_TG_CHAT_ID}")
fi
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

INNER_CMD="cd '${ROOT_DIR}' && ${CMD[*]} 2>&1 | tee -a '${LOG_PATH}'"
tmux new-session -d -s "${SESSION}" "bash -lc \"${INNER_CMD}\""

echo "[jt014-tmux] started session: ${SESSION}"
echo "[jt014-tmux] attach: tmux attach -t ${SESSION}"
echo "[jt014-tmux] log:    tail -f ${LOG_PATH}"
echo "[jt014-tmux] stop:   tmux kill-session -t ${SESSION}"
