#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [extra args...]"
  echo "Example: $0 jt014b_full_20260216 --jobs 2"
  exit 2
fi

RUN_ID="$1"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="jt014b_${RUN_ID}"
OUT_DIR="${ROOT_DIR}/results/jt014_regime_exits/${RUN_ID}"
LOG_PATH="${OUT_DIR}/_runner.log"

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[jt014b-tmux] session already exists: ${SESSION}"
  echo "[jt014b-tmux] attach: tmux attach -t ${SESSION}"
  echo "[jt014b-tmux] tail:   tail -f ${LOG_PATH}"
  exit 0
fi

CMD=(
  "${ROOT_DIR}/.venv/bin/python"
  "${ROOT_DIR}/tools/run_jt014_regime_conditioned_exits.py"
  "--start" "2023-01-01"
  "--end" "latest"
  "--run-id" "${RUN_ID}"
  "--jobs" "2"
  "--preset-mode" "mechanics"
  "--variant-retries" "2"
  "--policy-probe-mult" "0.25"
  "--no-policy-block-when-down"
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

echo "[jt014b-tmux] started session: ${SESSION}"
echo "[jt014b-tmux] attach: tmux attach -t ${SESSION}"
echo "[jt014b-tmux] log:    tail -f ${LOG_PATH}"
echo "[jt014b-tmux] stop:   tmux kill-session -t ${SESSION}"
