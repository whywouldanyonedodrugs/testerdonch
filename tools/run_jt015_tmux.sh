#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [extra args...]"
  echo "Example: $0 jt015_full_20260216 --jobs 2 --pstar 0.42"
  exit 2
fi

RUN_ID="$1"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="jt015_${RUN_ID}"
OUT_DIR="${ROOT_DIR}/results/policy_sweeps/${RUN_ID}"
LOG_PATH="${OUT_DIR}/_runner.log"

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[jt015-tmux] session already exists: ${SESSION}"
  echo "[jt015-tmux] attach: tmux attach -t ${SESSION}"
  echo "[jt015-tmux] tail:   tail -f ${LOG_PATH}"
  exit 0
fi

CMD=(
  "${ROOT_DIR}/.venv/bin/python"
  "${ROOT_DIR}/tools/sweep_policy_settings_v2.py"
  "--start" "2023-01-01"
  "--end" "latest"
  "--jobs" "2"
  "--run-id" "${RUN_ID}"
  "--pstar" "0.42"
  "--smoke-n" "0"
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

echo "[jt015-tmux] started session: ${SESSION}"
echo "[jt015-tmux] attach: tmux attach -t ${SESSION}"
echo "[jt015-tmux] log:    tail -f ${LOG_PATH}"
echo "[jt015-tmux] stop:   tmux kill-session -t ${SESSION}"
