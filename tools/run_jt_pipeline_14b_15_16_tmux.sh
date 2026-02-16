#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pipeline_run_id> [extra args to run_jt_pipeline_14b_15_16.py]"
  echo "Example: $0 jt_seq_20260216 --jobs 2 --jt015-pstar 0.42 --jt016-max-variants 24"
  exit 2
fi

RUN_ID="$1"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="jtseq_${RUN_ID}"
OUT_DIR="${ROOT_DIR}/results/jt_pipeline_runs/${RUN_ID}"
LOG_PATH="${OUT_DIR}/_runner.log"

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[jt-seq-tmux] session already exists: ${SESSION}"
  echo "[jt-seq-tmux] attach: tmux attach -t ${SESSION}"
  echo "[jt-seq-tmux] tail:   tail -f ${LOG_PATH}"
  exit 0
fi

CMD=(
  "${ROOT_DIR}/.venv/bin/python"
  "${ROOT_DIR}/tools/run_jt_pipeline_14b_15_16.py"
  "--start" "2023-01-01"
  "--end" "latest"
  "--jobs" "2"
  "--run-id" "${RUN_ID}"
  "--jt015-pstar" "0.42"
  "--jt016-max-variants" "24"
  "--jt016-top-k" "3"
  "--jt016-scout-workers" "2"
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

echo "[jt-seq-tmux] started session: ${SESSION}"
echo "[jt-seq-tmux] attach: tmux attach -t ${SESSION}"
echo "[jt-seq-tmux] log:    tail -f ${LOG_PATH}"
echo "[jt-seq-tmux] stop:   tmux kill-session -t ${SESSION}"
