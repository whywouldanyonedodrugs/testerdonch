#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-after_sidecar_20260416}"
shift || true

SIDECAR_SESSION="${SIDECAR_SESSION:-fader2_bybit_context_backfill_20260416_110304}"
SIDECAR_BACKFILL_ROOT="${SIDECAR_BACKFILL_ROOT:-/opt/fader2/reports/upstream_results/parity_exchange/offline_to_live/2026-04-16_bybit_context_sidecar_backfill_v1_20260416_110304}"
SIDECAR_STATUS_JSON="${SIDECAR_STATUS_JSON:-${SIDECAR_BACKFILL_ROOT}/STATUS.json}"
CONTEXT_ROOT="${CONTEXT_ROOT:-/opt/parquet/bybit_context_5m}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-results/rebaseline/phase_event_crowding_entry_sweep_event_crowding_20260416}"
RESULTS_ROOT="${RESULTS_ROOT:-results/rebaseline}"

RUN_NAME="phase_perp_state_v2_resweep_${RUN_ID}"
RUN_ROOT="${ROOT_DIR}/${RESULTS_ROOT}/${RUN_NAME}"
WATCH_SESSION="${RUN_NAME}_sidecar_watch"
RESWEEP_SESSION="${RUN_NAME}"
WATCH_LOG="${RUN_ROOT}/sidecar_waiter.log"
WATCH_SCRIPT="${RUN_ROOT}/sidecar_waiter_run.sh"

if tmux has-session -t "=${WATCH_SESSION}" 2>/dev/null; then
  echo "[perp-state-v2-after-sidecar] watcher already exists: ${WATCH_SESSION}"
  echo "[perp-state-v2-after-sidecar] attach watcher: tmux attach -t ${WATCH_SESSION}"
  echo "[perp-state-v2-after-sidecar] tail watcher:   tail -f ${WATCH_LOG}"
  echo "[perp-state-v2-after-sidecar] resweep session: ${RESWEEP_SESSION}"
  exit 0
fi

mkdir -p "${RUN_ROOT}"

cat > "${WATCH_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd '${ROOT_DIR}'
LOG_PATH='${WATCH_LOG}'
echo "[perp-state-v2-after-sidecar] watcher start utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "\${LOG_PATH}"
./.venv/bin/python -u tools/watch_perp_state_v2_sidecar_resweep.py \\
  --run-id '${RUN_ID}' \\
  --results-root '${RESULTS_ROOT}' \\
  --context-root '${CONTEXT_ROOT}' \\
  --baseline-run-root '${BASELINE_RUN_ROOT}' \\
  --sidecar-session '${SIDECAR_SESSION}' \\
  --sidecar-backfill-root '${SIDECAR_BACKFILL_ROOT}' \\
  --sidecar-status-json '${SIDECAR_STATUS_JSON}' \\
  --resweep-session '${RESWEEP_SESSION}' \\
  --heartbeat-hours 4 \\
  --poll-seconds 300 \\
  --send-startup \\
  "\$@" 2>&1 | tee -a "\${LOG_PATH}"
RC=\${PIPESTATUS[0]}
echo "[perp-state-v2-after-sidecar] watcher done utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=\${RC}" | tee -a "\${LOG_PATH}"
exit \${RC}
EOF

chmod +x "${WATCH_SCRIPT}"
tmux new-session -d -s "${WATCH_SESSION}" "bash '${WATCH_SCRIPT}' $*"

echo "[perp-state-v2-after-sidecar] watcher session: ${WATCH_SESSION}"
echo "[perp-state-v2-after-sidecar] sidecar session: ${SIDECAR_SESSION}"
echo "[perp-state-v2-after-sidecar] resweep session after preflight: ${RESWEEP_SESSION}"
echo "[perp-state-v2-after-sidecar] run root: ${RUN_ROOT}"
echo "[perp-state-v2-after-sidecar] attach watcher: tmux attach -t ${WATCH_SESSION}"
echo "[perp-state-v2-after-sidecar] tail watcher:   tail -f ${WATCH_LOG}"
echo "[perp-state-v2-after-sidecar] tail resweep:   tail -f ${RUN_ROOT}/runner.log"
echo "[perp-state-v2-after-sidecar] kill watcher:   tmux kill-session -t ${WATCH_SESSION}"
echo "[perp-state-v2-after-sidecar] kill resweep:   tmux kill-session -t ${RESWEEP_SESSION}"
