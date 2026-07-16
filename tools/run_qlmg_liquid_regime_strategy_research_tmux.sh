#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/testerdonch"
cd "$ROOT"
SESSION="qlmg_liquid_regime"
LAUNCH=0
SKIP_SMOKE=0
EXPLICIT_RUN_ROOT=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux-session-name)
      SESSION="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --launch-tmux)
      LAUNCH=1
      ARGS+=("$1")
      shift
      ;;
    --run-root)
      EXPLICIT_RUN_ROOT="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$EXPLICIT_RUN_ROOT" ]]; then
  RUN_ROOT="$EXPLICIT_RUN_ROOT"
else
  RUN_ROOT="$($ROOT/.venv/bin/python - <<'PY'
from tools import run_qlmg_liquid_regime_strategy_research as mod
args = mod.parse_args([])
root, _ = mod.resolve_run_root(args)
print(root)
PY
)"
fi

if [[ "$LAUNCH" != "1" ]]; then
  echo "Full tmux launch not started. Re-run with --launch-tmux after reviewing smoke/preflight and proxy terminal status."
  echo "Suggested command: bash tools/run_qlmg_liquid_regime_strategy_research_tmux.sh --tmux-session-name ${SESSION} --stage all --resume --discovery-budget 3600 --refine-budget 600 --top-per-family 40 --nulls-per-event 3 --include-branch-x-status --include-liquid-continuation --include-sector-catalyst --include-risk-off-shorts --aggressive-overlay --require-telegram --seed 20260628 --launch-tmux"
  exit 0
fi

PROXY_STATUS="$($ROOT/.venv/bin/python - <<'PY'
from tools import run_qlmg_liquid_regime_strategy_research as mod
import json
print(json.dumps(mod.proxy_status(), sort_keys=True))
PY
)"
PROXY_TERMINAL="$($ROOT/.venv/bin/python - <<'PY'
from tools import run_qlmg_liquid_regime_strategy_research as mod
print('1' if mod.proxy_status().get('terminal') else '0')
PY
)"
if [[ "$PROXY_TERMINAL" != "1" ]]; then
  echo "Full tmux launch blocked until proxy execution simulation has terminal decision status: $PROXY_STATUS" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/logs"
touch "$RUN_ROOT/logs/smoke_run.log" "$RUN_ROOT/logs/full_run.log"
cat > "$RUN_ROOT/watch_status.json" <<JSON
{"status":"launching","session":"$SESSION","run_root":"$RUN_ROOT"}
JSON

FULL_CMD=("$ROOT/.venv/bin/python" "tools/run_qlmg_liquid_regime_strategy_research.py" "--run-root" "$RUN_ROOT" "${ARGS[@]}")
SMOKE_CMD=("$ROOT/.venv/bin/python" "tools/run_qlmg_liquid_regime_strategy_research.py" "--stage" "all" "--smoke" "--max-symbols" "5" "--start" "2025-01-01" "--end" "2025-02-15" "--disable-telegram" "--discovery-budget" "80" "--refine-budget" "20" "--top-per-family" "5" "--nulls-per-event" "1")

if [[ "$SKIP_SMOKE" == "1" ]]; then
  INNER="set -euo pipefail; echo '[tmux] run root: $RUN_ROOT'; mkdir -p '$RUN_ROOT/logs'; ${FULL_CMD[*]} 2>&1 | tee '$RUN_ROOT/logs/full_run.log'"
else
  INNER="set -euo pipefail; echo '[tmux] run root: $RUN_ROOT'; mkdir -p '$RUN_ROOT/logs'; echo '[tmux] smoke first'; ${SMOKE_CMD[*]} 2>&1 | tee '$RUN_ROOT/logs/smoke_run.log'; echo '[tmux] full run'; ${FULL_CMD[*]} 2>&1 | tee '$RUN_ROOT/logs/full_run.log'"
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && $INNER"
echo "started tmux session: $SESSION"
echo "run root: $RUN_ROOT"
echo "watch: tmux attach -t $SESSION"
echo "log: tail -f $RUN_ROOT/logs/full_run.log"
