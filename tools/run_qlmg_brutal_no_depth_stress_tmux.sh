#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/testerdonch"
cd "$ROOT"
SESSION="qlmg_brutal_stress"
LAUNCH=0
SKIP_SMOKE=0
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

RUN_ROOT="$($ROOT/.venv/bin/python - <<'PY'
from tools import run_qlmg_brutal_no_depth_stress as mod
args = mod.parse_args([])
root, _ = mod.resolve_run_root(args)
print(root)
PY
)"

if [[ "$LAUNCH" != "1" ]]; then
  echo "Full tmux launch not started. Re-run with --launch-tmux after reviewing smoke/preflight."
  echo "Suggested command: bash tools/run_qlmg_brutal_no_depth_stress_tmux.sh --tmux-session-name ${SESSION} --stage all --resume --include-listing --include-controls --require-telegram --seed 20260628 --launch-tmux"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/logs"
cat > "$RUN_ROOT/watch_status.json" <<JSON
{"status":"launching","session":"$SESSION","run_root":"$RUN_ROOT"}
JSON

FULL_CMD=("$ROOT/.venv/bin/python" "tools/run_qlmg_brutal_no_depth_stress.py" "--run-root" "$RUN_ROOT" "${ARGS[@]}")
SMOKE_CMD=("$ROOT/.venv/bin/python" "tools/run_qlmg_brutal_no_depth_stress.py" "--stage" "all" "--smoke" "--max-symbols" "5" "--start" "2025-01-01" "--end" "2025-02-15" "--disable-telegram")

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
