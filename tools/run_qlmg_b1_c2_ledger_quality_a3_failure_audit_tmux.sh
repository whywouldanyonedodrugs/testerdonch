#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/testerdonch"
cd "$ROOT"
SESSION="qlmg_b1c2_a3_audit"
LAUNCH=0
SKIP_SMOKE=0
EXPLICIT_RUN_ROOT=""
REQUIRE_TELEGRAM=0
ALLOW_NO_TELEGRAM=0
DISABLE_TELEGRAM=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux-session-name)
      SESSION="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --launch-tmux)
      LAUNCH=1; ARGS+=("$1"); shift ;;
    --run-root)
      EXPLICIT_RUN_ROOT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --skip-smoke)
      SKIP_SMOKE=1; shift ;;
    --require-telegram)
      REQUIRE_TELEGRAM=1; ARGS+=("$1"); shift ;;
    --allow-no-telegram)
      ALLOW_NO_TELEGRAM=1; ARGS+=("$1"); shift ;;
    --disable-telegram)
      DISABLE_TELEGRAM=1; ARGS+=("$1"); shift ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done

if [[ -n "$EXPLICIT_RUN_ROOT" ]]; then
  RUN_ROOT="$EXPLICIT_RUN_ROOT"
else
  RUN_ROOT="$($ROOT/.venv/bin/python - <<'PY'
from tools import run_qlmg_b1_c2_ledger_quality_a3_failure_audit as mod
args = mod.parse_args([])
root, _ = mod.resolve_run_root(args)
print(root)
PY
)"
fi

if [[ "$LAUNCH" != "1" ]]; then
  echo "Full tmux launch not started. Re-run with --launch-tmux after reviewing smoke output."
  echo "Suggested command: bash tools/run_qlmg_b1_c2_ledger_quality_a3_failure_audit_tmux.sh --tmux-session-name ${SESSION} --stage all --resume --include-forensic-audit --include-a3-failure-audit --include-b1-ledger-quality --include-c2-ledger-quality --include-branch-x-capture-request --b1-repair-budget 1500 --c2-repair-budget 1200 --nulls-per-event 3 --top-per-family 60 --require-telegram --seed 20260629 --launch-tmux"
  exit 0
fi

if [[ "$REQUIRE_TELEGRAM" == "1" && "$ALLOW_NO_TELEGRAM" != "1" && "$DISABLE_TELEGRAM" != "1" ]]; then
  "$ROOT/.venv/bin/python" - <<PY
from pathlib import Path
from tools import run_qlmg_b1_c2_ledger_quality_a3_failure_audit as mod
root = Path(r"$RUN_ROOT")
try:
    mod.RunNotifier(root, disabled=False, require_remote=True, allow_no_remote=False)
except Exception as exc:
    raise SystemExit(f"remote Telegram required but unavailable: {exc}")
print("remote Telegram precheck passed")
PY
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

FULL_CMD=("$ROOT/.venv/bin/python" "tools/run_qlmg_b1_c2_ledger_quality_a3_failure_audit.py" "--run-root" "$RUN_ROOT" "${ARGS[@]}")
SMOKE_CMD=("$ROOT/.venv/bin/python" "tools/run_qlmg_b1_c2_ledger_quality_a3_failure_audit.py" "--stage" "all" "--smoke" "--max-symbols" "5" "--start" "2025-01-01" "--end" "2025-02-15" "--disable-telegram" "--b1-repair-budget" "20" "--c2-repair-budget" "20" "--nulls-per-event" "1" "--top-per-family" "5")

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
echo "telegram log: tail -f $RUN_ROOT/notifications/telegram_events.jsonl"
