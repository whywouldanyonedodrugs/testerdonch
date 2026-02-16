# JT-014: Regime-Conditioned Exits (TP/SL/Time-Exit Variants)

## Goal
Add and evaluate exit presets where `risk_off` uses tighter/faster exits than `risk_on`.

## Implementation
- Backtester support:
1. `REGIME_COND_EXITS_ENABLED`
2. `RISK_ON_SL_ATR_MULT`, `RISK_ON_TP_ATR_MULT`, `RISK_ON_TIME_EXIT_HOURS`
3. `RISK_OFF_SL_ATR_MULT`, `RISK_OFF_TP_ATR_MULT`, `RISK_OFF_TIME_EXIT_HOURS`

- Runner:
`tools/run_jt014_regime_conditioned_exits.py`

- Unattended launcher:
`tools/run_jt014_tmux.sh`

## Outputs
Per run directory:
`results/jt014_regime_exits/<run_id>/`

Main artifacts:
1. `summary.csv`
2. `evaluation.csv`
3. `sensitivity_matrix.csv`
4. `walkforward_windows.csv`
5. `recommendation.json`
6. `report.md`

## Smoke Command
```bash
./.venv/bin/python tools/run_jt014_regime_conditioned_exits.py \
  --start 2023-01-01 \
  --end 2023-03-31 \
  --smoke-n 1500 \
  --jobs 2 \
  --run-id jt014_smoke_20260215 \
  --preset-mode compact
```

## Full Command (compact presets)
```bash
./.venv/bin/python tools/run_jt014_regime_conditioned_exits.py \
  --start 2023-01-01 \
  --end latest \
  --jobs 3 \
  --run-id jt014_full_20260215 \
  --preset-mode compact \
  --policy-probe-mult 0.25 \
  --no-policy-block-when-down
```

## Full Command (grid sensitivity)
```bash
./.venv/bin/python tools/run_jt014_regime_conditioned_exits.py \
  --start 2023-01-01 \
  --end latest \
  --jobs 3 \
  --run-id jt014_grid_20260215 \
  --preset-mode grid \
  --risk-off-sl-values 1.0,1.25,1.5,2.0 \
  --risk-off-tp-values 3.0,4.0,6.0,8.0 \
  --risk-off-time-values 24,48,72 \
  --policy-probe-mult 0.25 \
  --no-policy-block-when-down
```

## tmux Launcher
```bash
export DONCH_TG_BOT_TOKEN='...'
export DONCH_TG_CHAT_ID='...'
tools/run_jt014_tmux.sh jt014_full_20260215 --preset-mode compact --jobs 3
```

Monitor:
```bash
tmux attach -t jt014_jt014_full_20260215
tail -f results/jt014_regime_exits/jt014_full_20260215/_runner.log
```
