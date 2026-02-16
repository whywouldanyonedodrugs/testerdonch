# JT-013: Risk-Off Containment Sweep

## Goal
Contain downside in unfavorable regimes by sweeping:
- `REGIME_BLOCK_WHEN_DOWN`
- `REGIME_SIZE_WHEN_DOWN`
- `RISK_OFF_PROBE_MULT`

with fixed guardrail-based selection (not raw PnL ranking).

## Runner
- `tools/run_jt013_risk_off_containment.py`

## KPIs Added
- `worst_month_mean_pnl_R`
- `worst_window_mean_pnl_R`
- `downside_deviation_R`
- `drawdown_tail_p05_R`
- plus `positive_month_ratio`, `max_drawdown_R`, `total_pnl_R`, `number_of_trades`

## Guardrail Logic
Candidate filtering is fixed and deterministic:
1. Hard proxy checks:
- minimum trades vs baseline
- downside deviation bound vs baseline
- drawdown-tail deterioration bound vs baseline
- minimum total PnL fraction vs baseline
2. Soft improvement checks:
- positive month ratio improvement and/or
- worst-window mean PnL_R improvement

Outputs:
- `summary.csv`
- `evaluation.csv`
- `recommendation.json`
- `report.md`

## Example Commands

Smoke:
```bash
./.venv/bin/python tools/run_jt013_risk_off_containment.py \
  --start 2023-01-01 \
  --end 2023-03-31 \
  --jobs 2 \
  --smoke-n 800 \
  --run-id jt013_smoke_20260215_b \
  --size-values 0.2,0.35 \
  --probe-values 0.01,0.05
```

Full:
```bash
./.venv/bin/python tools/run_jt013_risk_off_containment.py \
  --start 2023-01-01 \
  --end latest \
  --jobs 3 \
  --run-id jt013_full_20260215 \
  --size-values 0.2,0.35,0.5 \
  --probe-values 0.01,0.05,0.25
```

`tmux` unattended launcher (recommended):
```bash
export DONCH_TG_BOT_TOKEN='...'
export DONCH_TG_CHAT_ID='...'
tools/run_jt013_tmux.sh jt013_full_20260215 \
  --jobs 3 \
  --size-values 0.2,0.35,0.5 \
  --probe-values 0.01,0.05,0.25
```

Monitoring:
```bash
tmux attach -t jt013_jt013_full_20260215
tail -f results/jt013_risk_off_sweeps/jt013_full_20260215/_runner.log
cat results/jt013_risk_off_sweeps/jt013_full_20260215/_PASS_STATUS.json
```

## Active Run Note
At time of writing, full run `jt013_full_20260215` is active under:
- `results/jt013_risk_off_sweeps/jt013_full_20260215`
