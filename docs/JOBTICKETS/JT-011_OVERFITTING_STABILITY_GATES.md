# JT-011: Overfitting/Stability Proof Gates

Status: `done`  
Owner: `offline`  
Date: `2026-02-15`

## Implemented

1. Fixed-gate evaluator:
- `tools/evaluate_walkforward_stability.py`

2. Evaluator outputs per run (under `results/walkforward_oos/<run_id>/aggregate/`):
- `stability_verdict.json`
- `stability_executive_summary.md`
- `stability_executive_summary.html`

3. Explicit gate set (non-discretionary):
- minimum windows and months
- median and p90 calibration gap ceilings
- median and p25 top-decile lift floors
- worst-month / worst-window mean PnL bounds
- positive-month ratio and median Spearman floor

## Smoke Validation

Command:

```bash
./.venv/bin/python tools/evaluate_walkforward_stability.py \
  --root results/walkforward_oos \
  --run-id jt007_smoke_20260215_b
```

Result:
- `verdict=fail` (expected on smoke due insufficient history)
- artifact written:
  - `results/walkforward_oos/jt007_smoke_20260215_b/aggregate/stability_verdict.json`

## Full-Horizon Verdict

Scored run:
- `results/walkforward_oos/jt007_full_20260215`

Artifacts:
- `results/walkforward_oos/jt007_full_20260215/aggregate/stability_verdict.json`
- `results/walkforward_oos/jt007_full_20260215/aggregate/stability_executive_summary.md`
- `results/walkforward_oos/jt007_full_20260215/aggregate/stability_executive_summary.html`

Outcome:
- `verdict = fail`
- hard gates: passed
- soft gates failed:
  - `positive_month_ratio`
  - `worst_window_mean_pnl_R`

Interpretation:
- Overfitting-control framework is now operational and deterministic.
- Current strategy quality under defined OOS gates is not sufficient for full-size risk mode.
