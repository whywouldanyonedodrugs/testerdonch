# JT-007: Rolling Walk-Forward OOS Validation Pack

Status: `done`  
Owner: `offline`  
Date: `2026-02-15`

## What Was Implemented

1. New automation runner:
- `tools/run_walkforward_oos_validation.py`

2. Core behavior:
- Builds rolling train/validate windows (month-based).
- Runs `04_models_cv.py` per train window.
- Runs `05_calibration_ev.py` per train window.
- Selects the best model/calibration method per window from Step 05 holdout metrics.
- Scores the next validation window out-of-sample.
- Tracks calibration and rank-lift by window.
- Aggregates all validation windows into monthly OOS stability metrics.
- Publishes Markdown + HTML monthly OOS stability reports.

3. Output artifacts per run:
- `aggregate/window_metrics.csv`
- `aggregate/monthly_metrics_by_window.csv`
- `aggregate/monthly_oos_stability.csv`
- `aggregate/monthly_oos_stability_report.md`
- `aggregate/monthly_oos_stability_report.html`
- `aggregate/all_validation_scored.parquet`

## Smoke Verification

Command run:

```bash
./.venv/bin/python tools/run_walkforward_oos_validation.py \
  --run-id jt007_smoke_20260215_b \
  --outdir results/walkforward_oos \
  --trades results/trades.clean.csv \
  --targets results/walkforward_oos/jt007_smoke_20260215/_base/01_targets/targets.parquet \
  --regimes results/walkforward_oos/jt007_smoke_20260215/_base/02_regimes/regimes.parquet \
  --start 2023-06-01 \
  --end 2024-06-30 \
  --train-months 6 \
  --valid-months 1 \
  --step-months 1 \
  --max-windows 1 \
  --min-train-trades 200 \
  --min-valid-trades 50 \
  --n-splits 3
```

Result:
- `windows=1`
- `windows_ok=1`
- Report generated:
  - `results/walkforward_oos/jt007_smoke_20260215_b/aggregate/monthly_oos_stability_report.md`
  - `results/walkforward_oos/jt007_smoke_20260215_b/aggregate/monthly_oos_stability_report.html`

## Suggested Full Run

```bash
./.venv/bin/python tools/run_walkforward_oos_validation.py \
  --run-id jt007_full_20260215 \
  --outdir results/walkforward_oos \
  --trades results/trades.clean.csv \
  --start 2023-01-01 \
  --end 2025-11-15 \
  --train-months 12 \
  --valid-months 1 \
  --step-months 1 \
  --min-train-trades 1000 \
  --min-valid-trades 100 \
  --n-splits 6 \
  --resume
```

