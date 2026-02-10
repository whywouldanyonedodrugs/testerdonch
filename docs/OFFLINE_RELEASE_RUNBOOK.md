# Offline Release Runbook (Meta Model + Handoff Pack)

This runbook is for the offline/research team.

Goal: produce a deployable, verified bundle and a parity package that the live team can validate against their runtime logic.

## 1) What this runbook produces

Running `tools/run_offline_release_pipeline.py` creates a release folder:

- `results/offline_releases/<release_id>/inputs/trades.clean.csv`
- `results/offline_releases/<release_id>/research_outputs/*` (steps 00-08)
- `results/offline_releases/<release_id>/meta_export/*` (deploy artifacts)
- `results/offline_releases/<release_id>/meta_export/regimes_report.json` (copied from step 02)
- `results/offline_releases/<release_id>/meta_export/golden_features.parquet` (parity fixtures)
- `results/offline_releases/<release_id>/meta_export/bundle_smoke.json` (smoke summary)
- `results/offline_releases/<release_id>/release_manifest.json` (full metadata and paths)
- `results/offline_releases/<release_id>/logs/*.log` (per-stage logs)

## 2) Recommended flow

Use one of these two source modes:

1. `--signals-path ...` mode:
- Runs a corrected source backtest first.
- Optional enrichment.
- Then runs full retrain/export/handoff pipeline.

2. `--trades-in ...` mode:
- Uses an already prepared `trades.clean.csv`.
- Runs full retrain/export/handoff pipeline.

## 3) Command examples

### A) Full source rebuild from corrected signals

```bash
./.venv/bin/python tools/run_offline_release_pipeline.py \
  --release-id 20260209_meta_release_v1 \
  --signals-path results/policy_sweeps/20260207_full_sweep_enrich/_scoped_signals/signals.parquet \
  --start 2023-01-01 \
  --end 2025-11-15 \
  --bt-preset winner_no_meta \
  --funding-stage optional \
  --funding-throttle 0.0 \
  --train-scope ALL \
  --fit-scope ALL \
  --criterion mean \
  --golden-rows 1200
```

### B) Retrain from an existing clean trades file

```bash
./.venv/bin/python tools/run_offline_release_pipeline.py \
  --release-id 20260209_meta_release_v1 \
  --trades-in results/trades.clean.csv \
  --train-scope ALL \
  --fit-scope ALL \
  --criterion mean \
  --golden-rows 1200
```

### C) Dry-run preflight

```bash
./.venv/bin/python tools/run_offline_release_pipeline.py \
  --release-id dryrun_meta_release \
  --trades-in results/trades.clean.csv \
  --dry-run
```

## 4) Safety defaults baked into source backtest preset

Preset `winner_no_meta` applies:

- `META_PROB_THRESHOLD = None`
- `META_GATE_SCOPE = "all"`
- `META_GATE_FAIL_CLOSED = False`
- `REGIME_BLOCK_WHEN_DOWN = False`
- `RISK_OFF_PROBE_MULT = 0.05`
- `REGIME_SLOPE_FILTER_ENABLED = False`
- `BT_META_REPLAY_ENABLED = False`

This protects against replay-store leakage in source generation.

## 5) Required acceptance checks before handoff

1. `release_manifest.json` exists and has no `"error"` field.
2. `meta_export/checksums_sha256.json` exists.
3. `meta_export/deployment_config.json` exists.
4. `meta_export/feature_manifest.json` exists.
5. `meta_export/golden_features.parquet` exists.
6. `meta_export/bundle_smoke.json` exists and reports `is_loaded=true`.

## 6) What to send live team

Send:

1. Bundle directory: `results/offline_releases/<release_id>/meta_export`
2. Release manifest: `results/offline_releases/<release_id>/release_manifest.json`
3. This manual: `docs/LIVE_TEAM_PARITY_MANUAL.md`

Also include the exact release id and decision threshold/scope from:

- `meta_export/deployment_config.json`
- `meta_export/thresholds.json`

## 7) If a stage fails

1. Read the relevant log under `results/offline_releases/<release_id>/logs/`.
2. Fix the root cause.
3. Re-run with `--resume` to skip already completed stages.

