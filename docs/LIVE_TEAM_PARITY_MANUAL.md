# Live Team Parity Manual

This document is for the live-trading team.

Scope: verify that live execution mechanics match offline/backtest semantics for the shipped bundle. The live team is not responsible for retraining; only for correct runtime behavior and parity.

## 1) Inputs you must receive from offline team

You should receive one release package with:

1. `meta_export/` directory containing:
- `model.joblib`
- `feature_manifest.json`
- `calibration.json` (+ `isotonic.joblib` if method is isotonic)
- `thresholds.json`
- `sizing_curve.csv`
- `deployment_config.json`
- `checksums_sha256.json`
- `regimes_report.json`
- `golden_features.parquet`
- `bundle_smoke.json`
2. `release_manifest.json`
3. Release id string (for audit and logs)

Do not deploy from partial artifacts.

## 2) Integrity checks (must pass before parity checks)

1. Verify checksums match for all required files.
2. Verify loader can open bundle in strict mode.
3. Verify model and manifest schema lengths are sane and non-zero.
4. Verify `deployment_config.json` and `thresholds.json` agree on selected decision scope/threshold.

If any mismatch occurs, reject deployment and request re-export.

## 3) Scoring parity checks

Use `golden_features.parquet` from offline package.

Required checks:

1. Load first N rows (recommended N=100+).
2. Score each row through live scorer path (same code path used in production decision loop).
3. Compare live `p_cal` vs golden `p_cal`.
4. Acceptance:
- max absolute error <= 1e-6 for deterministic builds
- if non-deterministic transforms exist, agree tolerance with offline team in writing

If outside tolerance, block deployment.

## 4) Decision parity checks

For each golden row, validate:

1. Scope evaluation result (`scope_ok`) matches offline semantics.
2. Threshold evaluation (`p_cal >= pstar`) matches.
3. Final gate decision (enter/skip) matches.
4. Decision reason codes are consistent (`ok`, `below_pstar`, `scope_fail:*`, etc).

Special case rule:

If effective threshold is disabled (`META_PROB_THRESHOLD=None` / no `pstar`), live must not apply a secondary/fallback probability veto in order-open path.

## 5) Sizing parity checks

Sizing must use the same mechanics as offline decision logic.

Validate the full chain:

1. Base risk selection mode:
- percent mode: `base_risk = equity * RISK_PCT`
- fixed mode: `base_risk = FIXED_RISK_CASH` (or live alias)
2. Meta sizing mapping from probability to multiplier.
3. Regime-based size adjustments.
4. Risk-off probe cap behavior.
5. Min/max clamps.
6. Final quantity conversion and exchange precision/min-notional handling.

Acceptance:

For parity fixtures, `size_mult` should match exactly or within tiny float tolerance.

## 6) Regime parity checks

Daily regime calculations used in live decision context must be past-only.

Required:

1. Daily Markov probabilities must be filtered (not smoothed).
2. As-of timestamp handling must exclude incomplete bars.
3. Reported cycle regime and logged trade regime fields must reference explicit timestamps.

If a truth export ends earlier than validation timestamps, clip comparison to overlap for diagnostics. For production truth generation, missing horizon coverage should fail the job.

## 7) Runtime behavior checks in canary

Before full-size:

1. Run canary at reduced risk.
2. Log per-signal diagnostics:
- `p_raw`, `p_cal`, `pstar`, `scope`, `scope_ok`, `meta_ok`, `decision_reason`
- `size_mult`, final risk amount, final quantity
3. Confirm no hidden veto path appears after decision stage.
4. Confirm no schema fallbacks silently fill missing required columns.

If live logs show unexpected decision drops or fallback thresholding, stop and fix before scaling.

## 8) Go/No-Go checklist

Go only if all are true:

1. Bundle integrity checks pass.
2. Scoring parity against golden fixtures passes.
3. Decision parity passes.
4. Sizing parity passes.
5. Regime parity (filtered + as-of correctness) passes.
6. Canary runtime logs look consistent with offline expectations.

Any failed item is No-Go for full-size.

## 9) Escalation protocol

If parity fails:

1. Freeze deployment state and capture logs/artifact hashes.
2. Send offline team:
- release id
- failing row ids/timestamps/symbols
- expected vs actual `p_cal`, decision, and `size_mult`
- relevant config snapshot
3. Wait for corrected package or explicit offline sign-off on a revised tolerance.

