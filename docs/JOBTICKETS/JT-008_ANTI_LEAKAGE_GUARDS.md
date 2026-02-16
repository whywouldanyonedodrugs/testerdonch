# JT-008: Anti-Leakage Guards (Regime + Time Alignment)

Status: `done`  
Owner: `offline`  
Date: `2026-02-15`

## Scope

Enforce two non-regression guarantees:

1. Regime model probabilities used in production path are past-only:
- `filtered_marginal_probabilities` required
- `smoothed_marginal_probabilities` forbidden in production regime functions

2. `merge_asof` alignment in guarded production files is explicit and safe:
- `direction="backward"` required
- `allow_exact_matches=True` required
- explicit `tolerance` required for meta merge in `backtester.py`

## Implemented Artifacts

1. Guard script:
- `tools/ci_check_leakage_guards.py`

2. Guard tests:
- `unit_tests/test_jt008_leakage_guards.py`

3. Alignment hardening changes:
- `backfill_trade_features.py` (`allow_exact_matches=True` now explicit)
- `pull.py` (`allow_exact_matches=True` now explicit)

## Verification

Commands:

```bash
./.venv/bin/python tools/ci_check_leakage_guards.py --out results/ci_leakage_guards/report.json
./.venv/bin/python -m unittest -q unit_tests.test_jt008_leakage_guards
```

Expected:
- guard exits `0`
- `results/ci_leakage_guards/report.json` has:
  - `"status": "ok"`
  - `"violation_count": 0`

