# Validation

Status: **pass**.

- Official calibration: 4 symbols, 3 completed five-minute intervals each, 12/12 exact public-execution quantity sums equal candle volume; no truncation, duplicate UID, wrong symbol, or boundary row.
- Historical semantics: 450 PF symbols observed; 444 have consistent observed base/min-lot semantics. Six inconsistent symbols fail closed.
- Liquidity claim: only `close_based_usd_volume_proxy`; exact quote volume, capacity, spread, depth, and executable liquidity are not claimed.
- Stage 2B lineage: immutable tape SHA-256 `f273b6cc851341b2e3fdd49baf6ed48b39bd6a34f475b7bf129774d0df0a1efd`; feature hash `c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb`.
- Onset reconciliation: 1,805,966 active rows -> 65,738 raw causal onsets -> 38,347 accepted and 27,391 cohort exclusions.
- Identity: 38,347 unique event IDs, 38,347 unique Stage 2B source IDs, 11,343 canonical episodes.
- PIT gate: every accepted event has rank <=100 and at least 20 valid prior UTC days; future-day fixture is inert.
- Bounds: all decisions in `[2023-01-01,2026-01-01)`; protected, post-onset, and economic rows read are zero.
- Attempts: 12/12 retained. Primary and BTC-only robustness roles remain fixed.
- Focused tests: 9 passed. Broad relevant suite: 58 passed, 0 failed, 0 errors (the broad suite includes the 9 focused tests).

Machine evidence: `VALIDATION_EVIDENCE.json`.
