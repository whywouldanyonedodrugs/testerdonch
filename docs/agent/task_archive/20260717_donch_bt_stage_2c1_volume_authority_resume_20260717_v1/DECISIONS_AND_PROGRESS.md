# Decisions and Progress

- Verified clean synchronized starting commit `9b4fbe1bcf9e8a0f79fedde758534dc0d7c86611`; created `feature/stage-2c1-volume-authority-resume-20260717`.
- Froze `PF_XBTUSD`, `PF_ETHUSD`, `PF_XRPUSD`, and `PF_AAVEUSD` before calibration. `PF_DOGEUSD` was replaced before requests because official precision showed integer rather than fractional sizing.
- Froze three contiguous completed intervals: 2026-07-17 08:30, 08:35, and 08:40 UTC.
- Archived requested official URLs, current support specifications, current official API documentation snapshots, and archived official contract pages spanning 2023-2025.
- Passed 12/12 exact-decimal public-execution sum versus official trade-candle volume comparisons.
- Classified candle volume as PF base quantity only under versioned symbol semantics. Exact quote volume remains unavailable.
- Resumed Stage 2C using only `close_based_usd_volume_proxy`; no Stage 2B feature, threshold, or attempt changed.
- Emitted 38,347 causal cohort-eligible onset events from 65,738 raw onsets and retained all 12 attempts.
- Read zero post-onset, protected, or economic outcome rows. No economic screen ran.
- Corrected an initial review-only scope overreach in cross-family header inspection; the final rerun inspects only A1, RSBB, and repaired Backside metadata.
