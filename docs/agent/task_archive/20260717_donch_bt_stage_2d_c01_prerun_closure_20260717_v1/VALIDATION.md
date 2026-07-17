# Validation

Status: pass for non-economic pre-run closure.

- Accepted Stage 2C1 generator, draft, feature, cohort, and reference-panel hashes: exact match.
- Accepted onsets/canonical episodes: 38,347 / 11,343.
- Protected outcomes opened: 0. Economic outputs computed: 0.
- Definition register: 16 unique policies; 8 primary and 8 BTC-only robustness; zero-trade retention enabled.
- Boundary rule: all required timestamps must be inside `[2023-01-01, 2026-01-01)`; endpoint and missing timestamps fail closed.
- Non-overlap: definition/symbol-local, chronological, actual executable exits, complete accepted/skip reconciliation.
- Primary units: fixed-notional basis-point returns; no stop-distance sizing.
- Funding partitions: fully exact, mixed, fully imputed, and zero-boundary remain separate.
- Bootstrap: canonical-episode cluster, deterministic seed `20260717`, exactly 10,000 resamples.
- Robustness rescue: BTC-only definitions are ineligible to grant Level-4 permission.
- Matched controls: frozen calipers, deterministic nearest match, no widening.
- Package disposition: all seven gaps remain unavailable or deferred with exact tasks; release-ready remains false.
- Focused tests: 10 passed. Broad relevant suite: 68 passed.
