# U2 Coverage And Claim Boundary

## Decision

No contract is admitted to `kraken_u2_anchor_v1_20260716`. BTC and ETH identity and opening dates are high-confidence, and both were tradeable in all ten archived official Kraken instrument snapshots plus the current official snapshot. That evidence does not establish uninterrupted lifecycle status between sparse checkpoints; the longest archived gap is approximately 15 months. Unknown lifecycle state is not imputed active.

## Coverage boundary

- Requested interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Local official trade/mark metadata coverage: `[2023-01-01T00:00:00Z, 2025-12-31T00:00:00Z)` for both considered contracts, with no manifest interval gaps.
- Uncovered rankable tail: `[2025-12-31T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected outcome payloads opened: **zero**.
- Economic outputs computed: **zero**.

## Permitted claim

The artifacts establish stable Kraken identity, official March 2022 opening dates, tradeable status at the archived checkpoints, and local trade/mark metadata coverage through the stated exclusive end. They do **not** establish continuous historical eligibility, absence of temporary suspension/wind-down state, or a rankable U2 cohort.

## C01 boundary

C01 is not authorized because the U2 cohort is empty. The next task is a bounded lifecycle repair, not feature implementation or economic research.
