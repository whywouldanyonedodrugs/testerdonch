# C02 Resolution-Aware Generator Contract

Contract version: `c02_resolution_aware_v1_20260717`.

Source Stage 3B contract SHA-256: `25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb`.

The Stage 3B impulse, prior-day scale, threshold, 60-minute reset, exact sparse intersection, Top-100 cohort, lifecycle, and eligibility rules remain unchanged. Events are reconstructed only on exact UTC-aligned spot/PF bars.

For each frozen 15-minute or 30-minute lookback, the first directional `z >= 1.5` crossing is interval-censored to its observed five-minute bar. A crossing requires a consecutive prior bar below threshold.

- `resolved_spot_led`: spot crossing bar open is at least ten minutes before the PF crossing bar open.
- `resolved_perp_led`: PF crossing bar open is at least ten minutes before the spot crossing bar open.
- `coincident_or_unresolved`: same-bar, five-minute separation, missing crossing, sparse predecessor, or any other overlapping/indeterminate case.

The 15-minute lookback is primary. The frozen 30-minute lookback is robustness only. Shifted data cannot generate or select an event. Completed trade-and-mark failure is retained only when the primary state is `resolved_perp_led`.

Feasibility requires at least 100 primary resolved events, at least 20 in each of 2023/2024/2025, and at least 80% agreement on the same leader under the 30-minute lookback. These gates establish mechanical sample sufficiency only and do not imply edge.

No returns after decision, PnL, exits, MAE/MFE, controls, promotion metrics, or economic rankings are part of this contract.
