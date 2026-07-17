# C02 Spot and PF Alignment Contract

Status: frozen non-economic data contract. Panel: `Kraken_USD_spot_bar_existence_panel`.

## Identity

A row may align only a Kraken PF contract with the same canonical asset's official Kraken `USD` spot pair. `USDT`, another venue, an index, and synthetic substitution are prohibited. `BTC` PF identity maps explicitly to Kraken `XBT`; other aliases remain unresolved unless the authority table records them. The 2026-07-17 AssetPairs snapshot establishes current identity only. Historical authority exists only at timestamps where an official archive trade produced a bar.

## Time and availability

Normalized spot rows are sparse UTC five-minute intervals keyed by interval-open `timestamp`. `source_close_ts = timestamp + 5 minutes` and `feature_available_ts = source_close_ts`. A rankable join requires `spot.source_close_ts <= decision_ts`, `spot.feature_available_ts <= decision_ts`, and the equivalent PF availability constraints. Join keys are canonical asset plus the exact five-minute interval; inputs must be sorted first.

No spot interval is synthesized, forward-filled, backfilled, interpolated, or inferred from a current listing. C02 generation must use the exact intersection of completed PF and observed spot bars. Gap-mask rows describe only missing intervals between a pair's first and last observed bars. Time before first observation, after last observation, and current-status history remain unknown rather than tradable or missing-at-random.

## Source semantics

Official downloadable time-and-sales rows contain Unix timestamp, executed spot price, and base-asset volume. Deterministic bars use first/max/min/last price and summed base volume. Exact duplicate timestamp/price/volume tuples are disclosed and retained because the source has no trade ID and identical executions may be distinct. No lifecycle continuity is inferred from row presence beyond that observed timestamp.

## Boundaries and exclusions

Only `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)` is admissible. Archive bounds are validated before ZIP member reads; Q1 2026 was listed as metadata only and not downloaded or opened. Mixed or protected price payloads fail closed. A missing, ambiguous, non-USD, unit-ambiguous, or row-empty mapping is excluded with a recorded reason.

## Scope

This contract authorizes a later non-economic C02 generator contract only. It defines no lead/lag label, horizon, threshold, score, rank, candidate, return, or economic conclusion. The panel is not survivorship-free because complete historical listing/status intervals are not independently proven.
