# Funding Calibration and Alignment Summary

The campaign uses 5,658,890 official rankable hourly rows from 476 symbols. All 187 campaign symbols have exact export coverage and at least 4,079 rankable observations, so every campaign allowance is its own Decimal Hyndman–Fan type-7 q95/q99; none uses the pooled fallback. The equal-symbol-weighted pool remains frozen for future mechanically eligible symbols and contains 276 official unit-compatible PF symbols.

Every campaign PF is verified as Kraken `flexible_futures`, `contractSize=1`, base-unit quantity, USD quote. Nonzero `absolute_rate/relative_rate` anchors were checked against exact-timestamp rankable trade and mark opens. Mark relative error must be at most 10%; the trade sanity bound is 25% to accommodate sparse launch-period prints. No hidden multiplier is admitted.

Both timestamp interpretations are computed. Selection receives `min(0, signed_start, signed_end)` and therefore no favourable funding credit. Missing hours receive only nonpositive q95/q99 costs. Timestamp-sensitive positives are limitation-tagged and are not robust-funding promotions. Protected funding values contributed zero statistics.
