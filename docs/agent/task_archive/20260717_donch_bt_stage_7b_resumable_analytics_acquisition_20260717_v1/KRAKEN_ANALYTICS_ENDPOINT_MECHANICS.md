# Kraken Futures Analytics Endpoint Mechanics

## Verified behavior

- Route: `GET /api/charts/v1/analytics/:symbol/:analytics_type`.
- `since` and `to` are inclusive epoch-second bounds.
- End-exclusive research windows must use `to=end_exclusive-interval`; all corrected-attempt requests had `to < 2026-01-01T00:00:00Z`.
- Maximum observed response size: 2,000 rows.
- `more=true` requires continuation from the last returned timestamp. The continuation lower boundary repeats exactly one row; exact timestamp/value deduplication removed 792 identical boundary rows.
- `more=true` with an empty response occurred for 24 page requests (12 logical open-interest cells across first pass and replay). With no timestamp cursor, the cell terminates as an explicit pagination anomaly.
- 1,080 requests completed: 288 initial cells and 792 continuation pages. No retry, 429, 5xx, unsupported-symbol, or schema error occurred in the corrected attempt.
- All timestamps were aligned to 60 or 300 seconds. Corrected-attempt protected timestamps: zero.
- Exact normalized replay: 144/144 logical cells matched row count, schema hash, and timestamp/value-content hash.

## Retention observations

- Liquidation volume and future basis populated every frozen symbol/window/resolution cell from 2023-01-01 through the final pre-2026 interval.
- Open interest was empty for all six symbols in the 2023-01-01 through 2023-01-08 window, then populated the December 2023, June 2024, and December 2025 windows.
- These are bounded audit observations, not proof of uninterrupted retention outside the frozen windows.

## Quarantined first attempt

The first partial attempt incorrectly passed the end-exclusive `2026-01-01` boundary as inclusive `to`. Timestamp inspection detected the protected boundary and stopped before data values were traversed or normalized. That root is diagnostic-only and preserved unchanged. The corrected attempt uses fresh run and data roots.
