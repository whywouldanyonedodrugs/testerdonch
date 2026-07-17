# Final 2025 Day Coverage Validation

Status: **verified complete**.

- Exact official requests: 4/4 HTTP 200.
- Rows: 288 for each of XBT trade, XBT mark, ETH trade, and ETH mark.
- First timestamp: `2025-12-31T00:00:00Z`.
- Last five-minute open: `2025-12-31T23:55:00Z`; coverage end is `2026-01-01T00:00:00Z` exclusive.
- Missing five-minute intervals: 0 for every slice.
- Duplicate or out-of-order timestamps: 0 accepted.
- `more_candles`: false for every slice.
- Returned protected-period rows: 0.
- Previously identified mixed 2025/2026 chunk opened: no.
- Candidate, return, and funding calculations: 0.

The new interval starts at the prior authority's exclusive coverage boundary. It is retained as a separate bounded shard; no protected file was used as fallback.
