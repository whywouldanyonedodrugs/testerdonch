# Execution Plan

## Objective

Determine whether Kraken's official public Futures market-analytics endpoint retains reproducible pre-2026 hourly rows for open interest, funding, liquidation volume, and futures basis for PF_XBTUSD and PF_ETHUSD in the three frozen one-day windows.

## Frozen scope

- Starting commit: `f0c12311f31c24c2683d180166450bbacf7389bf`.
- Route: `https://futures.kraken.com/api/charts/v1/analytics/:symbol/:analytics_type`.
- Official types: `open-interest`, `funding`, `liquidation-volume`, `future-basis`.
- Query: epoch-second `since`, `to`, and `interval=3600`.
- Matrix: 2 symbols x 4 types x 3 windows = 24 requests; one exact replay = 48 maximum.
- Download cap: 50 MB total response bodies.
- No retries unless the exact unchanged request has a network failure; maximum two logged retries per request. The implementation defaults to zero retries.

## Non-goals

No full-history acquisition, price or return join, signal, threshold, rank, economics, order-event history, raw trade history, private endpoint, capture, or protected-period analysis.

## Milestones

1. Archive the official documentation page and headers; verify exact route, types, bounds, interval, and schema.
2. Implement a guarded fixed-matrix probe and synthetic tests for bounds, schema, replay, caps, and zero economic fields.
3. Execute exactly 24 core requests and one 24-request replay with deterministic throttling.
4. Classify every cell from returned timestamps and schema; do not infer availability from HTTP status alone.
5. Independently review the request matrix, raw hashes, bounds, replay, protected counts, and capability claims.
6. Update only factual source/capability/readiness/continuity snapshots, commit, non-force push, and round-trip verify the approved compact Drive handoff.

## Failure responses

- If official documentation does not establish an explicit `to` bound, stop before data requests.
- If a response exceeds caps or contains a protected timestamp, fail closed and retain only permitted metadata.
- If only some cells establish history, return `partial_historical_analytics_requires_review`.
- Empty, unsupported, recent-only, ignored bounds, or failed historical cells cannot establish authority.

## Rollback

All work is additive on a task branch and a new ignored result root. Existing roots and decisions remain unchanged. Remote handoff uses a new collision-checked folder and never overwrites or deletes.
