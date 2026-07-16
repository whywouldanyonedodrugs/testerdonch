# QLMG Kraken Data Contract

Primary active venue: Kraken perpetuals only. Bybit rows, Binance rows, and older Donch data are legacy or auxiliary provenance unless a later authority explicitly scopes them as non-rankable support.

This data contract does not authorize an economic run.

Required temporal semantics:

- all features, regime flags, rankings, controls, and risk decisions must be valid strictly as of `decision_ts`;
- for higher-timeframe and sidecar joins, require `source_close_ts <= decision_ts`;
- require `feature_available_ts <= decision_ts` when availability differs from bar close;
- sort explicitly before rolling, grouping, resampling, deduplication, or as-of joins;
- preserve timezone awareness and use UTC consistently.

Required venue and interval gates:

- no non-Kraken row may enter active output;
- no pre-2023 row may enter rankable research;
- no row at or after `2026-01-01T00:00:00Z` may enter strategy scoring, tuning, controls, selection, validation, or portfolio work;
- no instrument may signal before official listing or availability time;
- lifecycle ends, delistings, suspensions, settlements, and missing candles must be explicit. Unknown lifecycle data must be recorded as a cap, not silently filled from a current roster.

Required price and cashflow separation:

- last/trade price governs fills;
- mark price governs unrealized PnL, margin, and liquidation;
- index/oracle price validates anchoring and funding where available;
- execution price is the actual simulated or recorded fill price;
- funding is signed notional cashflow at actual venue settlement boundaries.

Missing required inputs must fail closed or route to an explicit no-trade path. Do not silently zero-fill, forward-fill, backfill, or substitute defaults for required fields unless an active contract says so.
