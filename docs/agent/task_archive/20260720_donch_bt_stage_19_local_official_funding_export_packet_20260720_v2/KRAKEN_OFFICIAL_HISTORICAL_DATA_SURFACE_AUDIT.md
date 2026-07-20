# Kraken Official Historical Data Surface Audit

This is a metadata-only audit performed on 2026-07-20 from official Kraken support pages, API documentation, official endpoint identities, and the existing Stage 7C acquisition record. No additional bulk market dataset was downloaded.

## Conclusions

Kraken now provides a true bulk derivatives funding export, and Stage 19 has physically isolated its rankable rows. Current official evidence does not establish comparable bulk ZIP exports for OI, future basis, liquidation volume, orderbook, spreads, liquidity, slippage, positioning, or other market analytics. An endpoint name is not evidence of historical retention.

Stage 7C already acquired rankable OI, future basis, and liquidation volume at broad five-minute coverage plus bounded one-minute coverage. Reacquiring them has low incremental value. The most useful unacquired analytics are aggressor differential/CVD, long-short positioning, and orderbook/spread/liquidity/slippage, but each first needs a small metadata/schema/retention pilot under separate authorization.

The documented `/derivatives/api/v3/history` surface is recent-history only—about seven days or the latest engine restart—and is not a historical backfill route. Public order events may add mechanics evidence, but neither retention nor completeness is currently proven.

Kraken spot time-and-sales and OHLCVT bulk archives are spot/reference data. They must never be labelled PF derivatives without member-level proof. Donch already has the stronger exact spot time-and-sales reference authority for C02, so separate OHLCVT acquisition is lower priority.

The largest remaining authority risk is point-in-time lifecycle, not another price series: the instruments endpoint is a current snapshot and does not prove historical listing/status continuity. Prospective official snapshot/changelog archiving is the highest-value next data programme.

## PIT and economic-use limits

- Documentation or current endpoint availability does not prove 2023–2025 retention.
- Current instrument rosters cannot be backfilled as historical truth.
- Unknown schemas, units, populations, depth bands, or aggregation windows remain unknown.
- Basis, relative funding analytics, mark, trade, index, and fill prices remain distinct.
- No item in this audit authorizes acquisition, protected access, a new strategy, or an economic screen.
