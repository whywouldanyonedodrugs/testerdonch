# C02 Spot Pilot Validation

Status: `pass`.

Frozen pairs: `['PF_XBTUSD', 'PF_ETHUSD', 'PF_1INCHUSD', 'PF_AAVEUSD']`.

Required non-empty pair/windows: `12`; observed: `12`.

Source identity: official Kraken USD spot time-and-sales. Timestamps are UTC Unix seconds; price is executed spot price; volume is base-asset trade amount. Sparse 5-minute bars are constructed deterministically without gap filling. No row at or after 2026-01-01 is admitted.
