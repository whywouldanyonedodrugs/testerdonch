# QLMG Perp Data Contract

Primary venue: Bybit linear USDT perpetuals. Binance USDⓈ-M data may be secondary research support.

Required semantics:
- all features must be known at decision_ts;
- higher-timeframe bars use last-closed bars only;
- sidecar mark/index/premium/LSR source_close_ts must be <= decision_ts and fresh;
- no last-price substitution for mark/index;
- no default premium=0 or LSR=1;
- listing/delisting/status must be handled point-in-time when available, otherwise marked proxy/unknown;
- OHLCV-only is screening, not final selection.
