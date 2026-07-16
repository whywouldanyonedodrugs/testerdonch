# QLMG Perp Backtesting Manual

Status: active program documentation as of 2026-06-24 UTC.

The active research program is Qullamaggie-inspired crypto perpetual research across long and short Bybit USDT linear perpetuals. Donchian/V3/S1 work is legacy reference only.

Core rules:
- long and short backtests are required;
- mark, index, last, fill, and liquidation prices are different objects;
- OHLCV-only results are screening only;
- final promotion requires funding cashflow, exchange fees, spread/slippage, symbol lifecycle, delisting, and mark-price liquidation modeling;
- small-cap and lower-liquidity symbols may be strategically valuable for small accounts, but only with conservative execution modeling;
- no live trading is authorized by this document.
