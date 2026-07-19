# QLMG Kraken Backtesting Manual

Status: active program documentation amended by the 2026-07-19 Stage 10 policy application.

The active research program is QLMG-inspired historical Kraken perpetual backtesting. This manual does not authorize any economic screen; a later task must approve an exact frozen contract before any rankable run.

Core rules:

- long and short research is allowed only when explicitly scoped by an approved task;
- mark, index, last/trade, fill, and liquidation prices are different objects;
- funding is signed notional cashflow at venue settlement boundaries and must not activate signals or gates when imputed;
- OHLCV-only results are screening evidence at most, not final selection or deployment evidence;
- promotion requires venue fees, spread/slippage, lifecycle, listing/delisting/status, funding, mark-price margin/liquidation, and reproducibility manifests;
- small-cap or lower-liquidity symbols require more conservative execution assumptions, not looser assumptions;
- point-in-time universe membership and feature availability are mandatory;
- no live trading, private-account access, order placement, or risk change is authorized by this document.

Rankable work must obey the active interval `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)` and keep `2026-01-01T00:00:00Z` onward sealed from strategy selection, scoring, tuning, validation, controls, and portfolio work.

Future economic contracts must declare, freeze, and serialize before outcomes:

- payoff archetype and intended claim scope;
- hard integrity and economic gates;
- generality, concentration, uncertainty, and execution routing diagnostics;
- control eligibility and exact control authority;
- independent or prospective evidence requirement.

The active gate-routing policy is prospective. Failed historical unconditional gates and terminal decisions remain binding. Concentration may narrow a future claim to an unvalidated conditional route, but cannot retroactively create a pass. Post-hoc contexts are explanatory on the same sample, may not be calendar labels, and require independent evidence before validation. See `docs/agent/TEST_AND_EVIDENCE_STANDARDS.md` and `docs/agent/RESEARCH_GATE_ROUTING_POLICY.json`.
