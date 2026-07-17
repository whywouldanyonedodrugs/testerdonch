# C01 Level-4 Control Contract

This contract is pre-registered only. Controls must not run unless a primary Level-3 definition passes every frozen gate.

Control attempts are: (1) raw 6h USD shock without residualization; (2) residual shock without path separation; (3) raw-return path classification at the same causal timestamps; (4) matched non-events; and (5) BTC-only residual robustness. Each is an additional registered attempt. BTC-only cannot rescue primary failure.

Matched non-events use exactly one control per event with the same symbol, calendar year, and direction. Frozen calipers are lagged 24h volatility within 20%, absolute BTC 6h return within 50 bps, and absolute ETH 6h return within 50 bps. A control cannot lie inside any same-symbol C01 canonical episode and must be at least 48h from event onset. Select the nearest deterministic match, breaking ties by timestamp. If no match exists, mark unavailable. Calipers may not widen after outcomes.

Candidate and control identities must be frozen before any control outcome read. Control adequacy cannot be inferred from a Level-3 result.
