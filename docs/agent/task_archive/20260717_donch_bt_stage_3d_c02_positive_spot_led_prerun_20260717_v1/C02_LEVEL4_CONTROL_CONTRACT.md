# C02 Frozen Level-4 Control Contract

Pre-registered only; execution requires separate approval and a primary Level-3 all-pass result.

Leadership control selects at most one positive coincident/unresolved event with the same PF symbol and calendar year, spot/perp z within 0.5, prior-day rank within 10, causal lagged PF 24h volatility within 20%, a different canonical episode, and at least 24h onset separation. Choose nearest timestamp and break ties by timestamp. Calipers never widen.

Leadership ablation uses the same positive confirmed-impulse generator and execution rules without resolved leadership. Measurement robustness is the frozen 30m-agreement subset and cannot substitute for the primary set. Control identities must freeze before outcomes.
