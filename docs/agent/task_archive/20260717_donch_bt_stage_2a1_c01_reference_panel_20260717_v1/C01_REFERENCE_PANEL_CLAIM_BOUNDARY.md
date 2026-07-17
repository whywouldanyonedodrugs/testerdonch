# C01 Reference Panel Claim Boundary

Reference panel: `kraken_c01_reference_panel_v1`
Panel hash: `2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763`

`PF_XBTUSD` and `PF_ETHUSD` are admitted only as causal market-reference/factor series. They are not a C01 candidate universe and are not evidence of a tradable strategy.

## Allowed

- Use a member as a factor only when every trade/mark row required by the event-time window is present inside `[2023-01-01, 2026-01-01)`.
- Fail closed when a required window is missing or a known terminal/maintenance status invalidates it.
- Cite stable official identity, March 2022 opening date, and absence from the cached cumulative terminal-event table as of access.

## Prohibited

- `continuous_tradeability_claim`: **no**.
- `survivorship_free_claim`: **no**.
- Absence from the terminal table is not a no-outage or no-suspension claim.
- The panel cannot be used as candidate-universe authority, economic evidence, or permission to inspect protected outcomes.

Temporary status-history reconstruction is explicitly deferred; this task does not create a status platform.
