# C01 Economic Contract Draft

Status: frozen draft only; no economic run is authorized.

## Entries

- Positive smooth: long at the next executable five-minute trade-bar open after onset.
- Negative smooth: symmetric short diagnostic at the next executable open.
- Positive jump-dominated: within 24h, first completed trade bar closing below the dominant residual bar low; short next executable open.
- Negative jump-dominated: within 24h, first completed trade bar closing above the dominant residual bar high; long next executable open.
- No confirmation means no jump-failure trade. Intermediate events are diagnostic/control only.

The dominant bar must be recomputed from the accepted causal residual component tape and frozen before outcome reads. For jump-dominated onsets its sign alignment is mathematically implied by largest absolute residual share >= 0.5 and nonzero cumulative shock; timestamp and OHLC extreme still require deterministic pre-outcome extraction.

## Exits

- Primary timeout: 6h after entry; robustness timeout: 24h.
- Smooth stop: completed mark close through the opposite six-hour shock-window extreme, executing next trade-bar open.
- Jump-failure stop: completed mark close beyond the dominant jump-bar extreme in the original shock direction, executing next trade-bar open.
- No partial exits, adds, passive/touch fills, leverage optimization, maximum-hold preblocking, or artificial boundary close.

## Costs and funding

- Authority: `/opt/testerdonch/results/rebaseline/phase_kraken_c2_shock_episode_budget_repair_20260713_v1/contract/economic_cost_and_funding_policy_v2.md` with SHA-256 `09054ab7ff7794af3a3c58ecff986d9ce8d4af646319ac08146532d00ae98176`.
- Base: 5 bps taker per side plus 4 bps round-trip slippage.
- Frozen stress: 10 bps taker per side plus 12 bps round-trip slippage.
- Exact funding where available. Imputed funding is a separately capped cost scenario, never signal or promotion evidence.
- Missing execution/depth evidence remains a claim cap.

## Controls and ablations

1. Raw 6h USD-return shock without residualization.
2. Residual shock without path separation.
3. Raw-return path classification at the same causal timestamps.
4. Matched symbol/year/lagged-volatility/parent-return non-events.
5. BTC-only residual robustness.

Continuation and failure branches remain separate. Run Level 3 kill screen first; only a survivor may receive Level 4 controls. Stop if residualization/path adds no increment, one year/symbol/episode dominates, costs remove the result, or threshold changes are required.
