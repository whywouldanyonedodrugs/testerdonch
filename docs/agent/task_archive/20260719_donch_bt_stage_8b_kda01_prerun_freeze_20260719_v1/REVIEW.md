# Independent Review

Status: approve for human Level-3 run-approval review. No economic execution is authorized.

## Scope reviewed

Reviewed the actual Stage 8B diff, the Stage 8A authorities and KDA01 v1 hash, causal feature extension, parent-state masks, episode/reset/hysteresis state machine, structural trade-and-mark close-through, attempt/branch identities, complete count matrices, feasibility gates, Level-3 definition contract, Level-4 controls, old-family identity overlap, deterministic replay, resource use, and protected/outcome exclusions.

## Findings resolved before approval

1. The first implementation admitted non-bar manifest rows to the OHLC loader. It failed closed on symbol 4; a schema allowlist now skips non-bar rows exactly as Stage 8A does. Three completed shards were reused only after hash validation.
2. Zero-count shards wrote schema-empty Parquet files that DuckDB refused to union. They remain complete evidence shards but are excluded from reducer inputs by their manifest counts.
3. C01/C02 overlap initially resolved relative to the isolated worktree and then used one incorrect timestamp assumption. The final projection uses only allowlisted symbol/timestamp identity columns from the preserved main checkout.
4. Independent semantic review rejected an initial price-progress distribution built from all positive-OI rows. The final contract normalizes price progress only on directionally coherent primary-or-robustness pre-progress parent rows. The earlier cache and outputs are preserved as superseded, non-economic provenance.
5. Final distribution review found 20 parent onsets with non-finite current path efficiency. The final parent mask fails those rows closed; a fresh generator hash and cache were produced.

## Final review conclusions

- Stage 8A v1 remains unchanged provenance, not relabeled economics.
- OI and price-progress normalizations use prior UTC days only; current/future same-day rows cannot modify earlier scores.
- Parent direction requires aligned trade and mark returns, materially positive OI, and directionally coherent material basis.
- Episodes require a 60-minute directional-parent absence reset, use 30-minute hysteresis exit and six-hour maximum, and retain zero-candidate episodes.
- Failure reversal requires completed trade and mark close-through after deterioration and a complete first-hour impulse. Touches, wicks, sign flips, and trade-only crossings do not qualify.
- Primary and robustness attempts remain separate; robustness cannot rescue primary.
- Event IDs, economic addresses, parent episode IDs, and replayed Parquet bytes are deterministic and unique.
- All four primary branches pass the predeclared mechanical gates. This is feasibility only, not evidence of positive returns.
- Seven controls are frozen but unexecuted. No caliper or threshold was selected from outcomes.
- No protected row, candidate return, PnL, MAE/MFE, exit simulation, funding outcome, or control outcome was opened or computed.

## Remaining caps

Inferred analytics semantics, current-roster/lifecycle cap, OI retention beginning in March 2023, no survivorship-free claim, no economic evidence, no funding evidence, and high raw episode count. Exact symbol-time overlaps do not establish causal equivalence or economic dependence.
