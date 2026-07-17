# Decisions and Progress

- Starting authority: clean `main` and `origin/main` at C01 closure commit `b4e6c24c7c1f6c54dea931d824383590d087b819`.
- C01 terminal state was verified and preserved: `level3_no_primary_pass_stop`.
- Official downloadable Kraken time-and-sales was selected as historical price/volume authority. Public REST Trades responses were bounded pilot semantics evidence, not the full panel source.
- Pilot selection was frozen before price read: `PF_XBTUSD`, `PF_ETHUSD`, then canonical `PF_1INCHUSD`, `PF_AAVEUSD`.
- Full acquisition proceeded only after independent pilot approval.
- Current AssetPairs supports identity only. Historical authority is row-observed and bounded; unknown lifecycle intervals remain unknown.
- Sparse intervals are not filled. Exact duplicate timestamp/price/volume tuples are reported but retained because no trade ID exists.
- The complete archive is read only inside declared rankable sub-bounds; Q3/Q4 supply non-overlapping tails. Q1 2026 was not downloaded or opened.
- Final authority: 253 cohort symbols; 210 unique USD mappings; 204 with observed archive rows; 6 mapped/no rows; 43 unresolved identities.
- Final status: `ready_for_C02_non_economic_generator_contract`; no economic work authorized.
