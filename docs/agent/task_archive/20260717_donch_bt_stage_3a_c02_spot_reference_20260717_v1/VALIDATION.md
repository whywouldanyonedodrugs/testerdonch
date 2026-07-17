# Validation

Status: **pass**.

- Source authority: official Kraken downloadable time-and-sales, public Spot Trades pilot, current AssetPairs identity snapshot, and official support documentation are hash-recorded in `C02_SPOT_SOURCE_LEDGER.csv`.
- Pilot: frozen before price read; 4/4 pairs and 12/12 predeclared pair/windows nonempty. Independent reconstruction reproduced every pilot frame exactly.
- Full acquisition: 210 uniquely mapped Stage 2C cohort pairs attempted; 204 had official rows and 6 had none. Forty-three additional cohort symbols lacked a unique official USD identity.
- Full panel: 19,458,116 unique sparse five-minute bars from 234,996,053 official trade rows. Every file is UTC ordered, pair/venue/quote exact, and availability equals interval end.
- Gaps: 5,581,777 internal gap intervals reconcile exactly to 24,834,785 absent slots between observed pair bounds. No interval was filled.
- Boundaries: normalized timestamps are in `[2023-01-01,2026-01-01)`; protected rows opened `0`. The excluded Q1 2026 archive was listed as metadata only and not downloaded or opened.
- Hashes: 204 normalized files, 204 gap masks, 3 raw archives, 12 response/source snapshots, three authority tables, and the manifest self-hash passed independent recomputation.
- Prohibited output scan: no future return, lead/lag, score, threshold, ranking, or candidate-return column. Economic outputs computed `0`.
- Focused tests: 12 passed. Protected/loader/reference guards: 26 passed. Relevant C01/U2 authority regressions: 27 passed. Total: 65 passed, 0 failed, 0 errors.
- Compile: pass for the runner and focused test module.
- Secret review: private credentials `0`; two public client-side identifiers in the exact official support snapshot were reviewed as false positives.

Manifest content hash: `3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046`.
