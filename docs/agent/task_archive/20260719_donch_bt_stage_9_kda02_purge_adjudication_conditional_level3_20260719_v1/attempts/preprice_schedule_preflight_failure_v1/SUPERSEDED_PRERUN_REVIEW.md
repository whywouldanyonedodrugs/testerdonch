# KDA02 Independent Pre-Run Review

Status: `approved`; independent approval: `true`.

The superseding v4 outcome-free freeze is approved for the one conditionally authorized exact KDA02 Level-3 run. The preserved `attempts/pre_review_blocked_v1` tree was treated only as superseded provenance. No PF trade-open value, economic return, funding outcome, protected row, control outcome, KDA01 outcome, KDA02B outcome, or Capital.com outcome was opened during this review.

## Prior blocking repairs

### Exact five-minute adjacency — repaired

- `strict_contiguous_mask` now requires every one of the three adjacent transitions used by a 15-minute feature to equal five minutes, and duplicate timestamps raise before feature construction.
- Episode/reset/impulse/cooldown contiguity independently checks every adjacent timestamp and is resolution-independent.
- Independent fixtures confirmed that `00:00, 00:04, 00:11, 00:15` produces no exact feature, a duplicated timestamp raises, and a `datetime64[us, UTC]` five-minute sequence remains valid.
- All 187 v4 feature shards were scanned independently: false exact-window accepts were zero. PF_XBTUSD and PF_ETHUSD feature extensions also replayed exactly from their Stage 8A inputs.

### Retained material OI — repaired and disclosed

- The versioned generator contract freezes retained material OI for reversal as `current_oi <= frozen onset_oi_close`, meaning the full parent-onset reset must remain; equality is the inclusive boundary.
- Independent synthetic boundaries confirmed: onset OI plus epsilon is rejected, equality is accepted, and deeper reduction is accepted.
- All 4,637 frozen reversal events were joined independently to timestamp-matched Stage 8A OI closes without reading prices: missing OI rows were zero and violations of `current_oi <= onset_oi_close` were zero.

### Cross-horizon funding identity and preflight ordering — repaired

- `attach_funding_diagnostics` preserves the candidate address separately and routes the unique definition-event `level3_economic_address` through the shared funding helper.
- An independent two-horizon synthetic fixture retained two distinct funding addresses and preserved the shared candidate identity without a one-to-one merge failure.
- In the runner, independent-review hash checks, frozen event/gate validation, Kraken authority-manifest loading, timestamp-only execution reconstruction, mechanical-count reconciliation, duplicate checks, and actual-exit non-overlap all complete before the output root is created or `price_and_score` can read an open price. Funding values remain part of the later authorized diagnostic outcome phase and were not read here.

## Independent recomputation evidence

- Stage 8A KDA02 event-tape SHA-256 matched `c4d553267e2107beeb042bc22f3280013c41a962d1674b4942d2b7f0de5e2b43`. Semantic, analytics-manifest, cohort, feature, and generator identities matched their frozen authority. Counts reconciled exactly to `21,241 / 43,946 / 1,176,354 / 3,089 / 7,602 / 0`, and all 187 Stage 8A feature-partition hashes matched.
- Feature-extension, generator, and final Level-3 contract hashes independently recomputed to `798b45d05f4a8a19650da593b26e1b193a94593f270d0c53d80c03fffb0bbf1a`, `b2b72377d04ace8069f30722cdcbcb937fd262b0e7a0b741e563acd07fa490f2`, and `b24097a1ef1e1babe8d03c68405410b9587fd2ac821bed0b444b16fcb639b9b8`. Every task-file hash embedded in code provenance matched.
- The v4 artifact manifest contained 53 archive entries and 750 v4 cache entries; pre-review byte-size and SHA-256 mismatches were zero. This review changes only the two authorized review artifacts, so the task owner must refresh the encompassing artifact manifest after review.
- Parent/event tapes independently reconciled to 8,281 episodes and 5,128 candidates. Duplicate parent IDs, event IDs, and candidate economic addresses were `0/0/0`; pre-2023 and protected timestamps were `0/0`; parent/event decision-time offsets, six-hour deadlines, parent/event relationships, per-episode candidate limits, deterministic episode/event IDs, and market-day/six-hour cluster IDs all replayed without mismatch.
- A fresh timestamp-only reconstruction over 54 event-bearing symbols produced 10,256 definition-event records: 9,975 accepted and 281 `actual_position_overlap` skips. It reproduced timestamp-authority hash `5fb1e9bbe2497f1d914f8f5d70f148039fa5879eb23b176b92756b057bbab50b`, the v4 timestamp cache, count matrix, feasibility gates, and definition register exactly.
- The mechanically feasible primary branches are negative and positive completed-purge reversal. The two continuation branches remain registered and mechanically infeasible without adaptation. The final register contains eight definitions: two horizons for each feasible primary branch and their corresponding robustness branches.
- Costs remain frozen at 14/32 bps; funding is diagnostic and excluded from gates; primary inference uses equal-weight market-day clusters; controls are frozen but unexecuted; KDA02B remains a separate zero-event, no-outcome lineage. No sampling, event cap, maximum-hold preblocking, artificial boundary close, or robustness rescue was found.
- Source, schema, and path checks found no active Capital.com, control, KDA01-outcome, KDA02B-outcome, protected-period, or other-family outcome payload in the v4 root or cache.

## Tests and access declaration

- `/opt/testerdonch/.venv/bin/python -m unittest unit_tests.test_kda02_v2 unit_tests.test_kda02_level3`: 27 tests passed.
- Independent exact-window, timestamp-resolution, retained-OI boundary, cross-horizon funding-address, hash, identity, OI-ledger, full schedule, count, gate, definition, and cache checks passed.
- Economic run launched: `no`.
- PF trade-open values or economic returns inspected: `no`.
- Funding outcomes inspected: `no`.
- Protected rows or outcomes inspected: `no`.
- Controls, KDA01 outcomes, KDA02B outcomes, or Capital.com outcomes inspected: `no`.

Remaining evidence limits are unchanged: inferred analytics units, price-inferred liquidation side, OI retention truncation, incomplete exact historical funding, and the current-roster/lifecycle-capped universe. Approval authorizes only the exact hash-pinned conditional run; it does not authorize adaptation, controls, KDA02B outcomes, protected data, or broader claims.
