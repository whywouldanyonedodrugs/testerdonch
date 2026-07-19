# KDA02 Independent Pre-Run Review

Status: `approved`; independent approval: `true`.

The superseding v5 freeze is approved for the one conditionally authorized exact KDA02 Level-3 run. The preserved pre-price schedule failure did not consume that authorization because execution stopped before any economic input or output was opened. No PF trade-open value, return, funding outcome, protected row, control outcome, KDA01 outcome, KDA02B outcome, or Capital.com outcome was accessed during this review.

## Preserved pre-price failure

- `attempts/preprice_schedule_preflight_failure_v1` records commit `f7a8a63ce41a718a51d3b4168e4e4b61e6603d46`, error `timestamp-only schedule no longer matches mechanical freeze`, `economic_output_root_created: false`, and zero price/funding/protected access.
- The runner at that exact commit independently hashes to the previously approved `dc2d788a389d942e9e0458e43cdf24c56128fdded7962c0eeece960fa043b3ed`. Its executable order is contract/review validation, Kraken authority loading, and `reconstruct_schedule`, followed only on success by output-root creation, accepted-schedule writes, `price_and_score`, and funding loading.
- The recorded error is raised inside `reconstruct_schedule`; therefore that invocation could not reach output-root creation, PF open loading, return computation, or funding loading. A filesystem scan found no KDA02 Level-3 accepted-execution, trade-tape, decision, or run-audit artifact in the worktree or v4 cache.
- This is a zero-outcome schedule-preflight attempt, not the conditionally authorized economic run. The one exact economic authorization remains unused.

## v5 schedule reconciliation repair

- `reconstruct_schedule` now derives the complete frozen primary-definition identity set from the eight-definition register and intersects mechanical gate rows with that set. Deliberately omitted infeasible continuation definitions cannot enter expected economic counts.
- The intersection must contain every frozen primary identity exactly once. Independent synthetic checks confirmed that an extra omitted infeasible gate row is ignored, while a missing frozen gate identity and a duplicated frozen gate identity each fail closed with `frozen primary definition/gate identity mismatch`.
- The repository regression reproduces the actual failure shape: one feasible primary definition, its robustness definition, and an omitted infeasible primary gate row. It confirms the omitted row no longer blocks reconstruction. Together with the independent missing/duplicate boundary checks and the full replay below, coverage is adequate for this surgical repair.

## Independent full schedule replay

- A fresh timestamp-only reconstruction used the frozen eight-definition register, the 5,128-event tape, all eight gate rows, and the manifest-authorized Kraken timestamp authority. It did not read an open-price column.
- Result: 9,274 definition-event records, 8,999 accepted, 275 `actual_position_overlap`, eight definitions, zero duplicate definition-event identities, and zero protected exits.
- Timestamp-authority hash independently reproduced `5fb1e9bbe2497f1d914f8f5d70f148039fa5879eb23b176b92756b057bbab50b`.
- Frozen primary accepted counts matched exactly: negative reversal 1h/6h `2812/2643`; positive reversal 1h/6h `1017/988`.
- Robustness accepted counts were `617/604/159/159` for negative 1h/6h and positive 1h/6h respectively. These are diagnostic-only and cannot rescue primary results.

## Frozen-state and prior-repair verification

- Stage 8A KDA02 event-tape SHA-256 independently matched `c4d553267e2107beeb042bc22f3280013c41a962d1674b4942d2b7f0de5e2b43`. Counts reconciled to `21,241 / 43,946 / 1,176,354 / 3,089 / 7,602 / 0`; all 187 Stage 8A feature-partition hashes matched.
- The v5 Level-3 contract hash independently recomputed to `5ca2ba8b762c4aa06b3c880a68112826764979d4e3f2f555316cece4248d280c`. Feature and generator contract hashes remained valid, and all task-file hashes embedded in code provenance matched current sources.
- Root tapes remain byte-identical to v4: 8,281 parents and 5,128 events. Duplicate parent IDs, event IDs, and candidate economic addresses were `0/0/0`; protected rows were zero; decision offsets, six-hour deadlines, deterministic parent/event identities, and market-day/six-hour cluster identities replayed without mismatch.
- The prior strict-window repair remains effective: irregular interior timestamps fail closed, duplicates raise, alternate timestamp resolution remains valid, and all v4 cache hashes remain intact.
- The retained-material-OI rule remains `current_oi <= onset_oi_close`: onset plus epsilon was rejected, equality accepted, and deeper reduction accepted.
- Cross-horizon funding still routes unique `level3_economic_address` values while preserving the shared candidate address. A two-horizon synthetic replay produced two unique funding-ledger identities without reading funding outcomes.
- The final register contains eight definitions and the feasibility file contains eight primary mechanical gate rows. Costs, clusters, controls, KDA02B separation, protected boundary, and decision vocabulary are unchanged.
- No active KDA02 Level-3 economic artifact exists. Root status continues to report 8,281 parents, 5,128 events, eight definitions, zero protected rows, and zero economic outputs.

## Hash and test evidence

- The exact nine runner-preflight files are pinned in `KDA02_PRERUN_REVIEW.json`; every SHA-256 value was independently recomputed.
- All 750 v4 cache entries in the encompassing manifest matched byte size and SHA-256. The root `ARTIFACT_MANIFEST.json` is an administrative v4 snapshot and is already stale for seven v5/review-administration files, including the superseded review and changed contract. It was not used as v5 authority. The owner must refresh it after this review and before final packaging/handoff; this does not alter any hash-pinned runner input.
- `/opt/testerdonch/.venv/bin/python -m unittest unit_tests.test_kda02_v2 unit_tests.test_kda02_level3`: 28 tests passed.
- Independent missing/duplicate/omitted gate-identity fixtures and the full eight-definition timestamp reconstruction passed.
- Economic run launched: `no`.
- PF trade-open values, economic returns, or funding outcomes inspected: `no`.
- Protected rows or outcomes inspected: `no`.
- Controls, KDA01 outcomes, KDA02B outcomes, or Capital.com outcomes inspected: `no`.

Remaining evidence limits are unchanged: inferred analytics units, price-inferred liquidation side, OI retention truncation, incomplete exact historical funding, and the current-roster/lifecycle-capped universe. Approval covers only the exact hash-pinned v5 runner and contract, without adaptation or expanded outcome authority.
