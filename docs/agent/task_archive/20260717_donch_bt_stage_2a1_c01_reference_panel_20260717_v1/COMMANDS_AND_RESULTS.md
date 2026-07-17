# Commands And Results

- Initial branch and task archive created from clean main.
- Verified start: local branch `main` and `origin/main` both at `2e9630646299757a8b5465f7ee1bd63fb5fa5d58`; created `data/stage-2a1-c01-reference-panel-20260717`.
- Four `curl` GETs used only the frozen official paths and query bounds `from=1767139200&to=1767225599`: all HTTP 200.
- One `curl` GET cached the official Kraken derivatives suspensions/delistings page: HTTP 200.
- Initial focused `unittest`: 10 tests, 2 failures. Both were test/parser defects (cross-symbol resumed-note matching and a string-based reader check), repaired before real build.
- First real builder invocation: correctly stopped on the official footnoted symbol `PF_RIVERUSD*`; parser was surgically amended and regression fixture updated.
- Repaired focused `unittest`: 10 tests, 0 failures; after prior-authority lineage enforcement: 11 tests, 0 failures.
- Real authority builder: success. Four normalized slices, each 288 rows; terminal rows parsed: 183.
- Same-directory deterministic builder replay passed. A stronger different-output-directory replay then exposed output-root strings in the source ledger; normalized paths were made archive-relative and the cross-directory replay was repeated.
- Final cross-directory deterministic replay: pass; both core artifact manifests had SHA-256 `894b68c05e14a5d720bb90e30b2e8373f08f0a674054a0a70647873ab3b69ef7` and `diff -qr` found no differences.
- Safe boundary reconciliation opened only the last pre-2026 row of the accepted pre-final-day shards: all four OHLCV rows matched the new `2025-12-31T00:00:00Z` rows exactly.
- Documented `pytest` command was attempted and unavailable (`No module named pytest`); no dependency was installed.
- Final direct `unittest` guard suite: 28 tests, 0 failures.
