# KDA02 Independent Pre-Run Review

Status: `blocked`; independent approval: `false`.

The conditional Level-3 economic gate is closed. No PF trade-open value, forward return, funding outcome, protected row, control outcome, KDA01 outcome, KDA02B outcome, or Capital.com payload was read during this review. No economic output was computed.

## Blocking findings

severity: blocking
path and line: `tools/qlmg_kda02_v2.py:77-82,97-105,175-176`
observed behavior: `exact_horizon_mask` and `_contiguous` prove only that the first and last indexed timestamps have the expected elapsed time. They do not prove that every interior row is a distinct five-minute successor. An independent synthetic fixture with timestamps `00:00, 00:04, 00:11, 00:15` was accepted as `exact_contiguous_15m_valid=True` and produced a three-row liquidation sum.
required contract: Every 15-minute feature and every episode slice must use exact contiguous completed five-minute bars and fail closed across gaps, irregular interior timestamps, and duplicates.
consequence: A malformed source grid can enter parent scores, hysteresis, impulse windows, cooldowns, and candidates while being represented as exact. The current cache happened to contain zero such false accepts, but the required fail-closed contract and regression are absent.
smallest repair: Make exact-window validity require each adjacent timestamp across the full requested window to equal five minutes, reject duplicate timestamps, and make episode contiguity use the same strict predicate. Add fixtures for irregular interior timestamps and duplicates, then rebuild the feature/generator hashes and all dependent tapes, counts, gates, definitions, and contract.
verification: Re-run the focused tests; independently scan every rebuilt feature shard for `exact_contiguous_15m_valid & ~strict_adjacent_5m`; require zero; replay parent/event identities and the timestamp schedule.

severity: blocking
path and line: `tools/qlmg_kda02_v2.py:258-259,311-323`
observed behavior: The completed-reversal branch defines “parent-qualified cumulative OI reduction remains material” as only `current_oi < pre_onset_oi`. An independent synthetic fixture with parent OI reset followed by recovery to `999.999999` against pre-onset OI `1000` still emitted a primary reversal.
required contract: The task requires cumulative OI reduction from the frozen pre-onset close to remain material under the parent attempt. “Material” cannot be silently reduced to an arbitrarily small strict inequality.
consequence: Completed-purge reversals can qualify after nearly the entire parent OI reset has disappeared, changing the economic mechanism and the two mechanically feasible branches without an authorized threshold definition.
smallest repair: Before outcomes, explicitly adjudicate and freeze the quantitative meaning of retained material OI reduction for primary and robustness attempts. If the intended rule truly is merely `current_oi < pre_onset_oi`, the task authority must state that this is the complete definition of “material”; otherwise implement the authorized attempt-consistent threshold. Add a regression proving an epsilon residual reduction is accepted or rejected exactly as frozen, then regenerate all dependent hashes and artifacts.
verification: Re-run the synthetic epsilon-recovery boundary test at equality and immediately on both sides of the authorized threshold; replay episode/event counts, feasibility gates, and definition selection.

severity: blocking
path and line: `tools/run_kda02_level3.py:317-321`; `tools/run_kraken_c01_level3_economic.py:397-447`
observed behavior: KDA02 funding attachment passes the event-level `economic_address` into a helper that groups and merges one-to-one on that address. The same event address is intentionally present in both 1h and 6h definitions. An independent two-definition synthetic fixture reproduced `MergeError: Merge keys are not unique in left dataset`.
required contract: Each frozen definition-event execution must retain a unique economic address through funding diagnostics, and the conditionally authorized run must pass all pre-outcome mechanical checks before opening prices.
consequence: The exact authorized runner would open entry/exit prices and then fail during funding attachment, leaving a partially created output root and consuming outcome access without producing the frozen decision.
smallest repair: Feed the unique `level3_economic_address` to funding attachment as its working economic-address key while preserving the pre-exit candidate address in a separate field; restore both identities explicitly afterward. Add a synthetic regression with one event in two horizons and assert distinct funding partitions/ledgers. Prefer completing all non-price preflight checks before creating the final output root.
verification: Run the new duplicate-across-horizons funding test, verify one-to-one attachment by `level3_economic_address`, and run a no-price dry preflight that reaches the outcome-read boundary without creating the economic output root.

## Independent recomputation evidence

- Stage 8A event-tape SHA-256 independently matched `c4d553267e2107beeb042bc22f3280013c41a962d1674b4942d2b7f0de5e2b43`. Semantic, analytics-manifest, cohort, feature, and generator identities matched the frozen authority. KDA02 counts independently reconciled to `21,241 / 43,946 / 1,176,354 / 3,089 / 7,602 / 0`. All 187 Stage 8A feature partition hashes matched.
- Feature-extension, generator, and Level-3 contract content hashes independently recomputed to `2d117dcbe3bb40f261263a53e985c97c5ba1e4b0b80eddaa0f8d06e3c892bd87`, `f69751c1826c7d393dc4c7fa203a7820c6d72f8c63a704a99c15fb0e8a2ab5fc`, and `36b9c5443fecbe09091e987518fbef484f33a003c920c3c616365bdb02df46f4` for the reviewed, blocked state.
- The pre-review artifact manifest contained 18 archive entries and 750 cache entries; byte sizes and SHA-256 values had zero mismatches. All frozen code hashes matched the contract. The review files written here require the owner to refresh the manifest after repair because this reviewer was authorized to edit only these review artifacts.
- Parent/event tapes independently reconciled to 8,281 episodes and 5,905 candidates with zero duplicate episode IDs, event IDs, or candidate economic addresses; zero pre-train/protected timestamps; exact `decision_ts = state_ts + 5m`; zero orphan or post-deadline events; zero more-than-one-per-episode-and-type; and 2,560 preserved episodes without candidates. Deterministic parent, event, market-day, and six-hour cluster identities all replayed with zero mismatches.
- A full timestamp-only reconstruction used 55 event-bearing symbols and replayed 11,810 definition-event schedule rows, 11,448 accepted rows, 362 actual-position-overlap skips, timestamp-authority hash `80a57846d32a192300e16d5f77ac9b29a061c039a66705f3fba8f8ba60846634`, the cached schedule, count matrix, and every feasibility-gate value exactly. The repaired at-or-after decision timestamp and definition-local actual-exit non-overlap logic are otherwise correct.
- The stored mechanical state has two feasible primary branches, both completed-purge reversals, and eight frozen definitions including their corresponding robustness definitions. Costs are frozen at 14/32 bps; funding is diagnostic and excluded from gates; controls are registered but not executed; market-day inference and sensitivity cluster identities match the task.
- `abs(trade_return_15m)` is a defensible outcome-free magnitude definition because direction is separately required from aligned trade/mark returns. Daily maximum normalization for liquidation follows the Stage 8A liquidation-intensity precedent; daily median normalization for OI change and displacement follows the shared normalizer default. The implementation uses only prior UTC days, a 60-calendar-day window, at least 30 valid days, at least 70% of expected days, and finite nonzero MAD. These design choices are recorded as explicit pre-outcome assumptions and are not additional blockers in this review.
- Source/path and schema checks found no active Capital.com, control, KDA01-outcome, KDA02B-outcome, protected-period, or other-family payload in the Stage 9 tapes or cache. KDA02B remains an inactive zero-event lineage.

## Tests and outcome-access statement

- `/opt/testerdonch/.venv/bin/python -m unittest unit_tests.test_kda02_v2 unit_tests.test_kda02_level3`: 24 tests passed.
- Repository `pytest` was unavailable in both the system interpreter and the available virtual environment; the same test modules passed under `unittest`.
- Independent synthetic exact-window, retained-OI-materiality, and cross-horizon funding-address reproducers produced the three blocking behaviors above.
- Economic run launched: `no`.
- PF trade-open values or returns inspected: `no`.
- Funding outcomes inspected: `no`.
- Protected outcomes or rows inspected: `no`.
- Controls, KDA01 outcomes, or KDA02B outcomes inspected: `no`.

Exact next step: repair only the three blocking surfaces, add the named regressions, rebuild the outcome-free freeze into a new/superseding version, and request a fresh independent pre-outcome review. Do not run `tools/run_kda02_level3.py` against this review.
