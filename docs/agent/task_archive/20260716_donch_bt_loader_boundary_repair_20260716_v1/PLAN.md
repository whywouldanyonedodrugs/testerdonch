# Minimal Rankable-Loader Boundary Repair and Readiness Rerun

Status: complete
Owner: backtesting Codex
Created UTC: 2026-07-16
Updated UTC: 2026-07-16
Repository root and commit: `/opt/testerdonch` at user-authorized `992e7928d0dd948c0bb3f3fc3c74b1095648df1b`

## Received task and archive context

- Exact task specification: `/root/.codex/attachments/8864d3f9-5158-463d-86ad-8fe688faab3a/pasted-text.txt`.
- The task's original expected commit `8cf3e227105fd7626445d27c8caf4c28bccc2ecb` is superseded only by the user's explicit authorization of `992e7928d0dd948c0bb3f3fc3c74b1095648df1b` as the new starting commit.
- Evidence source: `docs/agent/task_archive/20260716_donch_bt_first_wave_readiness_20260716_v1/`.
- Human approval: one task-scoped local commit after all gates pass; no push or merge.
- Durable task archive: `docs/agent/task_archive/20260716_donch_bt_loader_boundary_repair_20260716_v1/`.
- Remote handoff: not authorized.

## Objective

Repair only the demonstrated rankable market-data and funding loader boundary so file-level protected, mixed, calibration, prospective, external/unrankable, and unknown inputs fail before payload reads, while pre-2023 and non-Kraken rows are excluded before the first rankable downstream consumer. Verify with synthetic reader/downstream spies and rerun the archived readiness audit without economic or protected-payload access.

## Non-goals

- No economic screen, return calculation, candidate ranking, validation, capture access, acquisition, or protected-payload inspection.
- No strategy, universe, lifecycle, funding-model, package-protocol, multiplicity, C01/C02/C03, catalog, partition, governance, or documentation restructuring.
- No fixes for readiness findings outside this loader defect.

## Assumptions and unresolved questions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Repository identity | verified | Git root, branch, `HEAD`, `main`, and `origin/main` checked locally | Stop on divergence |
| Starting commit | verified | Explicit user authorization for `992e7928...` | Record override; do not reinterpret other task terms |
| Existing file-purpose hook | unavailable in affected loaders | `load_symbol_bars` and `load_funding` accept path roots only | Add the smallest loader-local metadata contract |
| Real protected payloads needed | false | Task requires synthetic fixtures and spies | Stop if verification would require a real payload |
| Economic run authorization | absent | Task says no | Keep all verification non-economic |

## Authority paths

| Path | Role | Version/hash | Verified current? |
|---|---|---|---|
| `AGENTS.md` | Repository instructions | at authorized commit | yes |
| `docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md` | Boundary authority | at authorized commit | yes |
| `docs/agent/RUN_AND_ARTIFACT_CONTRACT.md` | Archive/run authority | at authorized commit | yes |
| `docs/agent/KNOWN_FAILURE_PATTERNS.md` | Failure authority | at authorized commit | yes |
| `docs/agent/task_archive/20260716_donch_bt_first_wave_readiness_20260716_v1/FIRST_WAVE_READINESS_REPORT.md` | Demonstrated defect | SHA-256 `cdedeeadcb4613151ff0817d1eb9f2d1e17c872d1293e28257a2bf19773a505e` | yes |
| `tools/run_kraken_family_engine_aggregate_first_sweep.py` | Affected market/funding loader | at authorized commit | yes |

## Repository state preservation

- Checkout: `/opt/testerdonch`, branch `fix/rankable-loader-boundary-20260716`.
- Initial staged / unstaged / untracked / conflicted counts: `0 / 0 / 0 / 0`.
- Base, local `main`, and `origin/main`: `992e7928d0dd948c0bb3f3fc3c74b1095648df1b`.
- Safe-isolation rationale: dedicated task branch; no push/merge authorized.

## Scope and boundaries

- Venue: Kraken only.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected cutoff: `2026-01-01T00:00:00Z`.
- Economic run authorized: no.
- Protected outcomes authorized: no.
- Remote writes authorized: no.
- Forbidden: real protected payload reads, economic work, capture/network access, data mutation, unrelated refactor, push, merge, rebase, or history rewrite.

## Files expected to change

- `tools/run_kraken_family_engine_aggregate_first_sweep.py`: smallest loader-local file-authority gate and downstream row exclusion for market/mark/funding inputs.
- `unit_tests/test_rankable_loader_boundary.py`: synthetic metadata/rows plus payload-reader and downstream spies.
- `unit_tests/test_kraken_family_engine_aggregate_first_sweep.py`: declare rankable file authority in the one existing direct funding-loader fixture so it continues to test multi-chunk loading under the repaired contract.
- `docs/agent/task_archive/20260716_donch_bt_loader_boundary_repair_20260716_v1/*`: required task specification, defect evidence, command records, review, readiness rerun, manifest, completion, and next action.

No other production, test, governance, manual, registry, result-root, raw-data, or configuration file is expected to change.

## Milestones

### M1 - Preflight and defect trace

- Action: verify base, archive hashes, and exact loader call paths.
- Acceptance: clean authorized base; archived evidence verified; no real payload opened.
- Failure response: stop with bounded report.

### M2 - Failing synthetic reproduction

- Action: add focused tests before production edits.
- Acceptance: current code demonstrably invokes a synthetic reader for a file whose metadata is mixed/unrankable, or permits prohibited rows downstream.
- Failure response: do not patch production code.

### M3 - Surgical repair

- Action: add a loader-local metadata gate and row boundary filters only.
- Acceptance: invalid file authority fails before read; valid authorized files are read; pre-2023/non-Kraken rows do not reach downstream; market and funding behavior are equivalent.
- Failure response: restore production/test changes and retain failure archive.

### M4 - Verification and independent review

- Action: focused tests, related loader tests, smallest supported non-economic suite, compile, and separate diff review.
- Acceptance: all pass; scope and authority semantics approved; zero real protected reads/economic outputs.
- Failure response: repair only in scope or stop.

### M5 - Readiness rerun and closure

- Action: rerun the archived first-wave readiness task under corrected boundary interpretation; archive evidence; make one local commit only after all gates.
- Acceptance: readiness rerun complete without repairing other gaps; archive manifest valid; local commit created; no push/merge.
- Failure response: preserve branch/archive and report blocker without commit if gates fail.

## Validation commands

| Command | Why required | Expected result | Actual result |
|---|---|---|---|
| `./.venv/bin/python -m unittest -v unit_tests.test_rankable_loader_boundary` | Defect reproduction then focused verification | fail before patch, pass after | 14 pre-patch failures; 8/8 post-patch pass |
| `./.venv/bin/python -m unittest unit_tests.test_kraken_family_engine_aggregate_first_sweep` | Complete owning-module regression | pass | 286/286 pass |
| `./.venv/bin/python -m unittest -v unit_tests.test_project_deep_cleanup_20260624 unit_tests.test_sealed_slice_guard` | Repository-supported non-economic suite | pass | 9/9 pass |
| `./.venv/bin/python -m py_compile tools/run_kraken_family_engine_aggregate_first_sweep.py unit_tests/test_rankable_loader_boundary.py` | Syntax check | pass | pass |

## Risk and rollback

- Primary risk: accidentally impose a stronger pre-open rule on authorized files containing pre-2023/non-Kraken rows. Mitigation: assert row exclusion only at the downstream boundary.
- Primary safety risk: unknown file authority reaching `pd.read_parquet`. Mitigation: explicit fail-closed loader-local contract checked before reader invocation.
- Rollback: revert only task-owned production/test changes on this isolated branch; retain the task archive and failing evidence. No raw data or result roots are touched.

## Decision log

| UTC | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-07-16 | Use `992e7928...` as starting commit | Explicit user authorization | Original task hash is superseded only for base identity |
| 2026-07-16 | Keep authority contract local to affected loader path | No existing loader metadata hook found | No catalog or partition framework |

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|
| 2026-07-16 | M1 | Authorized base and exact loader paths verified; archive hash verification in progress | Preserve task spec/context, then add failing synthetic tests |
| 2026-07-16 | M2 | Synthetic reader/downstream spies reproduced 14 failing assertions | Apply bounded patch |
| 2026-07-16 | M3 | Market/mark/funding helper and row gates implemented | Run full verification |
| 2026-07-16 | M4 | 8 focused, 286 owning-module, and 9 guard tests passed; review approved | Rerun readiness |
| 2026-07-16 | M5 | Rerun complete; overall readiness blocked by three sibling readers and existing protocol/data gaps | Archive and local commit |

## Completion record

- Acceptance criteria met for the bounded repair: yes.
- Readiness rerun completed: yes; status remains blocked.
- Economic runs launched: no.
- Protected outcomes inspected: no.
- Remote writes: no.
- Next task: `Stage_1B_remaining_rankable_reader_boundary_closure`.
