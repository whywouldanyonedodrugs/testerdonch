# Stage 1B Remaining Rankable Reader Boundary Closure

Status: complete
Owner: backtesting Codex
Created UTC: 2026-07-16
Repository: `/opt/testerdonch`
Starting commit: `fc1113d61683e41e7cf9aa76b75b87933a70897c`
Branch: `fix/rankable-loader-boundary-20260716`

## Objective

Apply the existing rankable file-authority boundary to `load_symbol_signal_bars()`, `load_symbol_rank_close_window()`, and `a1_load_symbol_bars_window()`, and bind existing Kraken historical bar authority metadata through `data_paths()`.

## Non-goals

- No economic screen, protected payload read, capture access, acquisition, funding redesign, strategy change, lifecycle repair, package repair, C01/C02/C03 implementation, documentation restructuring, or generalized catalog.
- No changes outside the owning runner, focused tests, and this task archive.

## Files expected to change

- `tools/run_kraken_family_engine_aggregate_first_sweep.py`: authority binding and three reader gates only.
- `unit_tests/test_rankable_loader_boundary.py`: synthetic reader/downstream spies and metadata-binding fixture.
- `docs/agent/task_archive/20260716_donch_bt_stage_1b_remaining_loader_boundary_20260716_v1/`: task records.

## Milestones

1. Verify clean required ancestor and unchanged `main`/`origin/main`.
2. Add synthetic failing tests for all three readers and `data_paths()`.
3. Apply the smallest production patch that makes those tests pass.
4. Run focused, owning-module, guard, and compile checks; review the diff independently.
5. Rerun archived first-wave readiness without economic or protected-payload access.
6. If all gates pass and `main`/`origin/main` remain unchanged, create one task commit, fast-forward local `main`, and push only `main` without force.

## Safety and rollback

- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`; Kraken only.
- All payload tests use mocked readers and synthetic frames.
- Rollback is a normal revert of the single Stage 1B commit; no data or result roots are touched.
- Stop before integration if the remote or local main changes unexpectedly.

## Progress

- Preflight: passed at required commit with clean worktree.
- Economic run authorized: no.
- Protected payload access authorized: no.
- Focused tests: 11/11 pass after 13 pre-patch failures/errors demonstrated the defect.
- Owning module: 286/286 pass.
- Repository guards: 9/9 pass.
- Readiness rerun: complete; loader closed, unrelated U2/C01/C02/C03 blockers remain.
