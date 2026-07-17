# Stage 2D C01 Pre-Run Closure Plan

Status: implementation_complete_publication_pending
Owner: backtesting Codex
Created UTC: 2026-07-17
Updated UTC: 2026-07-17
Repository: `/opt/testerdonch` at `45d92488a41fb97a9a30936075c19581f358357d`

## Received task and archive context

- Task specification SHA-256: `579dddfa2c58bdddd2df8d4b4c664795749c9b67ac5f967929acdc5f4d2d4743`.
- Durable archive: `docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/`.
- Accepted C01 source: Stage 2C1 at commit `45d92488a41fb97a9a30936075c19581f358357d` and its recorded generator, feature, cohort, reference-panel, and draft hashes.
- External package source remains immutable: `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/`.
- Approved Drive target: repository default `DONCH_BACKTESTING_HANDOFFS`, non-overwrite only.

## Objective

Produce an outcome-free, approval-ready C01 Level-3 contract and close the external package's ambiguous workflow blocker through truthful claim narrowing while retaining `release_ready=false`.

## Non-goals

- No C01 outcomes, returns, controls, scoring, or ranking.
- No protected-period or capture access.
- No legacy result mutation, missing-payload reconstruction, or economic runner execution.

## Authority and assumptions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Claim narrowing permitted | verified | `QLMG_Next_Research_Plan_2026-07-16_v1.md` | Stop if contradicted by machine contract. |
| Stage 2C1 lineage | verified | accepted hashes and current commit | Fail closed on mismatch. |
| Old package independently reproducible | unavailable | readiness protocol gap list | Keep release-ready false. |
| Later economic command | proposed frozen interface | approval packet | Do not execute; implementation/execution requires separate approval. |

## Files expected to change

- `tools/kraken_c01_prerun_contract.py`: deterministic contract helpers only.
- `tools/build_kraken_c01_prerun_closure.py`: outcome-free artifact builder.
- `unit_tests/test_kraken_c01_prerun_contract.py`: synthetic contract fixtures.
- This task archive: required contracts, disposition, validation, review, manifest, completion, and handoff records.

## Milestones

1. Verify accepted hashes and package gaps; freeze disposition and plan.
2. Implement definition, boundary, non-overlap, arithmetic, funding, concentration, bootstrap, rescue, caliper, and disposition helpers.
3. Generate the 16-definition register and all required human/machine contracts.
4. Run focused and relevant guard tests; review actual diff and artifacts; verify hashes and secrets.
5. Create one commit, fast-forward and non-force push `main`, then upload a unique reduced handoff and round-trip verify all files.

## Risk and rollback

Source roots remain immutable. New work is isolated on `feature/stage-2d-c01-prerun-closure-20260717`. Rollback is branch non-integration; no destructive command is permitted. Remote writes use a new collision-checked folder and immutable upload.

## Progress

- Repository and accepted commit verified clean.
- Package authority supports formal claim narrowing.
- Machine contract, 16-definition register, package disposition, approval packet, validation, and review completed.
- Focused tests passed 10/10; broad relevant tests passed 68/68; manifest and secret checks passed.
- Commit, `main` publication, and Drive round-trip remain the final authorized operational milestone.
- No economic or protected outcome access occurred.
