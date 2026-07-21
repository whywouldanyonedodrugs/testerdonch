# Stage 23 — remediate Stage-22 V04 and publish final packet

Status: `independent_review_block`
Owner: backtesting agent
Created UTC: `2026-07-21T06:25:02Z`
Updated UTC: `2026-07-21T06:25:02Z`
Repository root and commit: `/opt/testerdonch-stage22-20260720` at
`84fda51be77431de7151209d28c810fe027b0133`

## Received task and archive context

- Exact task: `received/STAGE23_TASK.txt`, SHA-256
  `879869d9b3a698600c13322cf222aa0016e5437ec6d2637ed83166e58ced14a4`.
- Stage-22 handoff ZIP SHA-256:
  `bdcdec481d5c0705981a8140a2000340c1dc33c98ee69f087c5da077b8e29124`;
  155 embedded files and manifest SHA-256
  `26ff8196d7eb3de97764751992c91f56567035ec9c69381663d262e4feca9b3b`
  verified.
- Stage-22 independent review SHA-256:
  `7e97143f89c07fed180aa9e3e5d492ab779ccfc58187d3a71b0a27c4ec45b958`.
- Dynamic continuity sequence 6, pointer SHA-256
  `35f5f755822e347a1549e928dcb422f048d7a35de63111af7caefc1e64b2fc3f`,
  and snapshot physical SHA-256
  `e60aa93d5b79d2d82b47782a096f83c4e6f80abb329da84041c029f70fe81bd1`
  verified from the stable Drive ledger.
- Approved Drive target: default `qlmg_sweep_drive:` root, unique
  non-overwriting child and full round-trip verification.

## Objective

Close every finding and subfinding in the Stage-22 V04 independent review in
code, prove the complete production-shaped outcome-firewalled path, obtain one
complete independent PASS, and only then publish the executable manifest,
approval request and launch task for the sole remaining human economic approval.

## Non-goals

- No economic outcome, candidate ranking, outer-fold outcome or control outcome.
- No protected-period or Capital.com payload access.
- No change to the frozen 11,968-row typed search space, 11,963 unique
  executions, 800 controls, axes, folds, costs, funding, selection or routes.
- No acquisition, capture restart, C17, account action, order, deployment or
  economic campaign launch.
- No mutation of `/opt/testerdonch/code` or any unrelated original-checkout
  state.

## Assumptions and unresolved questions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Candidate baseline | verified | clean preferred worktree at `84fda51...` | stop on divergence |
| Original dirty binary | verified unrelated | no literal config/runtime references; exact binary-safe record | preserve; stop if a runtime dependency appears |
| Economic authorization | verified absent | exact Stage-23 task | keep payoff readers closed |
| Production input access | verified allowed only for 2023–2025 decision-time construction | Stage-23 sections 4, 6 | instrument reads and reject payoff columns |
| Search-space identity | verified frozen | V04 registries and Stage-23 section 1 | stop on count/address change |
| Independent review | required | Stage-23 section 7 | no final publication unless PASS |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, `main` and `origin/main` at
  `56059c1a2e86c91c32087bd4e4da9257b632fbd9`.
- Original staged/unstaged/untracked/conflicted counts: `0/1/0/0`; only
  `/opt/testerdonch/code` is modified.
- Recovery record: `dirty_original_checkout/`; binary content is intentionally
  not copied under the binding Stage-23 handling rule.
- Original checkout left unchanged: yes.
- Isolated worktree: `/opt/testerdonch-stage22-20260720`.
- Stage-23 branch: `agent/stage23-remediate-stage22-v04-20260721`.
- Base commit: `84fda51be77431de7151209d28c810fe027b0133`.
- Safe-isolation rationale: the candidate worktree is clean, reviewed, and has
  no dependency on the unrelated dirty binary.

## Scope and boundaries

- Venue: Kraken derivatives only.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected cutoff: `2026-01-01T00:00:00Z`.
- Economic run authorized: no.
- Remote writes: only the approved final review handoff after PASS.
- Git push: non-force only after complete independent PASS.

## Files expected to change

- `tools/core_liquid_campaign/`: production source/cache compiler, selection,
  A2, controls, temporal state, runtime, benchmark and terminal engine.
- `tools/build_stage22_core_liquid_campaign.py` and/or a Stage-23 builder:
  deterministic evidence and final packet construction.
- `tools/run_stage22_core_liquid_campaign.py`: reviewed clean-worktree runtime
  and detach boundary.
- `unit_tests/test_core_liquid_campaign.py` and focused Stage-23 tests.
- This task archive and new versioned result/handoff roots.

## Milestones

### M1 — Authority, isolation and firewall

- Acceptance: all task/continuity/handoff/review hashes pass; dirty binary is
  untouched and unused; outcome reader deny-list is executable.
- Failure: stop before source payload access or code change.

### M2 — Production cache and temporal contract

- Acceptance: exact authorized sources compile a full 187-symbol decision-time
  cache twice with byte-identical manifests/equivalent contents; protected and
  payoff columns are physically excluded; temporal and A1 state fixtures pass.
- Failure: preserve the new root and keep finalization blocked.

### M3 — Selection, A2 and controls

- Acceptance: production-shaped aggregate selection enforces roles, overlap,
  paired A2 rules and exact materialization; all 800 controls use exact seeds
  and duplicate/unavailable reconciliation.
- Failure: no packet finalization.

### M4 — Runtime, capacity and terminal state

- Acceptance: atomic large-result persistence, detachment/restart, heartbeat
  health gates, representative strata benchmark, full terminal routes,
  forensics and exact status reconciliation pass without payoff access.
- Failure: no packet finalization.

### M5 — Deterministic replacement candidate

- Acceptance: exact registry/control/economic-address counts remain frozen;
  every required Stage-23 artifact exists and replays byte-identically; focused
  and relevant broader tests pass.
- Failure: repair mechanically or report one complete blocker list.

### M6 — Independent review and publication

- Acceptance: independent reviewer returns PASS on exact source and generated
  bytes; final manifest/approval/launch task are then created, hash-bound,
  committed, integrated non-force, pushed, handed off and published to
  continuity.
- Failure: do not update `main`/`origin/main`; report every remaining finding.

## Validation commands

- `.venv/bin/python -m unittest -v unit_tests.test_core_liquid_campaign`
- supported deterministic packet builder and independent replay commands
  discovered in `tools/core_liquid_campaign/`;
- `.venv/bin/python -m compileall tools/core_liquid_campaign unit_tests`;
- `git diff --check`, source/packet manifests, secret scan, protected/payoff
  access audit and independent review replay.

## Risk and rollback

- User work risk: the original dirty checkout is read-only and excluded.
- Data risk: all source roots are read-only; new caches/results use fresh
  versioned roots and atomic writes.
- Rollback: preserve commits and superseded result roots; no reset, clean,
  deletion, overwrite or force operation.
- Remote policy: unique task folder only; no overwrite; round-trip all files.

## Decision log

| UTC | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-07-21T06:25:02Z | Continue from exact candidate commit in existing isolated worktree | clean `84fda51...` | no reconstruction needed |
| 2026-07-21T06:25:02Z | Treat dirty `code` binary as preserved unrelated state | Stage-23 section 2 and no references found | never copy, execute or mutate it |
| 2026-07-21T06:25:02Z | Keep outcome firewall closed | economic authorization explicitly no | production features/schedules and synthetic payoff only |

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|
| 2026-07-21T06:25:02Z | M1 | authority, continuity, handoff and clean worktree verified | inventory code and implement seven lanes |
| 2026-07-21T08:23:33Z | M2-M5 | partial mechanical remediation and v07 evidence completed; 45 focused tests and runtime canaries passed | complete independent review |
| 2026-07-21T08:23:33Z | M6 | independent review BLOCK; seven findings retain blocking subfindings; review SHA-256 `ae531e76...` | stop final publication and return consolidated report |
