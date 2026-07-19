# Stage 9 — KDA02 purge-state adjudication and conditional Level-3 execution

Status: in_progress
Owner: backtesting agent
Created UTC: 2026-07-19
Repository root and commit: `/opt/testerdonch-stage9-20260719` at `baaa10c224807e1dc7e32bfee7227711cb0c1279`

## Received task and archive context

- Exact task specification: `TASK_SPEC.md`, SHA-256 `a32cf0e0545a307e175305e42264793ad9e822681a5ecaaee5aee97fe7615ef7`.
- Human authority: outcome-free KDA02 v2 adjudication and one exact conditional economic run after mechanical gates and an independent pre-outcome approval; task-scoped commits, non-force push, and approved-default Drive handoff are authorized.
- Closed lineage: KDA01 terminal decision `KDA01_level3_repaired_no_primary_pass_stop` at commit `baaa10c224807e1dc7e32bfee7227711cb0c1279` and run-manifest SHA-256 `2e8d8cce4ed5507bbac11b86d98158e7bcfc3ce60905c82c1cc22224636eff05`.
- Durable archive: this directory.
- Drive target: `qlmg_sweep_drive:` under root folder ID `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`, unique task child only.

## Objective

Produce a point-in-time-causal, episode-level KDA02A v2 liquidation-plus-OI-reset tape; separate KDA02B; freeze all feasible definitions before returns; conditionally execute exactly once; and return the predeclared terminal decision with complete local and compact handoff evidence.

## Non-goals

- No KDA01 rescue, control outcomes, KDA02B outcomes, threshold search, new data, Capital.com payloads, protected rows, or live actions.
- No use of mark as fill, no native liquidation-side claim, and no claim of survivorship-free coverage.

## Assumptions and unresolved questions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Starting commit | verified | Git HEAD equals expected commit | stop on change |
| Stage 8A identities and KDA02 tape | verified | tracked Stage 8A records plus local tape hash `c4d553…` | stop on mismatch |
| Analytics units | inferred | Stage 8A `inferred_authoritative_v1`; common base-unit ratio cancels | preserve inference label |
| Universe | verified capped | 187-symbol current-roster bar-existence cohort hash `5df5b3…` | never call survivorship-free |
| Rankable boundary | verified | `[2023-01-01, 2026-01-01)` | fail before protected read |
| Conditional economics | authorized but gated | exact task text | do not open prices before approval |

## Authority paths

| Path | Role | Version/hash | Verified current? |
|---|---|---|---|
| Stage 8A task archive | source contracts and counts | commit `41b64b5…` | yes |
| Stage 8A local KDA02 tape | provenance reconciliation only | `c4d553…` | yes |
| Stage 8A feature cache | outcome-free source features | feature hash `4673ff…` | yes |
| Kraken acquisition manifest | authorized PF trade/mark bars | content hash `f1520f…` | to reverify in runner |
| Stage 8C1 archive | repaired timestamp semantics | manifest SHA-256 `2e8d8c…` | yes |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, `main`, `baaa10c…`.
- Staged / unstaged / untracked / conflicted: 0 / 0 / 41 Git-visible files in 3 roots / 0; two relevant ignored ZIPs were also preserved.
- Recovery bundle: `/opt/testerdonch-recovery-stage9-20260719`, manifest SHA-256 `6666fbd76fbe5adb0ca8b3ed1cfa1cb640c0931bbe9dc18fc1ea436582de9634`.
- Original checkout left unchanged: yes.
- Isolated worktree: `/opt/testerdonch-stage9-20260719`; branch `agent/stage9-kda02-purge-adjudication-20260719`; base `baaa10c…`.
- Intended Stage 9 paths do not overlap the preserved untracked roots.

## Scope and boundaries

- Venue: Kraken linear PF derivatives in the manifest-authorized capped cohort.
- Train interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected cutoff: `2026-01-01T00:00:00Z` exclusive.
- Economic run: conditionally authorized once for the exact frozen KDA02 v2 contract only.
- Remote writes: authorized only to the approved default non-overwriting Drive target after the local package closes.
- Forbidden: controls, KDA01/KDA03 economics, KDA02B outcomes, Capital.com/capture, protected access, force push, merge, overwrite, or deletion.

## Files expected to change

- `tools/qlmg_kda02_v2.py`: causal features and episode/event identity.
- `tools/build_kda02_v2_prerun_freeze.py`: source verification, shard generation, mechanical gates, and contract freeze.
- `tools/qlmg_kda02_level3.py` and `tools/run_kda02_level3.py`: exact frozen timestamp/execution and economic calculations.
- `unit_tests/test_kda02_v2.py` and `unit_tests/test_kda02_level3.py`: required boundary and regression coverage.
- This task archive and factual registries/continuity files discovered during integration.

## Milestones

### M1 — Outcome-free implementation and adjudication

- Build causal 15-minute features and KDA02A episode/event tapes while preserving Stage 8A and KDA02B lineage.
- Acceptance: source hashes/counts reconcile; exact-window, prior-day normalization, identity, protected, and contamination tests pass.
- Failure: preserve evidence and return the exact mechanical remedy.

### M2 — Mechanical gates and freeze

- Attach corrected timestamp eligibility, actual-exit non-overlap, clusters, definitions, controls, and hashes without opening price columns.
- Acceptance: at least two primary branches pass or stop with `KDA02_v2_mechanically_unavailable`; independent pre-outcome review approves an immutable contract.

### M3 — Conditional Level-3 run

- Open only authorized PF 5m trade-bar opens after M2 approval; compute frozen costs, equal-day inference, bootstrap, concentration, funding diagnostics, and gates once.
- Acceptance: deterministic replay, arithmetic, post-run review, and one terminal decision.

### M4 — Integration and handoff

- Update factual registries, manifest all artifacts, commit reviewed changes, non-force push, create unique ZIP, upload no more than five compact files, and round-trip verify SHA-256.
- Failure: retain local archive and report the exact Git or remote blocker without overwriting.

## Validation commands

Repository-supported patterns are `python3 -m pytest <focused files>` and the task runners added above. Exact commands and outcomes will be recorded in `COMMANDS_AND_RESULTS.md`; no economic runner is invoked before the pre-outcome review artifact approves the matching hashes.

## Risk and rollback

- User work is protected by the external recovery bundle and isolated branch.
- All generated caches and run roots are new/versioned; source roots are read-only.
- Rollback is branch/worktree removal only after evidence retention; no destructive command is needed.
- Drive collision policy is increment-only, never overwrite.

## Decision and progress log

| UTC | Decision/result | Consequence |
|---|---|---|
| 2026-07-19 | Expected HEAD verified; dirty main contained unrelated untracked work | created validated recovery and isolated worktree |
| 2026-07-19 | Stage 8A KDA02 tape and required recorded hashes matched | implementation may proceed outcome-free |
