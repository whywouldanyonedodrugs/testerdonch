# Stage 10 documentation and aggressive gate-policy application

Status: complete
Owner: backtesting agent
Created UTC: 2026-07-19
Updated UTC: 2026-07-19
Repository root and commit: `/opt/testerdonch-stage10-20260719` at `3ea0d320d71716a5c0890f4c924ed924224beda2`

## Received task and archive context

- Exact task specification: `TASK_SPEC.md`, SHA-256 `73d1cebffec48b568227d090a250ab092638d8eba5dd0d7e98036913c4591499`.
- Donch package: `donch_documentation_gate_policy_update_20260719_v1`; received ZIP SHA-256 `8096498ef634dbea0520cce1dbfb0c341cb8ccdfe67b8b84ece7c67468aef241`.
- Source files: 14 exact received files under `received/`; package-manifest bytes and hashes must validate before application.
- Human approval: documentation/registry edits, task-scoped commit, non-force push, and approved-default Drive handoff. No economic run, candidate returns, protected outcomes, acquisition, Capital.com payload, strategy code, loader, simulator, data, or run-root changes.
- Durable archive: this directory.
- Drive target: `qlmg_sweep_drive:` under root folder ID `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`, unique task folder only.

## Objective

Apply version 1.0 of the prospective aggressive conditional-alpha gate-routing policy and the durable Stage 7C-9 Kraken learnings to active documentation and append-only registries, while preserving all finalized roots and historical terminal decisions byte-for-byte.

## Non-goals

- No economic computation, return inspection, controls, validation, promotion, deployment, data acquisition, payload read, or code change.
- No retroactive change to KDA01 or KDA02A gates, decisions, manifests, roots, or evidence levels.
- No Capital.com documentation or input change beyond preserving existing boundaries.

## Assumptions and unresolved questions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Stage 9 base is current | verified | `origin/main` and supplied task both resolve to `3ea0d320...` | Stop on divergence. |
| KDA01 terminal | verified | Stage 8C1 finalized archive | Preserve exact token. |
| KDA02A terminal and Stage 9 manifest | verified | Stage 9 finalized archive and manifest SHA-256 | Preserve exact token and root. |
| Policy is prospective only | verified | received policy/task | Reject retroactive promotion wording. |
| Economic authorization | verified absent | task front matter | Record `economic_run_not_authorized`; use document/schema/hash checks only. |

## Authority paths

| Path | Role | Version/hash | Verified current? |
|---|---|---|---|
| `docs/agent/task_archive/20260719_donch_bt_stage_9_kda02_purge_adjudication_conditional_level3_20260719_v1/` | latest finalized economic lineage | manifest `2f19fc...` | yes |
| `docs/agent/task_archive/20260719_donch_bt_stage_8c1_kda01_forensic_timing_audit_20260719_v1/` | repaired KDA01 authority | terminal exact token | yes |
| `docs/agent/task_archive/20260717_donch_bt_stage_7c_resume_analytics_acquisition_20260717_v1/` | analytics authority | content hash `f1520f...` | yes |
| `docs/agent/` and active `docs/QLMG_*` | current narrative contracts | base commit `3ea0d320...` | yes |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, branch `main`, commit `baaa10c224807e1dc7e32bfee7227711cb0c1279`; behind `origin/main` by three commits before fetch.
- Dirty inventory: 0 staged files, 1 unstaged file, 82 untracked files in 3 reported roots, 0 conflicts, 0 submodules.
- Recovery bundle: `/opt/testerdonch-stage10-recovery-20260719`; manifest SHA-256 `a767ae58000e667f3ecee313e342b52defc6143f71d9529e61d7fc4e23cd838d`.
- Original checkout left unchanged by task edits: yes.
- Isolated worktree: `/opt/testerdonch-stage10-20260719`, branch `agent/stage10-documentation-gate-policy-20260719`, exact base `3ea0d320...`.
- Overlap: original dirty `AGENTS.md` overlaps a planned active-policy file, so all integration occurs only against verified `origin/main` in isolation.

## Scope and boundaries

- Venue: documentation covers Kraken derivatives and preserves separately manifest-authorized Capital.com boundaries.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected cutoff: `2026-01-01T00:00:00Z`.
- Economic run authorized: no (`economic_run_not_authorized`).
- Remote writes authorized: yes, only to the approved default Drive root and non-force `origin/main` workflow.
- Forbidden: strategy/data/run-root mutation; outcome/payload readers; protected or Capital.com payload access; controls; same-sample rescue; force push; overwrite.

## Files expected to change

- `AGENTS.md`, `docs/agent/REPOSITORY_MAP.md`, active QLMG manuals/state: prospective policy and source navigation.
- New active continuity, decisions, policy, learnings, and machine-readable JSON/CSV registry files under `docs/agent/`.
- This task archive: received bytes, plan, evidence, review, manifest, completion, and handoff records.

## Milestones

### M1 — Preserve and verify inputs

- Action: validate dirty recovery, source package, base commit, authority roots, and supplied hashes.
- Acceptance: exact matches; original checkout untouched.
- Verification: Git/hash/package-manifest checks.
- Failure response: stop before policy edits.

### M2 — Apply prospective policy and registries

- Action: update active documentation and append Stage 7C-9 records without rewriting old rows.
- Acceptance: exact terminal decisions remain; policy fields and routes parse and resolve.
- Verification: CSV/JSON/schema/link/source-map checks and diff review.
- Failure response: correct the smallest documentation/registry surface or stop on authority conflict.

### M3 — Independent review and closure

- Action: answer seven binding review questions, run secret/diff/test checks, build manifest and closed ZIP.
- Acceptance: no code/data/run-root changes; all checks pass; review approves prospective-only routing.
- Verification: repository-supported tests plus mechanical validators.
- Failure response: keep task blocked and do not push/upload.

### M4 — Publish authorized evidence

- Action: commit, non-force push, upload unique Drive folder, round-trip verify all bytes.
- Acceptance: `origin/main` contains the task commit and remote hashes equal local hashes.
- Failure response: retain local archive and report exact blocker without overwrite.

## Risk and rollback

- User work at risk: none in original checkout; isolated task branch only.
- Data/output mutation risk: prohibited and checked by path-based diff review.
- Rollback: revert the task-scoped documentation commit; finalized roots remain unchanged.
- Remote collision policy: increment version, never overwrite/delete/sync.

## Decision log

| UTC | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-07-19 | Use `origin/main` at supplied Stage 9 commit as base. | Fetch and commit equality. | Safe isolated worktree. |
| 2026-07-19 | Treat routing as a separate prospective dimension. | Approved policy and rankable contract. | Historical gates/decisions remain immutable. |

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|
| 2026-07-19 | M1 | Base, recovery, source package and primary hashes verified. | Apply active docs and registries. |
| 2026-07-19 | M2 | Active policy, manuals, learnings, and append-only registries applied. | Mechanical validation and independent review. |
| 2026-07-19 | M3 | Mechanical validation passed; independent reviewer approved after four documented repairs. | Commit, publish, and hand off the closed package. |

## Completion record

- Acceptance criteria met: yes for local documentation/registry application and independent review.
- Economic runs launched: no.
- Protected outcomes inspected: no.
- Remote writes: application commit `91ba2ab07630556984bcdf6b2c650fa6b84fcf7f` pushed non-force to `origin/main`; approved-default Drive folder `1U_KaAb4OOvP0rpdlTxJXAUc65r9yytVk` round-trip verified.
- Unresolved blockers: none for task closure; repository dependencies `pytest` and `pandas` remain unavailable but do not block the documentation-specific mechanical evidence.
