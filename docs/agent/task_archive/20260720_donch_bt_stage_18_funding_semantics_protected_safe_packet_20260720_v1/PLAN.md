# Stage 18 Funding Semantics and Protected-Safe Source

Status: blocked
Owner: Codex
Created UTC: 2026-07-20
Updated UTC: 2026-07-20
Repository root and commit: `/opt/testerdonch-stage18-20260720` at `91b9d8829a21e3c4eb429186c1075bddeb52523a`

## Received task and archive context

- Exact task specification: `/root/.codex/attachments/7e35bca9-18d2-4e95-947a-15d4227760d3/pasted-text.txt`, SHA-256 `6874d84fd9239c548101a597dc6cefe1d7ddcfeffb3f9c419deaf358ca73cbfb`.
- Stage 17 terminal status: `blocked_preoutcome_common_funding_and_protected_read_defect`.
- Human authority: outcome-free funding engineering, protected-safe exact reads, packet regeneration only if all gates pass, commits, non-force push, and approved default Drive handoff.
- Economic Phases 2–5, protected payloads, strategy outcomes, Telegram, Capital.com, new acquisition, orders, and live trading are forbidden.
- Durable task archive: this directory.
- Drive target: `qlmg_sweep_drive:` under folder ID `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`, unique non-overwriting child only.

## Objective

Implement and verify a metadata-first, fail-closed funding row-group reader and determine whether the authorized local sources can produce a physically rankable-only absolute-funding package without opening protected, mixed, or unknown payloads. Regenerate funding semantics and the campaign packet only if exact-source and calibration gates pass.

## Non-goals

- No real strategy outcome, campaign launch, candidate selection, control, Telegram message, Capital.com access, network acquisition, or live action.
- No alteration of historical family decisions or prior run roots.
- No inferred funding allowance, timestamp interpretation, index/spot proxy, or zero-funding fallback.

## Assumptions and unresolved questions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Starting authority is exact | verified | `HEAD == origin/main == 91b9d88…`; clean checkout | stop on drift |
| Stage 17 incident facts | verified | Stage 17 terminal archive and task specification | preserve without recounting protected rows |
| Absolute funding source has safe row groups | unavailable | metadata audit: zero safe; 666 mixed; 34 protected across repository copies | build guard/ledgers; do not read payload; block calibration and packet |
| Funding row timestamp maps to accrual-period start | unresolved | Kraken specifies next-period and continuous accrual, but current endpoint documentation does not bind returned row timestamp | stop before economics/packet |
| Empirical q95/q99 allowance can be calibrated | blocked | zero safely readable exact absolute rows | do not invent fallback |

## Authority paths

| Path | Role | Version/hash | Verified current? |
|---|---|---|---|
| `AGENTS.md` | repository policy | current at start | yes |
| Stage 16 task archive | frozen campaign authority | manifest `cc07499c…`, packet `c01281e5…` | yes |
| Stage 17 task archive | incident and stop authority | artifact manifest `c4b40b62…` | yes |
| K0 funding manifest and acquired audit | source provenance | hashes recorded in their manifests | metadata audit pending ledger closure |
| Kraken linear derivatives specification | official funding units/accrual | accessed 2026-07-20 | yes for cashflow mechanics; not sufficient for API-row timestamp mapping |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, `main`, clean at `91b9d8829a21e3c4eb429186c1075bddeb52523a`.
- Staged / unstaged / untracked / conflicted: `0 / 0 / 0 / 0`.
- Recovery bundle: not required for a clean checkout.
- Original checkout left unchanged during implementation: yes.
- Isolated worktree: `/opt/testerdonch-stage18-20260720`.
- Branch: `agent/stage18-funding-semantics-protected-safe-packet-20260720`.
- Base commit: `91b9d8829a21e3c4eb429186c1075bddeb52523a`.

## Scope and boundaries

- Venue: Kraken linear perpetual derivatives only.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected cutoff: `2026-01-01T00:00:00Z`.
- Economic run authorized: no.
- Metadata/footer reads: authorized. Exact payload reads: only safe row groups admitted before deserialization.
- Remote write: only the approved default Drive handoff after local closure.

## Files expected to change

- `tools/kraken_protected_safe_funding.py`: metadata classification, guarded row-group reader, audit and package builder.
- `unit_tests/test_kraken_protected_safe_funding.py`: adversarial synthetic row-group and funding arithmetic tests feasible without real outcomes.
- `docs/agent/PROTECTED_ACCESS_INCIDENT_20260720_STAGE17.*`: durable incident registration.
- `docs/agent/CURRENT_CONTINUITY.md`, `CURRENT_RESEARCH_DECISIONS.md`, `TEST_AND_EVIDENCE_STANDARDS.md`, `KRAKEN_DERIVATIVES_LEARNINGS.md`: surgical continuity updates.
- This task archive and new local result/package root.
- Stage 16 authority files remain immutable unless all regeneration gates pass.

## Milestones

### M1 — Authority and physical-source preflight

- Action: verify hashes and classify every source row group from trusted footer statistics before payload access.
- Acceptance: complete ledger; zero unauthorized row-group requests; exact counts reconcile.
- Failure response: block physical package/calibration if no safe groups exist.

### M2 — Protected-safe reader and adversarial tests

- Action: implement schema/hash/footer guard, explicit row-group admission, request audit, and payload contradiction assertion.
- Acceptance: synthetic safe/protected/mixed/unknown/false-stat cases pass; broad-read bypass absent from campaign path.
- Failure response: repair smallest owning surface or stop.

### M3 — Funding semantics and allowance

- Action: only if safe exact rows exist, adjudicate timestamp meaning and calibrate frozen q95/q99 allowances without outcomes.
- Acceptance: authoritative timestamp mapping, valid mark calibration, complete symbol coverage.
- Failure response: block without packet regeneration.

### M4 — Packet, review, publication, handoff

- Action: only if M3 passes, regenerate economic identities and packet; otherwise create a truthful blocked archive. Run independent adversarial review, tests, manifests, commits, push, and Drive round-trip.
- Acceptance: no protected/economic access, clean checkouts, verified remote package.

## Validation commands

- `/opt/testerdonch/.venv/bin/python -m unittest unit_tests.test_kraken_protected_safe_funding -v`
- `/opt/testerdonch/.venv/bin/python -m unittest unit_tests.test_stage16_campaign_semantics unit_tests.test_qlmg_research_campaign -v`
- `git diff --check`
- JSON/CSV/hash reconciliation and secret scan recorded in `VALIDATION.md`.

## Risk and rollback

- Source Parquet files and finalized roots are immutable.
- All generated outputs use a new versioned root.
- A payload request for any non-safe row group raises before deserialization and globally blocks the build.
- Rollback is commit reversion; evidence roots remain preserved.
- Drive uses a new collision-checked child and immutable upload behavior.

## Decision log

| UTC | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-07-20 | Use the preferred uniform adverse-allowance selection contract if calibration becomes valid | explicit task default | exact funding remains diagnostic only |
| 2026-07-20 | Treat current physical source as unavailable for payload | 666 mixed and 34 protected row groups; zero safe | no exact payload read, no calibration, no packet regeneration unless another authorized source passes |

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|
| 2026-07-20 | M1 metadata discovery | zero safe row groups found; no payload opened | implement guard/tests and produce authoritative ledgers |
| 2026-07-20 | M2 reader and tests | protected-safe reader implemented; 9 focused and 30 campaign tests passed | close blocked evidence |
| 2026-07-20 | M3 funding semantics | timestamp meaning unresolved; allowance has zero admissible observations | do not regenerate packet |
| 2026-07-20 | M4 closure | truthful blocked package prepared; economic outputs remain absent | commit, push, and verified Drive handoff |

## Completion record

- Acceptance criteria met: M1 and M2; M3 and packet regeneration blocked by mandatory stop conditions.
- Economic runs launched: none.
- Protected outcomes inspected: none.
- Stage 18 protected funding payload rows opened: zero.
- New diagnostic root: `/opt/testerdonch/results/rebaseline/phase_kraken_rankable_exact_funding_safe_20260720_v2`.
- Unresolved blockers: no safe absolute-funding row group; unresolved historical endpoint timestamp identity; no admissible allowance calibration sample; independent acceptance unavailable.
