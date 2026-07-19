# Stage 11 — KDA03 Basis-Shock Research

Status: implementation_and_reviews_complete_handoff_prepared
Owner: Codex backtesting agent
Created UTC: 2026-07-19T17:32:25Z
Updated UTC: 2026-07-19T19:30:00Z
Repository root and commit: `/opt/testerdonch-stage11-20260719` at `e841469984478f7436db824587eac46dcd454c6d`

## Received task and archive context

- Exact task specification: `received/TASK_SPEC.md`; SHA-256 `ef75b31cfaedb6fd563324c1831f280f37be979146b5fb14ea1a2f0dca789967`.
- Task ID: `donch_bt_stage_11_kda03_basis_shock_20260719_v2`.
- Preserve terminal decisions: `KDA01_level3_repaired_no_primary_pass_stop`, `KDA02_level3_no_primary_pass_stop`, and KDA02 negative reversal 6h route `conditional_context_candidate_unvalidated`.
- Human approval: outcome-free adjudication; one conditional frozen KDA03 Level-3 run only after independent pre-outcome approval; commit, non-force push, and approved-default Drive handoff authorized.
- Durable archive: this directory.
- Approved Drive target: `qlmg_sweep_drive:` under folder ID `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`, unique child only, round-trip verification required.

## Objective

Build and independently review a causal, point-in-time KDA03 basis-shock contract; mechanically adjudicate all frozen primary/robustness branches; and, only if at least one primary branch is eligible, execute exactly one frozen Level-3 run and assign policy-v1.0 routes.

## Non-goals

- No KDA01/KDA02 rescue or mutation.
- No control execution; controls are frozen only.
- No protected-period, Capital.com payload, new-data acquisition, live trading, merge, force push, or remote overwrite.
- No arbitrage/spread claim: KDA03A is a directional PF-futures reference-led proxy.
- No post-outcome filter, threshold, definition, horizon, context, or diagnostic mutation.

## Assumptions and unresolved questions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Required base is current authority | verified | task specification and commit `e841469…` | stop on mismatch |
| Policy JSON SHA-256 | verified | `c54d4a…cdf1aa` from exact base | stop on mismatch |
| Stage 8A semantic, feature, analytics, cohort authority | verified references; payload hashes pending full preflight | Stage 8A manifests and completion summary | fail closed before generation |
| Stage 8A 494,270 rows are broad masks, not economic-ready events | verified | completion summary and feasibility matrix | preserve only as adjudication provenance |
| OI historical retention and current-roster cohort cap | verified evidence limit | Stage 8A authority docs | disclose; do not claim survivorship freedom |
| Official PF trade and mark 5m bars are interval-start timestamps | verified by existing contracts; re-test locally | KDA01 repair and task clock | stop on ambiguity |
| At least one branch will meet mechanical gates | proposed/unknown | outcome-free generation required | return `KDA03_mechanically_unavailable` if none |

## Authority paths

| Path | Role | Version/hash | Verified current? |
|---|---|---|---|
| `docs/agent/RESEARCH_GATE_ROUTING_POLICY.json` | route policy | SHA-256 `c54d4a…cdf1aa` | yes |
| Stage 8A archive | semantics/cache/feasibility authority | semantic `289368…eea60`; feature `4673ff…193b4` | yes, payload verification pending |
| `/opt/parquet/kraken_derivatives/analytics/stage7c_v1` | analytics authority | manifest `f1520f…d92a6d` | registry verified; object checks required |
| Stage 8A feature partitions | causal base grid | cache manifest plus per-partition SHA-256 | must verify before read |
| Kraken K0 manifest | official PF trade/mark 5m authority | repository-discovered manifest path | must fail closed per existing loader |
| `docs/agent/CURRENT_*_REGISTRY.csv` | continuity and terminal decisions | commit `e841469…` | yes |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, branch `main`, commit `baaa10c224807e1dc7e32bfee7227711cb0c1279`.
- State: 0 staged / 2 unstaged / 43 untracked files / 0 conflicts.
- Recovery bundle: `/opt/testerdonch-recovery-stage11-20260719`; manifest SHA-256 `b1c102357b0008cf01fda652ca24990ab43d13115794096a6c3d920e2609e201`.
- Sensitive/protected exclusions: none; content preserved in closed safe-copies tar without protected-outcome inspection.
- Original checkout left unchanged: yes; status snapshot SHA-256 remained `cc8d03…03670`.
- Isolated worktree: `/opt/testerdonch-stage11-20260719`, branch `agent/stage11-kda03-basis-shock-20260719`, exact base `e841469…`.
- Overlap: none; task files do not overlay dirty root paths.

## Scope and boundaries

- Venue/instrument: Kraken linear perpetual futures PF contracts; directional fixed-notional study.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Protected cutoff: `2026-01-01T00:00:00Z`; physical and logical exclusion before strategy processing.
- Economic run: conditionally authorized exactly once for independently approved frozen KDA03 definitions, 14/32 bps, 1h/6h, seed 20260719.
- Remote write: one non-overwriting task handoff to the approved Drive target after package validation.
- Forbidden: controls, protected/Capital.com outcomes or payloads, KDA01/KDA02 outcomes beyond preserved registry facts, acquisition, outcome-conditioned mutation, event caps/sampling, force operations.

## Files expected to change

- `tools/qlmg_kda03_v1.py`: causal features and episode/event identities.
- `tools/build_kda03_v1_prerun_freeze.py`: Stage 8A adjudication, mechanical schedule, frozen definitions/controls/contracts.
- `tools/qlmg_kda03_level3.py`: deterministic costs, estimands, bootstrap, and policy-v1.0 routing.
- `tools/run_kda03_level3.py`: review-gated, official-open economic runner.
- `unit_tests/test_kda03_v1.py`, `unit_tests/test_kda03_level3.py`: causal, semantic, timing, identity, cost, and routing tests.
- This task archive: specification, plan, contracts, reviews, validation, manifests, completion, next action, and selected compact run evidence.
- Current KDA registries: KDA03-only append/update without mutating KDA01/KDA02 terminal facts.
- Large feature/event/bootstrap Parquet: isolated local roots only, hash-manifested and not committed.

## Milestones

### M1 — Preflight and Stage 8A adjudication

- Verify every authority/hash; reconcile 186,265 / 247,169 / 60,836 = 494,270 broad rows.
- Acceptance: zero authority mismatches; explicit KDA02 overlap exclusion; no outcomes opened.
- Failure: stop with `blocked_with_exact_mechanical_remedy`.

### M2 — Causal generator and frozen schedule

- Implement exact contiguous 15m windows; prior-day 60-calendar-day normalization; KDA03A/B/C episodes, immediate/rejection events, clusters, corrected at-or-after execution schedule; definitions and controls freeze.
- Acceptance: tests pass; all primary/robustness identities deterministic; zero duplicate addresses/protected rows; branch gates reported.
- Failure: repair only the mechanical contract before any outcome access.

### M3 — Independent pre-outcome review

- Freeze and hash generator, builder, event/episode tapes, eligible definitions, controls, and final contract; obtain separate-agent review.
- Acceptance: explicit approval tied to exact hashes and no unresolved blocking findings.
- Failure: no economic run; repair, re-freeze, and re-review.

### M4 — Conditional Level-3 run and routes

- If any primary definition is mechanically eligible, run exactly once using official PF 5m opens, fixed notional, 14/32 bps, 1h/6h, funding diagnostics, 10,000 market-day resamples seed 20260719.
- Acceptance: deterministic replay; decision tapes and all estimands/concentration/context diagnostics reconcile; exact policy route per primary definition.
- Failure: preserve attempt and return exact mechanical remedy; never retune.

### M5 — Review, registry, package, commit/push/handoff

- Obtain independent post-run review when economics executes; update KDA03 registries; validate manifests/secrets; commit, non-force push, and round-trip verify unique Drive package.
- Acceptance: package/ZIP hashes pass, protected/control/Capital.com counts zero, remote bytes and SHA-256 match.
- Failure: retain local archive and report the exact blocker without overwrite.

## Validation commands

Commands will be recorded only after discovery/execution. Initial supported pattern is `python3 -m pytest <focused test files>` from `REPOSITORY_MAP.md`; exact Stage 11 commands will be added after implementation.

## Artifact paths

| Artifact | Path | Retention |
|---|---|---|
| Task archive | this directory | Git + Drive ZIP |
| Feature partitions | `/opt/parquet/kraken_derivatives/analytics/stage11_kda03_v1/` | local, hash-manifested |
| Economic output, if authorized gate opens | `/opt/testerdonch-stage11-20260719/results/rebaseline/phase_kraken_kda03_level3_20260719_v1/` | local, compact evidence copied to task archive |
| Handoff ZIP | task archive `handoff/` | local retained + verified Drive copy |

## Risk and rollback

- User work at risk: none; original checkout is untouched and independently recoverable.
- Data/output mutation: all Stage 11 paths are new; builders reject existing final roots.
- Rollback: preserve commits/artifacts; supersede rather than delete; abandon isolated branch only after handoff if separately requested.
- Remote policy: pre-list exact child, increment `vNN`, never overwrite/delete/sync.

## Decision log

| UTC | Decision | Evidence | Consequence |
|---|---|---|---|
| 2026-07-19T17:32:25Z | Use isolated worktree at exact task base | dirty root and matching Stage 10 commit | root remains unchanged |
| 2026-07-19T17:32:25Z | Treat 494,270 rows as masks only | task and Stage 8A contracts | fresh causal KDA03 generation required |
| 2026-07-19T17:32:25Z | Gate economics on hash-bound independent review | explicit conditional authorization | open prices remain unread until approval |

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|
| 2026-07-19T17:32:25Z | M1 preflight | base/policy/recovery/worktree verified | verify Stage 8A objects and implement generator |
| 2026-07-19T18:26:00Z | M2 outcome-free freeze | 104,937 episodes; 113,539 events; six primary branches feasible | independent review |
| 2026-07-19T19:12:00Z | M3 corrected pre-outcome approval | first freeze preserved; repaired contract approved; deterministic replay matched | commit freeze and execute once |
| 2026-07-19T19:24:00Z | M4 conditional Level-3 | one run; 199,787 trades; 11 rejected and one sample-limited primary route | independent post-run review |
| 2026-07-19T19:32:00Z | M5 independent post-run review | all economics and 18 output hashes reproduced; no findings | finalize package and verified handoff |

## Completion record

Terminal status: `KDA03_level3_routes_assigned`. Pre-outcome and post-run reviews approved. One frozen economic run executed. Controls, protected data, and Capital.com data remained unopened. The only non-rejected primary is negative completed-basis rejection 6h routed `sample_limited_prospective_candidate`; it remains unvalidated and not control-eligible.
