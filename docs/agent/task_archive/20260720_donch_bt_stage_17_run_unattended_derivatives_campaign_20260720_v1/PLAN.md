# Stage 17 Approved Unattended Derivatives Campaign Plan

Status: blocked_preoutcome_common_funding_and_protected_read_defect
Owner: backtesting agent
Created UTC: 2026-07-20
Repository root and commit: `/opt/testerdonch-stage17-20260720` at `a3981b505e908b5fb617a0921f45869535e2b542`

## Received task and archive context

- Exact task specification: attached Stage 17 instruction, SHA-256 `ff0858322cd95d1d44cac8a150b1349977170792bff2f698db63ac2b5a9fd859`.
- External approval: attached JSON bytes, SHA-256 `fe57d5c1efca3af3cb83c3e07b399e03c51f5dbe635b03bd48201944506c6853`.
- Human approval scope: exact Stage-16 packet, phases 2–5, three named lanes, 186 executable cells.
- Durable task archive: this directory.
- Approved Drive target: `qlmg_sweep_drive:` root folder ID `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`, unique non-overwriting child only.

## Objective

Implement only the mechanical runtime needed by the frozen Stage-16 packet, close every pre-outcome gate, run the authorized 186-cell Kraken campaign under persistent supervision, verify terminal evidence independently, publish reviewed commits non-force, and round-trip verify a compact Drive handoff.

## Non-goals

- No Phase 6 controls or Phase 7 validation/deployment.
- No protected-period, Capital.com, new-acquisition, cohort-rebuild, symbol-expansion, capture, account, order, or live-trading work.
- No changes to cells, folds, bins, rules, costs, funding semantics, translations, selection, or decision vocabulary.
- No manual inspection or outcome-conditioned intervention during partial execution.

## Assumptions and unresolved questions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Approval bytes bind the final Stage-16 packet | verified | supplied hash and local SHA-256 | global pre-outcome stop on mismatch |
| Main and origin/main start at the approved commit | verified | Git preflight | stop on drift |
| Original checkout is clean | verified | `git status --short --branch` | preserve and isolate if state changes |
| Existing PIT data and funding authorities satisfy frozen gates | proposed pending audit | Stage-16 manifest and repository roots | stop before outcomes if not proven |
| Secure Telegram delivery is available | proposed pending dry-run | existing notifier configuration only | `blocked_telegram_notifier_unavailable` |
| Runtime can implement packet without new economics | proposed pending implementation/review | Stage-16 executable semantics | stop and require a new packet if false |

## Authority paths

| Path | Role | Version/hash | Verified current? |
|---|---|---|---|
| Stage-16 `CAMPAIGN_MANIFEST.json` | frozen campaign | `cc07499c...` | yes |
| Stage-16 approval packet | frozen approval request | `c01281e5...` | yes |
| attached `HUMAN_APPROVAL.json` | sole execution approval | `fe57d5c1...` | yes |
| Stage-16 dependency JSON files | economic/runtime semantics | approval-bound hashes | yes, preliminary |
| `/opt/parquet/kraken_derivatives` | Kraken data roots | manifest audit pending | pending |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, branch `main`, commit `a3981b505e908b5fb617a0921f45869535e2b542`.
- Staged / unstaged / untracked / conflicted: `0 / 0 / 0 / 0` at start.
- Recovery bundle: not required for a verified clean checkout.
- Original checkout left unchanged during implementation: yes.
- Isolated worktree: `/opt/testerdonch-stage17-20260720`.
- Task branch: `agent/stage17-run-unattended-derivatives-campaign-20260720`.
- Base commit: `a3981b505e908b5fb617a0921f45869535e2b542`.

## Scope and boundaries

- Venue: Kraken linear PF derivatives only.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Economic run authorized: yes, exact attached Stage-16 approval only.
- Protected outcomes authorized: no.
- Remote writes authorized: non-force Git push and approved-default unique Drive handoff only.
- Forbidden actions: every item in Stage 17's non-goals and all destructive Git/remote overwrite operations.

## Hypothesis-development admission

- Phases: registered development, deterministic freeze, outer evaluation, forward rolling replication.
- Hypotheses and economic identities: exactly the Stage-16 translation registry.
- Search budget: 186 executable cells; 42 KDX continuation-null attempts multiplicity-only; total lineage attempts 228.
- Folds: exact 27-fold inner/outer map with 12-hour purge/embargo.
- Costs: 14/32 bps pre-funding round trip plus frozen funding model.
- Beam: deterministic maximum five per family/fold.
- Claim cap: programme-exposed historical development/forward-rolling evidence only; not independent validation or live readiness.

## Files expected to change

- `tools/`: smallest packet-faithful campaign runtime, launch validator, and persistent supervisor wiring if missing.
- `unit_tests/`: focused temporal, execution, funding, authority, restart, and artifact tests.
- this task archive: exact inputs, plan, preflight, reviews, completion, manifests, handoff record, and compact evidence.
- `docs/agent/` registries/continuity: only terminal durable state required by the campaign contract.
- `results/rebaseline/`: new versioned local run root and local review package; never prior finalized roots.

## Milestones

1. Verify authority, hashes, data roots, resources, Drive target, and exact approval.
2. Implement and test only packet-faithful runtime/supervision gaps.
3. Run launch validator, synthetic canary, funding extension/coverage, independent pre-outcome review, and Telegram validation.
4. Commit reviewed mechanical runtime and pre-outcome closure separately.
5. Launch under persistent supervision; observe only health, first reconciled cell, first scheduled heartbeat, and hashes; then wait for terminal state.
6. Independently recompute terminal candidates, folds, multiplicity, routes, hashes, protected counts, and secrets.
7. Commit terminal evidence/registries, push non-force, package, round-trip verify Drive, record handoff, and leave both checkouts clean.

## Validation commands

Only commands discovered from Stage-16 tests/validator, existing repository runners, and newly reviewed Stage-17 CLI help/tests will be recorded in `COMMANDS_AND_RESULTS.md` before use.

## Risk and rollback

- User work at risk: none at start; original checkout remains untouched until reviewed integration.
- Output risk: all outputs use a new versioned root and atomic state writes.
- Rollback: stop supervisor gracefully, retain state/run root, revert only task-scoped commits with new commits if required; never reset or delete evidence.
- Remote policy: non-force Git; unique Drive child; no overwrite, sync, delete, or purge.

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|
| 2026-07-20 | start/preflight | exact approval hash, packet hashes, clean approved base, and isolated worktree verified | audit runtime/data and close pre-outcome gates |
| 2026-07-20 | funding boundary materialization | 2,197,950/2,197,950 eligible hourly boundaries available; 0 imputed gate eligible | independent funding arithmetic and protected-read review |
| 2026-07-20 | independent pre-outcome review | global stop: boundary-notional conversion not frozen; inherited funding loader opened protected payload before filtering | require replacement packet and approval; do not launch |
