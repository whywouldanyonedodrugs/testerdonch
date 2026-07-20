# Stage 20 RSS ESRCH patch and resume plan

Status: in_progress
Owner: backtesting agent
Created UTC: 2026-07-20
Repository root and commit: `/opt/testerdonch-stage20-20260720` at `7e7c7b47a7693b7923097f01c22b4b7287b6d971`

## Objective

Patch only `process_tree_rss` so vanished `/proc` entries are tolerated without
masking other failures, independently review the patch and resume boundary,
and idempotently resume the existing approved Stage 20 Phase 2–5 campaign
within its original 14,400-second wall-time budget.

## Non-goals and hard boundaries

- No changes to campaign economics, translations, cells, folds, metrics,
  selection, costs, funding, universe, lanes, or execution semantics.
- No Phase 6, C17, protected outcomes, Capital.com, acquisition, capture,
  account, order, deployment, or live action.
- Do not inspect partial outcomes or rebuild valid artifacts.

## Verified authority

- `main` and `origin/main`: `7e7c7b47a7693b7923097f01c22b4b7287b6d971`.
- campaign manifest: `e7d618a2a24c574c9ba83d323df605a6434c2e816fe24c465d183ba5d6256990`.
- approval packet: `3fde09f16efff2479ee847fe26c859a67ef1d516b0f0acbcbaec141154a87bd6`.
- bound-stop ZIP: `808562f0236ce819dcf52606450049cef13c771c20e6dd0fd765e849d11823f7`.
- continuity pointer sequence 2; snapshot physical SHA-256
  `4e4ecb46384e73ef71743bbf4f34fac85d780f2df3a595b3ca52b61a65d27f26`.
- resumable root contains 6,459 completed job markers pending full hash
  reconciliation before reuse.

## Files expected to change

- `tools/run_stage20_economic_campaign.py`: only `process_tree_rss`.
- `unit_tests/test_stage20_campaign.py`: deterministic RSS failure-path tests.
- this task archive: plan, commands, reviews, manifests, completion and handoff.

## Milestones

1. Reconcile authority, all existing marker/artifact hashes, stop state, and
   remaining original wall-time allowance; stop on mismatch.
2. Apply the surgical ESRCH handling and root-sample fail-closed guard; pass
   focused and relevant broader tests.
3. Refresh affected source hashes only, obtain independent review, repeat
   secure Telegram and atomic final authority gates.
4. Resume idempotently with only missing jobs and a runtime allowance capped
   by the original campaign deadline; stop on any gate or resource failure.
5. At terminal completion or a new bound stop, independently reconcile,
   commit/push non-force, hand off a unique closed package, and publish one
   dynamic continuity transaction.

## Rollback

The prior commit and bound-stop handoff remain immutable. Valid markers and
artifacts are never deleted or overwritten. A failed resume preserves the same
run root and produces a new explicit stop record.
