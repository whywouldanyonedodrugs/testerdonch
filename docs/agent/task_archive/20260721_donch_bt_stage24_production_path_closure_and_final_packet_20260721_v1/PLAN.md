# Stage 24 production-path closure plan

Status: `in_progress`

## Objective

Close all seven Stage-23 production defects without changing the frozen 11,968
registered rows, 11,963 economic executions, 800 controls, economic axes, folds,
costs, funding, selection or routes. Run one real outcome-firewalled shadow
production gate and iterate independent review internally until `PASS`, then
publish the final launch-approval packet without launching economics.

## Authority and isolation

- Exact Stage-24 task SHA-256:
  `9e546e9376408f97bc3bc2ef2862c06e746864eacbd9a5a70f0e71680eeeccdf`.
- Clean isolated worktree: `/opt/testerdonch-stage22-20260720`.
- Starting branch/commit:
  `agent/stage23-remediate-stage22-v04-20260721` at
  `32ae1153ba572d0669dedc8ee10784db65b0967a`.
- Original `/opt/testerdonch/code` is preserved byte-for-byte and excluded.
- Dynamic continuity sequence 6 and Stage-22/23 evidence hashes passed.
- Economic run authorization: absent. Real post-entry payoff, protected and
  Capital.com readers remain closed.

## Implementation milestones

1. Replace the non-executable source-index cache with a single
   `CacheAuthority`-accepted semantic cache and a lazy, point-in-time
   `FamilyInput` adapter covering every family/fold or exact unavailable reason.
2. Add a hash-bound synthetic shadow payoff provider after real event/schedule
   generation; run actual selection, A2 resolution, materialization and all 20
   controls without reading real post-entry returns.
3. Persist/replay the complete A1 temporal state machine across gaps, chunks and
   restarts.
4. Exercise the real launch CLI, authorization, cache, scheduler, notifier-test,
   recovery and terminal path in an installed detached shadow service.
5. Measure the real shadow stages and produce terminal complete/bound-stop and
   independent recomputation evidence.
6. Expose one `shadow_no_outcome` production-readiness command. Repeat fresh
   independent review and repair until `PASS`.
7. Only after `PASS`: finalize immutable packet, non-force integrate/push,
   round-trip Drive handoff, and publish one continuity transaction.

## Acceptance and failure behavior

- Every command fails closed on source/hash/schema/protected/firewall drift.
- Registry/control counts and hashes must stay frozen.
- A review `BLOCK` inside the known seven lanes triggers another repair round,
  not a human handoff.
- Stop only for a new economic semantic choice, unavailable source authority,
  protected/Capital.com access requirement, or unsafe destructive action.
- Old result roots remain immutable; every gate/review round uses a new root.

## Supported verification

- `.venv/bin/python -m unittest -v unit_tests.test_core_liquid_campaign unit_tests.test_core_liquid_campaign_stage23 unit_tests.test_core_liquid_campaign_stage24`
- `.venv/bin/python -m tools.core_liquid_campaign.production_readiness_gate --mode shadow_no_outcome --output <new-root>`
- `.venv/bin/python -m compileall -q tools/core_liquid_campaign unit_tests`
- `git diff --check`

## Progress

- 2026-07-21: Stage-24 authority, original dirty-state isolation, continuity
  sequence 6, prior handoff/review hashes and clean starting branch verified.
