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
- 2026-07-21: Review round 1 at `b25e71b` returned seven material engineering
  findings. The round is archived in `INDEPENDENT_REVIEW_ROUND_01.json`; task
  disposition is repair and repeat internally.
- 2026-07-21: The first complete gate invocation at `a9ac528` wrote a canonical
  fail report. Cache completeness failed (eight XBT-only outer artifacts, no
  inner partitions); every other implemented check passed. The review also
  established that the gate must be expanded before any PASS can be accepted.
- 2026-07-22: Detached shadow v17 completed and reconciled all 1,984 A1-A4
  inner-development markers, then stopped resumably at KDA02B. All 11 KDA02B
  variants exhausted with `inconsistent aggregate denominator`; completed
  inner markers remain untouched.
- 2026-07-22: Minimal `KDA02B_009:identity_replay` replay located the first
  divergence at lazy `FamilyInput` metadata: the adapter used 117 currently
  eligible KDA symbols while immutable Stage20 `metric_row` used the full 187
  campaign symbols. A second cross-fold boundary retained mixed 90/91/92-day
  denominators instead of the additive nine-quarter 823-day denominator.
- 2026-07-22: A path-local repair now validates every source-quarter
  denominator, enforces the exact nine contiguous Stage20 folds and 187-symbol
  base, sums their disjoint denominators, and verifies streaming/materialized
  equality. The real minimal job and all-11-variant worker/order/restart
  invariance validation passed without economic or protected outcome access.
- 2026-07-22: The independent KDA02B resume review at `fbd60c8` returned
  `PASS` with no material findings. It authorized exact reuse of all 1,984
  reconciled inner markers and a detached resume from the KDA02B boundary after
  final commit binding, authority audit and secure Telegram preflight. The
  missing bound-stop Telegram message was traced to an absent notification call
  in the real shadow exception path and repaired with a focused passing test.
- 2026-07-22: The first reviewed resume entered the new KDA02B batch and then
  stopped safely because the new supervisor generation treated the prior
  generation's persisted monotonic heartbeat time as current liveness. The
  restart-local heartbeat baseline was repaired without changing the 30-minute
  schedule or 3,900-second stale bound; focused restart/stale-stop tests pass.
