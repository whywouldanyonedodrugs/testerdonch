# Commands and results

- Verified canonical `main`, `origin/main`, the Stage 20 authority hashes, the
  sequence-2 continuity pointer/snapshot, and the prior bound-stop handoff.
- Reconciled all 6,459 pre-existing job markers and 3,193 artifact claims
  before reuse. No protected row or Capital.com payload was opened.
- Patched only `process_tree_rss` and directly related tests. The 27 relevant
  unit tests passed, `git diff --check` passed, and a live `/proc` probe
  returned a positive supervisor-tree RSS sample.
- Repeated deterministic replay and the synthetic end-to-end resume canary.
  Their bindings were respectively
  `82fcbe735f719f4173bc37b29ba8e8a48ca48290d9894ffb89ad9188e78b076b`
  and `aa41739075f52ea544d6f985fe8cd71a19b0fa6e31409f55dcfd1803c637e168`.
- Independent pre-outcome patch/resume review passed. Secure Telegram
  preflight and the final atomic launch audit passed.
- Resumed in place. The supervisor reused 6,459 completed jobs, submitted only
  missing work, delivered the overdue first heartbeat, and released health
  only after a reconciled real cell and verified campaign state existed.
- Campaign status became `terminal_complete` at
  `2026-07-20T16:06:05.607944Z` within the original wall-time contract.
- Terminal mechanical reconciliation passed: 186/186 executable cells, zero
  omitted cells, 42 inherited non-executable attempts, 228 programme attempts,
  8,415 unique job markers, and 4,556 verified artifact claims.
- Controls, Phase 6, C17, protected outcomes, Capital.com, acquisition,
  account actions, orders, deployment, and live trading were not executed.
