# Donch Stage 19 authority verification and source-refresh decision

Date: 2026-07-20
Status: `verified_with_live_local_cleanliness_to_be_rechecked_by_agent`

## Verified authority chain

- Canonical sources `00`, `01`, and `09` were read.
- `CURRENT_STATE_POINTER.json`: 313 bytes; physical SHA-256 `28ef052ff9bb7c2523accec5c8acd36e4f903ac7efd84be44cd3972faaae1c00`; sequence 1.
- Referenced snapshot `snapshots/state_000001_20260720T121247Z.json`: 2,439 bytes; physical SHA-256 `c3bdfa6fa7c9d9427e4b39ca109262f0becd0ff004d145a5a1674694b61094bf`.
- The snapshot's embedded `snapshot_sha256` is a separate canonical-null self-hash by design; it is not the pointer-bound physical file hash.
- Stage 19 immutable handoff folder: `https://drive.google.com/drive/folders/1Iqhr3hXx4zZYvQqr1IqA5yqXuopSFavB`.
- Stage 19 compact archive: 229,604 bytes; SHA-256 `a7065fa8b88cdd9bc47e03e71c86af83735d3b31178cdacc91fdabf3343c9e90`.
- Every file listed in the Stage 19 transfer manifest matched its byte count and SHA-256 after download.
- The archive-contained campaign manifest physically hashes to `e7d618a2a24c574c9ba83d323df605a6434c2e816fe24c465d183ba5d6256990`.
- The archive-contained approval packet physically hashes to `3fde09f16efff2479ee847fe26c859a67ef1d516b0f0acbcbaec141154a87bd6`.
- GitHub repository `whywouldanyonedodrugs/testerdonch` has default branch `main`; remote `main` is exactly `245b375b00167f1b4a81f6a4449e7de1d1db83a2`.
- That commit is three commits ahead of the Stage 19 handoff-closing commit `03092c5e20a3738ddc7994b9382df01366937776`; the intervening changes are the dynamic-continuity implementation, archive closure, and handoff verification.
- The dynamic snapshot reports the local checkout clean. This environment independently verified remote `main`, Drive evidence, and hashes, but cannot directly shell into `/opt/testerdonch`; the executing agent must recheck live local branch, origin, worktrees, and cleanliness before outcomes.

## Current campaign authority

```text
lanes: KDA02B, KDA02C, KDX01
executable cells: 186
status: launch_complete_non_authorizing
campaign manifest: e7d618a2a24c574c9ba83d323df605a6434c2e816fe24c465d183ba5d6256990
approval packet: 3fde09f16efff2479ee847fe26c859a67ef1d516b0f0acbcbaec141154a87bd6
blocker: exact external human approval before Phases 2–5
```

Historical terminal decisions and the Stage 17 protected-funding access incident remain immutable. Stage 19 protected funding partitioning remains an authorized non-outcome engineering action.

## Source-refresh decision

Decision: `remain_dynamic_until_next_daily_consolidation`.

No immediate canonical-source edit is required because the compact sources already contain the dynamic-continuity policy, stable folder, read protocol, authority order, and refresh cadence. The post-source-bundle change is implementation and repository-state evidence: remote main advanced from `03092c5...` to `245b375...`, while campaign hashes, programme scope, protected handling, and terminal decisions did not change. These are exactly the facts the dynamic pointer is intended to carry.

At the next daily consolidation, update the repository-state references and source-refresh watermark, record/clear the pending continuity implementation item, and preserve the dynamic snapshot as the provenance source. An immediate edit would duplicate fast-moving repository state inside slow-moving canonical sources without changing authority or safety.

## Generated artifacts

- `HUMAN_APPROVAL_Kraken_Derivatives_Campaign_001_Stage19_Phases_2_5_2026-07-20_v1.json` — SHA-256 `57d521488e88373afd557eb457ffd119089aac0d863a0c319e73d70cf9f7690c`.
- `DONCH_TO_BACKTESTING_Stage20_Run_Stage19_Derivatives_Campaign_2026-07-20_v1.md` — SHA-256 `49a8b6de271c666e3552c66a781dbbc0a60566e438f6a4a54ce468f2fd3bcde4`.
