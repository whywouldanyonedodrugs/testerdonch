# Bootstrap Donch Dynamic Continuity Ledger

Status: `in_progress`
Owner: Backtesting agent
Created UTC: `2026-07-20T11:52:32Z`
Repository root and commit: `/opt/testerdonch` at `03092c5e20a3738ddc7994b9382df01366937776`

## Received task and archive context

- Specification: `received/DONCH_TO_BACKTESTING_Bootstrap_Dynamic_Continuity_2026-07-20_v1.md`, SHA-256 `8ee3b26b61bc9e4ecd0746908bd4024d983be3b2b6d5b6f86c29e9a6c4396440`.
- Source ZIP: `/opt/testerdonch/research_inputs/continuity_tools.zip`, SHA-256 `6561e99593dda278e3556ce5ad8fb015d5edf906bf8f32d5a03e6ad9297a3ff7`.
- Definition/state inputs: schema `c1697064…`, event template `1b81594e…`, initial snapshot `7a7bec15…`, initial pointer `c1ccd47c…`.
- Human authority: non-economic implementation, commits, non-force pushes, `_DONCH_CONTINUITY` publication, and approved-default task handoff are authorized.
- Durable archive: `docs/agent/task_archive/20260720_donch_bt_bootstrap_dynamic_continuity_20260720_v1/`.

## Objective

Implement and verify a dependency-free continuity publisher/validator, bootstrap the immutable Stage 19 sequence-0 snapshot and replaceable pointer in the stable Drive ledger, require future material tasks to publish continuity, and complete a verified normal handoff.

## Non-goals

- No economic run, market outcome, protected payload, Capital.com payload, acquisition, private/account access, order, or live action.
- No continuity update for non-material chat.
- No mutation or deletion of immutable events or snapshots.

## Verified assumptions and authority

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Original checkout clean | verified | empty `git status --short` | isolate from exact base |
| Base equals origin/main | verified | both `03092c5…` | stop if remote advances before integration |
| Snapshot physical hash | verified | supplied pointer equals `7a7bec15…` | reject drift |
| Snapshot embedded hash rule | verified | canonical JSON with `snapshot_sha256=null` equals `0011b911…` | encode and test exact rule |
| Stable Drive target | verified | `DRIVE_HANDOFF_TARGET.md` | stop before write on identity/root/collision failure |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, `main`, clean at `03092c5e20a3738ddc7994b9382df01366937776`.
- Isolated worktree: `/opt/testerdonch-continuity-20260720`.
- Branch: `agent/bootstrap-dynamic-continuity-20260720`.
- Staged/unstaged/untracked/conflicted at start: `0/0/0/0`.
- No recovery bundle required because ownership and cleanliness were verified.

## Files expected to change

- `tools/donch_continuity.py`: validation, transactional publication, and digest generation.
- `unit_tests/test_donch_continuity.py`: synthetic safety, sequence, hash, pointer, and round-trip coverage.
- `docs/agent/continuity/`: active schema/template and operating protocol.
- `AGENTS.md`: material-task publication rule.
- `docs/agent/REPOSITORY_MAP.md`, `SOURCE_MAP.md`: active tool and ledger routing.
- This task archive: received authority, plan, validation, review, manifest, completion, handoff, and continuity bootstrap evidence.

## Milestones

1. Implement strict dependency-free contracts and local/rclone stores; accept when focused synthetic tests prove sequence, hashes, immutable writes, atomic pointer replacement, stale detection, malformed-event rejection, prohibited-field rejection, and round-trip verification.
2. Bootstrap `_DONCH_CONTINUITY` with README, schema, directories, supplied snapshot, and pointer; accept only after separate Drive downloads and exact SHA-256/size verification.
3. Update agent instructions and continuity documentation; accept when instruction tests and diff review pass.
4. Close the archive, commit and non-force push, publish the normal compact handoff, round-trip verify it, and finish clean.

## Risk and rollback

- Remote immutable objects are never overwritten or deleted. A pointer failure leaves verified immutable evidence and reports `continuity_pointer_stale`.
- Pointer upload uses a unique temporary object followed by a move to the single replaceable pointer path.
- Local rollback is a new corrective commit; prior commits and remote immutable objects remain evidence.
- Both Drive paths use explicit root IDs, collision checks, and separate round-trip downloads.

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|
| 2026-07-20T11:52:32Z | Preflight | Clean exact base, ZIP integrity and supplied hash semantics verified | Implement tool and tests |
