# Bootstrap the Donch Dynamic Continuity Ledger

```text
task_id: donch_bt_bootstrap_dynamic_continuity_20260720_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
economic_outputs_authorized: no
protected_payload_access_authorized: no
Capitalcom_payload_access_authorized: no
new_acquisition_authorized: no
commit_authorized: yes
push_authorized: yes_non_force
drive_handoff: approved_default
```

## Objective

Create and verify a stable, agent-maintained Donch continuity ledger at:

```text
qlmg_sweep_drive:DONCH_BACKTESTING_HANDOFFS/_DONCH_CONTINUITY/
```

Use the attached:

```text
DONCH_CONTINUITY_SCHEMA_v1.json
INITIAL_STATE_SNAPSHOT_000000_STAGE19.json
INITIAL_CURRENT_STATE_POINTER.json
```

Do not open strategy outcomes.

## Required Drive structure

```text
README.md
SCHEMA.json
CURRENT_STATE_POINTER.json
events/
snapshots/
daily/
```

Events and snapshots are immutable. Only the pointer is replaceable.

## Repository work

1. Verify current repository/AGENTS/Git/worktrees.
2. Add a small dependency-free publisher/validator under existing agent tooling.
3. Add focused synthetic tests for sequence, hashes, atomic pointer replacement, stale pointer, malformed event, secret-field rejection, and Drive round-trip verification.
4. Update repository agent instructions so every material future task publishes one event and one complete snapshot after its immutable task handoff is verified.
5. Do not require updates for non-material chat discussion.
6. Add a daily digest generator; it may summarize events but is not authority.
7. Ensure the ledger excludes credentials, protected outcomes, partial candidate rankings, raw market values, and private/account data.

## Transaction

Upload immutable event/snapshot first, round-trip verify size/SHA-256, then replace the pointer through a temporary object. On pointer failure retain immutable evidence and report `continuity_pointer_stale`.

## Initial content

Publish the supplied Stage 19 snapshot as sequence 0. Verify the pointer and snapshot independently from Drive.

## Completion

Commit and non-force push reviewed changes, keep Git clean, archive the task, complete the normal immutable Drive handoff, and report the stable continuity folder URL and hashes. No economics or protected data may be opened.
