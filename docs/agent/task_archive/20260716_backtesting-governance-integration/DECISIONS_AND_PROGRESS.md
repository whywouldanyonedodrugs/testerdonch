# Decisions And Progress

## Decisions

- Verified `/opt/testerdonch` as the Git root and treated existing dirty state as user/unknown work.
- Used `qlmg_sweep_drive:` because it could address the exact approved Drive folder ID without exposing credentials.
- Downloaded and verified the approved ZIP before extraction; rejected use of similarly named or unverified paths.
- Kept the full dirty-checkout recovery bundle outside the Git worktree because it is 168 MB and contains binary-capable recovery material. The repository archive records its manifest and hash.
- Used `404cd207085406e2dc9e19bbae7558392c750c95` as the safe base because it was the verified original `HEAD`.
- Created `/opt/testerdonch-agent-governance-20260716` on branch `agent-governance/backtesting-harness-20260716` for all repository writes.
- Reconciled active docs toward Kraken-only authority because existing base docs still described Bybit as active; pre-governance copies were preserved before revision.
- Kept old Donch and Bybit-oriented materials as provenance only, not active authority.
- Did not modify economic code, run economic commands, inspect protected outcomes, push, merge, reset, restore, clean, or overwrite remote artifacts.

## Progress Log

- Read user task specification from the attached pasted text.
- Verified repository root, branch, commit, worktree list, submodule status, sparse-checkout state, and sanitized remote.
- Located existing docs and confirmed no base `AGENTS.md` or `docs/agent/` tree.
- Verified `rclone` version and `qlmg_sweep_drive:` remote label.
- Listed the approved Drive folder by exact folder ID and found the expected ZIP and transfer manifest.
- Downloaded approved transfer files to `/tmp/codex_governance_20260716/drive_download`.
- Verified ZIP size and SHA-256 against the task and transfer manifest.
- Extracted the ZIP to `/tmp/codex_governance_20260716/package_extract`.
- Verified 30 embedded package entries by size and SHA-256.
- Built dirty-checkout recovery bundle under `/tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z`.
- Confirmed the original checkout status hash was unchanged after recovery.
- Created the isolated governance worktree and branch.
- Copied verified ready-to-apply governance files into the isolated worktree.
- Preserved pre-governance active docs under `docs/agent/superseded/20260716_pre_governance/`.
- Rewrote root `AGENTS.md`, active docs, and repository map to match verified repository state and Kraken-only authority.
