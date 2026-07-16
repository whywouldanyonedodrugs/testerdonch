# Plan

Task: backtesting agent governance integration.

Date: 2026-07-16.

## Objective

Preserve the dirty `/opt/testerdonch` checkout without changing its worktree files, create an isolated governance branch, reconcile the approved agent-governance package with repository-native authority, validate using non-economic checks, and package/upload the completed task archive to the approved Drive folder when safely possible.

## Non-Goals

- No economic screen, backtest, parameter sweep, validation run, or strategy-selection output inspection.
- No protected outcome inspection at or after `2026-01-01T00:00:00Z`.
- No live trading, private-account action, order placement, risk change, push, merge, rebase, reset, restore, clean, or remote overwrite.
- No modification of economic strategy behavior.

## Authority And Inputs

- Exact user task: `TASK_SPEC.md`.
- Approved Drive folder ID: `1eWdmin0E7QKYJ5AvFcpRmvvDG3YiWlBz`.
- Approved ZIP: `qlmg_backtesting-agent-governance-handoff_20260716_v01.zip`.
- Verified ZIP bytes: `106041`.
- Verified ZIP SHA-256: `fc06159e5caee52e3d5a0521601d31c9fb9fc8c0300a63ecab052b17cd4fbae2`.
- Package manifest: `received/PACKAGE_CONTENT_MANIFEST.json`.
- Transfer manifest: `received/TRANSFER_MANIFEST.json`.

## Repository State

- Original root: `/opt/testerdonch`.
- Original branch: `main`.
- Original commit: `404cd207085406e2dc9e19bbae7558392c750c95`.
- Dirty counts: 139 staged entries, 4 unstaged entries, 96 untracked entries, 0 unmerged entries.
- Recovery bundle: `/tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z`.
- Recovery manifest SHA-256: `4298c44952057de78cacd4125a4f8e8b557c8340153fa4b58294a90e4489bbc6`.
- Isolated worktree: `/opt/testerdonch-agent-governance-20260716`.
- Isolated branch: `agent-governance/backtesting-harness-20260716`.
- Base commit: `404cd207085406e2dc9e19bbae7558392c750c95`.

## Expected Changed Files

- Add root `AGENTS.md`.
- Add `docs/agent/` manuals from the approved package, revised where repository-specific facts are known.
- Add `.agents/skills/qlmg-*` and `skill_evals/qlmg-*`.
- Revise active docs under `docs/` to remove Bybit-era active authority and encode Kraken-only protected-period rules.
- Preserve pre-governance active docs under `docs/agent/superseded/20260716_pre_governance/`.
- Add this task archive under `docs/agent/task_archive/20260716_backtesting-governance-integration/`.

## Milestones

1. Verify repository identity, package integrity, and Drive accessibility.
   - Status: complete.
   - Verification: rclone listing, ZIP size/hash, embedded manifest hash comparison.
2. Preserve dirty checkout.
   - Status: complete.
   - Verification: recovery bundle with patches, manifests, Git bundle, and unchanged status hash.
3. Create isolated worktree.
   - Status: complete.
   - Verification: clean status on branch `agent-governance/backtesting-harness-20260716`.
4. Integrate governance docs and skills.
   - Status: in progress.
   - Verification: diff review, markdown/link checks, skill structure checks.
5. Run non-economic validation and independent review.
   - Status: pending.
   - Verification: recorded command exit codes and review findings.
6. Package local archive and attempt approved Drive handoff.
   - Status: pending.
   - Verification: ZIP hash, remote collision check, upload, round-trip hash verification or exact blocker.

## Rollback

No dirty checkout rollback is required because original worktree files were not changed. To abandon this integration, remove the isolated worktree and branch after confirming no needed archive artifacts remain only there. The external recovery bundle remains available at the recorded path.
