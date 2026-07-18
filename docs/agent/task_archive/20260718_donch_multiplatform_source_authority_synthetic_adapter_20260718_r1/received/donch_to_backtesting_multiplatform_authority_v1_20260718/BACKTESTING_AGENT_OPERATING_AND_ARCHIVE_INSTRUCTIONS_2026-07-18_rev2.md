---
status: proposed repository instruction source; application pending approved task
date: 2026-07-18
revision: 2.0
scope: backtesting agent behavior for Kraken, Capital.com adapters, directed research, archives and Drive handoff
authority: approved multi-platform policy subordinate to repository machine contracts
supersedes: BACKTESTING_AGENT_OPERATING_AND_ARCHIVE_INSTRUCTIONS.md after reviewed application
provenance: revision 1; preflight; approved implementation task
known limitations: current repository authority remains Kraken-only until the approved patch is applied
---

# Backtesting Agent Operating and Archive Instructions

## Start

Verify root, `AGENTS.md` chain, branch, commit, remotes, worktree, commands, data roots, archive convention and Drive target. Preserve user work and use an isolated branch/worktree for approved changes.

## Boundaries

```text
research_platforms: Kraken derivatives; verified-manifest Capital.com instruments
rankable_interval: [2023-01-01, 2026-01-01)
protected_period: 2026 onward
Kraken July 2026: execution_calibration_only
Capital.com 2026+: data_engineering_only
paid vendors: prohibited
orders/account mutation/private actions: prohibited
economic run: only with exact frozen approval
push/merge/deploy: only with exact approval
```

Existing Kraken IDs, hashes, results and roots are immutable. New outer platform-aware identities may reference them.

## Platform invariants

Kraken and Capital.com use separate adapters and economics. Capital.com requires bid/ask, epic, contract form, calendar, lifecycle, financing, corporate actions and conversion authority. Do not use midpoint fills, infer historical rules from current metadata, or treat reported volume as verified exchange volume.

Directed cross-platform work freezes source availability, target first executable quote, direction, controls and multiplicity. Reverse direction is separate.

## Current task authorization

For `Multi-Platform Rankable Source Authority Contract and Synthetic Adapter Boundary v1`, implement only the approved source guard, compatibility call, synthetic adapter, focused documentation and tests. A local reviewed commit is authorized. Real Capital.com payloads, economics, protected outcomes, push and merge are prohibited.

## Archive and Drive

Archive the received task and Donch context, plan, progress/decisions, changed files, commands/results, validation, artifact manifest, review, completion and next action. Update durable repository authority only where the task changes facts. Use `drive_handoff: approved_default`, verify remote content by hash and retain the local copy.

## Completion

Report repository identity, diff, commands/tests, economic/protected status, artifacts/hashes, local and Drive archives, review, unverified items, prohibited actions, rollback, approvals and next bounded task.
