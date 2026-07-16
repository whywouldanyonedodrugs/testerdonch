---
name: qlmg-plan-and-preflight
description: Create evidence-based, repository-safe plans, dirty-checkout recovery records, and preflights for QLMG backtesting work. Use when a task spans multiple files, is long-running, changes data or artifact structure, has ambiguous authority or repository state, begins from staged, unstaged, or untracked work, needs a branch or worktree decision, or requires milestone acceptance and rollback planning before implementation.
---

# QLMG Plan and Preflight

## Establish the repository state

1. Resolve the Git top-level directory. Treat `/opt/testerdonch` as unverified until this succeeds.
2. Record the branch, concise working-tree state, applicable instruction chain, and remotes with credentials redacted.
3. Preserve uncommitted user work. Do not stash, clean, reset, restore, delete, or commit unknown work merely to make the tree clean.
4. When the tree is dirty or ownership is uncertain, read [DIRTY_REPOSITORY_RECOVERY.md](../../../docs/agent/DIRTY_REPOSITORY_RECOVERY.md). Inventory staged, unstaged, untracked, conflicted, submodule, large, and excluded-sensitive state; validate a recovery bundle; and use a task-authorized isolated worktree from a verified base. Do not ask the human to interpret routine diffs or choose Git mechanics.
5. Read [REPOSITORY_MAP.md](../../../docs/agent/REPOSITORY_MAP.md) and locate current machine contracts, manifests, continuity brief, manuals, and supported commands.
6. Mark each material statement `verified`, `inferred`, `proposed`, `unavailable`, or `blocked`.

## Classify the task

Choose one primary class:

- documentation or instruction change;
- non-economic code or test change;
- rankable or economic research;
- data acquisition or data-authority change;
- review-package validation or handoff.

For rankable or economic work, also use `$qlmg-rankable-backtest-contract`. For bundle work, also use `$qlmg-review-package-verification`. Do not let planning itself authorize an economic run or remote write.

## Write the plan

Use [EXECUTION_PLAN_TEMPLATE.md](../../../docs/agent/EXECUTION_PLAN_TEMPLATE.md). Include:

- one observable objective and explicit non-goals;
- assumptions, authority paths, and unresolved conflicts;
- exact files expected to change;
- forbidden actions and approval boundaries;
- milestones with action, acceptance criteria, verification, artifact path, and failure response;
- repository-discovered commands, never remembered or invented commands;
- dirty-checkout recovery, safe-isolation rationale, worktree branch, and exact base commit when applicable;
- risk, rollback, decision log, and progress log.

Keep the plan in the repository's supported plan location. If none is documented, mark the location `REPO_DEPENDENT_PLACEHOLDER` and ask before adding a new convention when the choice is consequential.

## Gate implementation

Proceed only when:

- the repository and authority paths are resolved;
- dirty work is isolated or explicitly accepted;
- the objective and non-goals are coherent;
- every consequential action is authorized;
- acceptance checks can distinguish success from intent.

If remote control is unavailable, prepare a local patch without claiming application. If an authority conflict affects the result, stop and request resolution.

## Maintain continuity

Update progress and decision logs during long work. Preserve completed artifacts and record resumable state. Do not rely on chat history as the only task record.

## Boundaries

Do not use this skill as the main workflow for:

- a single read-only factual answer after all source paths are known;
- a tiny documentation typo with no wider behavioral effect;
- implementation verification after a complete plan already exists;
- review-bundle validation without repository planning needs.

Use the more specific skill in those cases.
