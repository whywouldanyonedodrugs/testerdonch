---
name: qlmg-code-change-verification
description: Implement and verify bounded code or test changes in the QLMG historical backtester. Use for bug fixes, loaders, evidence contracts, artifact writers, deterministic utilities, refactors, or tests where success must be shown with repository-supported commands, focused fixtures, diff review, and preserved provenance without launching a new economic screen.
---

# QLMG Code Change Verification

## Prepare the change

1. Read the active instructions and task plan. Use `$qlmg-plan-and-preflight` first when the task is multi-file, long, ambiguous, or data-changing.
2. Read [CODE_REVIEW.md](../../../docs/agent/CODE_REVIEW.md) and the relevant contract or finalized manifest.
3. Confirm the change does not require an economic run. Code and fixture tests do not authorize one.
4. Discover supported test, lint, and format commands from the checked-out repository. Do not invent commands.
5. Identify unrelated working-tree changes and leave them untouched.

## Implement surgically

- Change only files required by the objective.
- Preserve repository style and public interfaces unless the task authorizes a change.
- Fail closed when uncertainty could admit protected, wrong-venue, future-aware, duplicate, or unproven evidence.
- Add a minimal reproducing test for a bug when feasible.
- Prefer synthetic or fixed fixtures for boundary, timestamp, identity, funding, and failure-path tests.
- Preserve old run roots and artifacts. Write a new versioned root when an output contract changes.

If code changes candidate generation, control identity, outcome accounting, rankability, or economic interpretation, also use `$qlmg-rankable-backtest-contract`.

## Verify in layers

Run the smallest repository-supported checks that prove the change, then broaden in proportion to risk:

1. reproducing test or focused fixture;
2. affected module or contract tests;
3. relevant broader suite;
4. documentation, schema, or manifest checks when outputs change;
5. final diff review.

Do not use protected outcomes as a fixture. Do not launch a historical screen merely to exercise code. Record the exact command, exit status, and concise result. If a command is unavailable, report it as unrun and explain why.

## Review the diff

Check:

- scope and unrelated changes;
- train and protected boundary behavior;
- Kraken-only routing and point-in-time availability;
- candidate and control freezes where applicable;
- actual-exit non-overlap and boundary censoring;
- last, mark, index, and funding roles;
- atomic output, manifest, provenance, and supersession;
- secret exposure and remote side effects.

Repair failures before completion. A passing unit test does not override a contract violation.

## Report

List files changed, tests added, commands actually run, results, artifacts, hashes, unrun checks, and remaining risks. State explicitly that no economic run, protected-outcome inspection, remote write, push, or merge occurred unless the task separately authorized and recorded one.

## Boundaries

Do not use this skill as the main workflow for:

- a documentation-only edit;
- planning a new hypothesis or running a rankable screen;
- interpreting strategy performance;
- building or handing off a review package without code changes.
