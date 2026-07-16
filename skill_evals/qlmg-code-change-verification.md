# Skill Evaluation: qlmg-code-change-verification

Purpose: test routing and required behavior. These prompts are evaluation cases, not authorization to execute them.

## Prompts that should trigger

| ID | Prompt | Expected routing |
|---|---|---|
| P1 | “Fix the funding loader so protected rows are filtered before strategy processing. Add synthetic cutoff tests and do not run a backtest.” | Trigger this skill and also consult the protected-period contract. |
| P2 | “Repair the maximum-hold preblock bug in the event engine and add a regression test that proves a later signal is accepted after an early actual exit.” | Trigger this skill; also use `$qlmg-rankable-backtest-contract` because candidate identity changes. |
| P3 | “Make the package manifest writer atomic and add fixture tests for byte sizes and SHA-256 values. Do not upload anything.” | Trigger this skill; verify with repository-supported fixture tests. |

## Prompts that should not trigger

| ID | Prompt | Expected routing |
|---|---|---|
| N1 | “Write a plan for a multi-day lifecycle data-authority phase, but do not change code yet.” | Route to `$qlmg-plan-and-preflight`. |
| N2 | “Run a new threshold sweep for a failed-bounce strategy and rank the results.” | Route to `$qlmg-rankable-backtest-contract`; require exact economic authorization. |
| N3 | “Check whether a review bundle contains all required raw verification extracts and prepare its Drive handoff.” | Route to `$qlmg-review-package-verification`. |

## Expected files and actions

- Read the applicable contracts and inspect the exact diff scope.
- Modify only repository-verified implementation and test paths.
- Add fixed or synthetic regression fixtures; do not use protected strategy outcomes.
- Discover test and lint commands from repository-owned files.
- Run focused checks, relevant broader checks, and final diff review in proportion to risk.
- Preserve finalized artifacts and avoid remote writes, pushes, and merges.

## Acceptance checks

- All three positive prompts route to the skill and all three negative prompts route elsewhere.
- Candidate or outcome-semantic changes also load the rankable contract.
- No remembered test command is asserted as repository-supported.
- Each defect fix has a reproducing test when feasible.
- The final record distinguishes commands run, commands unrun, artifacts, and unresolved risks.
- No economic run or protected-outcome inspection is used as verification.
