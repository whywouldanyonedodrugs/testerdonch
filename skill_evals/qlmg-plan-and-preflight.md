# Skill Evaluation: qlmg-plan-and-preflight

Purpose: test routing and required behavior. These prompts are evaluation cases, not authorization to execute them.

## Prompts that should trigger

| ID | Prompt | Expected routing |
|---|---|---|
| P1 | “The QLMG repository may be at `/opt/testerdonch` and has hundreds of staged and unstaged changes. Preserve everything, create a recovery bundle and safe worktree, then plan a documentation-only governance update without asking me to review the code.” | Trigger this skill; verify the root, follow dirty-repository recovery, validate preservation, and record the isolated base before producing the plan. |
| P2 | “Prepare an outcome-free plan for the Kraken lifecycle-authority and protected-input separation phase. Include milestones, acceptance criteria, artifacts, and rollback.” | Trigger this skill; plan a consequential data-authority phase without launching research. |
| P3 | “This repair may run for several hours and touch manifests, loaders, and docs. Create a resumable execution plan and decide how to isolate the work safely.” | Trigger this skill; record decision and progress logs plus repository-state prerequisites. |

## Prompts that should not trigger

| ID | Prompt | Expected routing |
|---|---|---|
| N1 | “What date does the protected strategy-selection period begin? Answer from the active instructions.” | Do not trigger; give the read-only fact. |
| N2 | “Correct one spelling error in `docs/agent/CODE_REVIEW.md`; no behavior changes.” | Do not trigger as the main workflow; make the bounded documentation edit. |
| N3 | “Validate this already-built QLMG review ZIP and upload it to the explicitly approved Drive destination.” | Route to `$qlmg-review-package-verification`; use planning only if repository changes or unresolved scope require it. |

## Expected files and actions

- Read the applicable `AGENTS.md` chain and `docs/agent/REPOSITORY_MAP.md`.
- Read `docs/agent/DIRTY_REPOSITORY_RECOVERY.md` when the checkout is dirty or ownership is uncertain.
- Resolve the Git root rather than accepting `/opt/testerdonch` from the prompt.
- Inspect and preserve branch and dirty state; create a validated recovery bundle and task-authorized isolated worktree without changing the original checkout.
- Create or update a plan using `docs/agent/EXECUTION_PLAN_TEMPLATE.md` in a repository-supported location.
- Record authority paths, expected files, repository-discovered validation commands, artifacts, risk, rollback, decisions, and progress.
- Do not run an economic screen, inspect protected outcomes, or write remotely.

## Acceptance checks

- All three positive prompts route to the skill and all three negative prompts do not route to it as the primary skill.
- The plan contains every required template field and uses status labels for assumptions.
- Repository paths and commands remain `REPO_DEPENDENT_PLACEHOLDER` until verified.
- Dirty user work is independently recoverable, the original checkout is unchanged, and the isolated branch/base are recorded.
- Planning does not imply economic, holdout, remote-write, push, or merge authorization.
