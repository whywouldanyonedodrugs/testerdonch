# Skill Evaluation: qlmg-rankable-backtest-contract

Purpose: test routing and required behavior. These prompts are evaluation cases, not authorization to execute them.

## Prompts that should trigger

| ID | Prompt | Expected routing |
|---|---|---|
| P1 | “Implement and run the explicitly approved, frozen Kraken train-only screen described in the attached contract. Reject all rows outside 2023–2025 and record candidate/control hashes.” | Trigger this skill; verify the exact authorization and frozen contract before any outcome work. |
| P2 | “Review this diff: it changes control matching, candidate deduplication, and net-R scoring, but no run is requested.” | Trigger this skill for semantic review; do not run economics. |
| P3 | “Add boundary fixtures for a candidate opened on 2025-12-31 whose hold and funding interval would cross into 2026.” | Trigger this skill; verify the frozen censor/drop rule with synthetic rows. |

## Prompts that should not trigger

| ID | Prompt | Expected routing |
|---|---|---|
| N1 | “Refactor a log formatter used by offline utilities; output meaning and research data are unchanged.” | Route to `$qlmg-code-change-verification` only. |
| N2 | “Fix a typo in the repository map.” | Do not trigger; use a bounded documentation edit. |
| N3 | “Verify an existing review ZIP, its manifest, and its authorized Google Drive upload.” | Route to `$qlmg-review-package-verification`. |

## Expected files and actions

- Read machine evidence and signal-state contracts plus finalized manifests.
- Read `docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md`, `RUN_AND_ARTIFACT_CONTRACT.md`, and `KNOWN_FAILURE_PATTERNS.md`.
- Verify exact economic authorization; without it, limit work to plans, code, synthetic fixtures, and mechanical checks.
- Freeze mechanism, definitions, interval, universe, candidate/control identity, costs, funding, boundaries, gates, and provenance before outcomes.
- Enforce Kraken-only, `[2023-01-01, 2026-01-01)`, PIT availability, actual-exit non-overlap, real controls, and no event sampling.
- Preserve old roots and write explicit supersession for any new run root.

## Acceptance checks

- All three positive prompts trigger and all three negative prompts do not trigger as the primary skill.
- P2 and P3 do not cause an economic run.
- The agent refuses or pauses outcome work when exact authorization or a frozen contract is missing.
- Protected rows cannot reach signals, controls, outcomes, summaries, or export.
- Imputed funding cannot activate a signal; last, mark, index, and funding roles remain separate.
- Completion reports exact commands, gates, provenance, evidence level, run status, and protected-outcome status.
