# QLMG Backtesting Repository Instructions

This repository is governed as a historical multi-platform backtesting and research project. Kraken and Capital.com mechanics remain platform-specific. Keep this root file short; detailed procedures live under [docs/agent/](docs/agent/).

## Required Reads

- Use [docs/agent/REPOSITORY_MAP.md](docs/agent/REPOSITORY_MAP.md) to locate current authority, commands, archives, and repository state.
- Use [docs/agent/DIRTY_REPOSITORY_RECOVERY.md](docs/agent/DIRTY_REPOSITORY_RECOVERY.md) before changing files when a checkout is dirty or ownership is uncertain.
- Use [docs/agent/EXECUTION_PLAN_TEMPLATE.md](docs/agent/EXECUTION_PLAN_TEMPLATE.md) for multi-file, long-running, ambiguous, data-changing, or economic work.
- Use [docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md](docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md) for data, universe, signal, control, funding, or outcome work.
- Use [docs/agent/RUN_AND_ARTIFACT_CONTRACT.md](docs/agent/RUN_AND_ARTIFACT_CONTRACT.md) for runs, manifests, review bundles, archives, and handoffs.
- Use [docs/agent/DRIVE_HANDOFF_TARGET.md](docs/agent/DRIVE_HANDOFF_TARGET.md) for the approved default Donch handoff destination and non-overwrite procedure.
- Use [docs/agent/CODE_REVIEW.md](docs/agent/CODE_REVIEW.md) for review tasks.
- Use [docs/agent/KNOWN_FAILURE_PATTERNS.md](docs/agent/KNOWN_FAILURE_PATTERNS.md) before changing event generation, controls, or accounting.

## Simplicity and surgical scope

Choose the smallest compliant change that satisfies the current acceptance criteria.

- Do not add a framework, abstraction, configuration option, registry, or document unless the current task requires it.
- Every changed line and generated artifact must trace directly to the task objective or a binding safety requirement.
- Do not repair adjacent defects, perform unrelated cleanup, or prepare speculative future functionality.
- If a guard and focused tests solve the problem, stop there.
- If the task expands materially, split it and request approval for the new scope.
- Scale plans, archives, reviews, and completion reports to the task. Small fixes require small records.
- Simplicity never overrides protected-data, authorization, evidence, or rollback boundaries.

Before completion, ask: “Would a senior engineer consider this solution unnecessarily elaborate?” If yes, simplify it.

## Hard Boundaries

- Approved research platforms: Kraken derivatives and Capital.com instruments present in a verified acquisition manifest for the configured account and legal environment.
- Rankable interval: `2023-01-01T00:00:00Z` inclusive through `2026-01-01T00:00:00Z` exclusive.
- Protected period: `2026-01-01T00:00:00Z` onward. Do not use it for strategy scoring, tuning, controls, selection, validation, or portfolio work.
- July 2026 Kraken capture may be used only for strategy-agnostic execution calibration. Capital.com data from 2026 onward default to data-engineering-only.
- Real Capital.com import remains blocked pending a verified acquisition package. Every economic or directed cross-platform contract requires separate explicit approval.
- Paid historical vendor data is prohibited unless a later formal policy change supersedes this rule. Live trading, private-account actions, order placement, risk changes, pushes, merges, destructive Git operations, and remote overwrites require explicit task authorization.
- Documentation or coding approval does not authorize an economic screen or protected-outcome inspection.
- This file does not authorize an economic run.

## Evidence Invariants

- Enforce `source_close_ts <= decision_ts` and `feature_available_ts <= decision_ts`.
- Require a manifest-authorized platform and the selected platform adapter before any rankable payload reader runs.
- Keep last/trade, mark, index, fill, liquidation, and funding semantics separate.
- Do not collapse Kraken exchange mechanics and Capital.com OTC CFD bid/ask, calendar, financing, expiry, or corporate-action mechanics into a generic execution model.
- Require point-in-time universe, listing, delisting, status, and lifecycle handling; do not treat a current roster as historical truth.
- Freeze candidate identity and real control identity before outcomes.
- Use actual executable exits for non-overlap. Do not use maximum-hold preblocking, summary rows as trades, placeholder controls, event caps, same-bar execution heroics, or silent artificial boundary closes.
- Preserve old artifacts and run roots through supersession records rather than deletion.

## Agent Ownership

The backtesting agent owns routine technical administration for approved work: repository discovery, dirty-state preservation, plans, logs, registries, manifests, validation, review records, task archives, and approved Drive handoffs. The human supplies trading sanity checks, broad direction, authority-conflict decisions, and consequential approvals.

Never ask the human to interpret routine diffs, maintain logs, reconcile manifests, or choose safe Git mechanics. Escalate only when evidence is unavailable and the next step would materially change, destroy, publish, trade, or inspect protected outcomes.
