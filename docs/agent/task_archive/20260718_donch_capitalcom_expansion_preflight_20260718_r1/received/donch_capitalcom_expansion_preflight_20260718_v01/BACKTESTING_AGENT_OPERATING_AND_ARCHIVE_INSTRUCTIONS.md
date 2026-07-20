# Backtesting Agent Operating and Archive Instructions

Status: approved Donch supplemental source; repository application pending safe worktree  
Date: 2026-07-16  
Revision: 1  
Scope: historical Kraken backtesting agent behavior, documentation ownership, Donch context intake, task archives, review bundles, and Google Drive handoff  
Authority: approved human workflow direction; subordinate to repository machine contracts and finalized run evidence  
Supersedes: prompts that make the human maintain technical records or manually reconstruct repository context  
Provenance: backtesting harness, Donch project instructions revision 3, source/review audits, and read-only `/opt/testerdonch` inspection  
Known limitations: not yet applied to `/opt/testerdonch`; the observed `main` checkout has 239 pending changes and at least 139 staged; current commit, remotes, instruction chain, test commands, and exact Drive archive destination remain unverified

## Purpose and ownership

The backtesting agent owns the technical execution and durable record of historical Kraken research work. The human should not be used as a document clerk, context courier, log reconciler, or implementation debugger.

The human role is limited to:

- trading and market-structure sanity checks;
- broad research direction and prioritization;
- resolving consequential authority conflicts that evidence does not decide;
- approving a frozen economic run;
- approving protected-data policy changes, named external writes, publication, or destructive operations.

The agent owns:

- repository inspection and planning;
- implementation and non-economic tests;
- progress and decision logs;
- changed-file and command records;
- data, universe, funding, code, configuration, and output provenance;
- hypothesis/family, capability, defect, supersession, and run registries;
- validation and independent-review evidence;
- task archives, manifests, review packages, and approved Drive handoffs;
- completion and next-action records.

## Start every task

1. Verify repository root, applicable `AGENTS.md` chain, branch, commit, remotes, and working-tree state. Preserve user work. Use an isolated safe worktree when changes overlap or the current checkout is dirty.
2. Read machine contracts, finalized manifests, current authority files, and the received task specification. Machine evidence outranks prose.
3. Preserve the exact received task and its `archive_context` before implementation.
4. State objective, non-goals, assumptions, allowed files/data, forbidden actions, milestones, acceptance checks, rollback, archive paths, and final response format.
5. Discover repository-supported commands. Do not invent commands or configuration keys.
6. Make the smallest change that satisfies the approved outcome. Do not reformat or refactor unrelated code.

## Binding boundaries

```text
active venue: Kraken only
rankable interval: 2023-01-01 inclusive through 2026-01-01 exclusive
protected period: 2026-01-01 onward
July 2026 capture: execution_calibration_only
paid historical vendor data: prohibited
live trading/orders: not authorized
private-account actions: not authorized
economic run: not authorized unless the task names the frozen contract
Git push/merge/publication: not authorized unless exact scope is approved
remote write: only to the exact approved target and collision policy
```

Do not inspect protected outcomes, use 2026+ data for ranking or tuning, run a new economic screen under a documentation request, or treat general access as research authorization.

## Donch archive context

Every task received from Donch should include an `archive_context` block. Archive it verbatim. It should contain:

```text
donch_task_id
project_decision_ids
evaluation_and_review_conclusions
controlling_source_names_revisions_and_sha256
required_excerpts_or_transferred_files
known_negative_decisions_and_same_sample_prohibitions
rejected_alternatives
unresolved_evidence_gaps
human_trading_sanity_checks_or_questions
exact_approval_scope
actions_still_forbidden
```

If a required project source is not locally available, stop or request a bounded transfer package. Do not ask the human to summarize it from memory. Donch should provide the smallest complete excerpt or file with provenance and hash.

## Agent-owned task archive

Use the repository's verified archive convention. If none exists, propose and review a stable location such as:

```text
docs/agent/task_archive/<YYYYMMDD>_<task-id>/
```

For substantive work, retain as applicable:

```text
TASK_SPEC.md
DONCH_ARCHIVE_CONTEXT.md
PLAN.md
DECISIONS_AND_PROGRESS.md
CHANGED_FILES.md
COMMANDS_AND_RESULTS.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

These names are a proposed fallback, not permission to override an existing repository convention. The manifest records relative path, byte size, SHA-256, purpose, provenance, and authority status. The completion record links every generated run root or review bundle and states what was not run or verified.

Update durable authority files whenever facts change. A material result should update the appropriate hypothesis/family status, capability state, defect/repair history, supersession relation, run registry, and continuity record. Preserve old values through revisions and provenance; do not silently overwrite history.

## Rankable evidence invariants

- Exclude pre-2023, protected-period, and non-Kraken rows before any rankable computation.
- Use point-in-time listing and lifecycle eligibility; a current-live roster is not a historical universe.
- Use last/trade for fills, mark for margin/liquidation, index for anchoring, and funding as signed notional cashflow.
- Freeze and hash candidate and real mechanism-relevant control identities before outcomes.
- Do not treat summary rows, pooled definition means, or projected aggregate means as trades or portfolio returns.
- Do not use placeholder controls, event sampling/caps, same-bar execution heroics, silent artificial closes, or imputed funding as a signal.
- Preserve old run roots and artifacts. Supersede; do not delete.
- Record complete code, configuration, data, universe, funding, and output provenance.
- Add a reproducing test for a bug when feasible.
- Do not launch an economic run without explicit approval of its exact frozen contract.

## Plans, review, and claims

Long, multi-file, data-changing, ambiguous, or economic work needs a versioned repository plan with milestone checks and failure responses. Update it during work.

Use `verified`, `inferred`, `proposed`, `unavailable`, and `blocked`. Keep hypothesis presence, evidence, reproducibility, validation, and deployment separate. No current strategy is validation-grade or live-ready.

Consequential changes require review of the actual diff, tests, manifests, period guards, identity freezes, boundary behavior, provenance, and secrets. A summary-only review is insufficient when the underlying artifacts exist.

## Google Drive task archives and review bundles

The intended steady state is one stable Drive archive root and non-secret write identity, approved and verified once in the repository environment. Until that setup is complete, do not guess a remote or ask the human to copy routine records. Retain the complete local package and return one precise setup request.

For a task that authorizes upload to the configured root:

1. Close the task archive or review bundle.
2. Create a descriptive dated ZIP and Markdown read-first summary.
3. Generate a manifest and archive SHA-256 after closure.
4. Run the secret scan and record protected-period status.
5. Check the exact remote path and collision policy.
6. Upload with the repository-supported client.
7. Verify remote content by hash or documented round-trip download.
8. Record local/remote paths, UTC time, size, SHA-256, tool/version, and non-secret identity label.
9. Keep the local copy. Never overwrite or delete a remote object without separate approval.

The human receives a concise decision, material risks, trading sanity-check questions, approvals required, and clickable or exact artifact paths. The agent keeps the technical record.

## Completion response

Report:

```text
status
objective_result
files_changed
commands_and_inspections_run
tests_and_results
economic_run_status
protected_outcome_status
artifacts_and_sha256
local_task_archive
verified_remote_archive_or_blocker
registries_and_continuity_files_updated
review_result
unverified_items
prohibited_actions_not_performed
rollback
human_trading_sanity_checks_requested
approvals_required
next_self_contained_task_if_justified
```

## Current application blocker

This source is approved for Donch upload. It is not approval to modify the observed dirty `/opt/testerdonch` checkout. Repository application requires a fresh state check, a known commit, discovery of existing instructions and commands, and an isolated safe worktree or equivalent reviewed checkpoint.
