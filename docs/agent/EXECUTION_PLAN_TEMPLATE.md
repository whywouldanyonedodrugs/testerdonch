# Execution Plan Template

Use this template for multi-file, long-running, ambiguous, data-changing, or economic work. Keep the plan in a repository-approved plan location discovered during preflight. `REPO_DEPENDENT_PLACEHOLDER`: no plan directory was verified while this template was prepared.

```markdown
# <Task title>

Status: proposed | approved | in_progress | blocked | complete
Owner:
Created UTC:
Updated UTC:
Repository root and commit:

## Received task and archive context

- Exact task specification path/hash:
- Donch decision and evaluation IDs:
- Source files/revisions/hashes or transferred inputs:
- Rejected alternatives and unresolved gaps:
- Human approval scope:
- Durable task-archive path:
- Approved Drive archive target, or `remote_handoff_blocked` reason:

## Objective

<One observable outcome.>

## Non-goals

- <Explicitly excluded work.>

## Assumptions and unresolved questions

| Item | Status: verified/inferred/proposed/unavailable | Evidence | Failure response |
|---|---|---|---|

## Authority paths

| Path | Role | Version/hash | Verified current? |
|---|---|---|---|

## Repository state preservation

- Original checkout path, branch, and commit:
- Staged / unstaged / untracked / conflicted counts:
- Recovery bundle and manifest path/hash:
- Sensitive or protected paths excluded from content capture:
- Original checkout left unchanged: yes/no
- Isolated worktree path, branch, and base commit:
- Overlap with existing dirty paths:
- Safe-isolation rationale:

## Scope and boundaries

- Venue:
- Train interval:
- Protected cutoff:
- Economic run authorized: yes/no; exact authorization:
- Protected outcomes authorized: no unless formal policy change is cited
- Remote writes authorized: yes/no; exact destination and identity prerequisite:
- Forbidden actions:

## Hypothesis-development admission (economic tasks only)

- Protocol phase and fold role:
- Compelled actor or structural mechanism:
- Measurement semantics valid:
- Event frequency consistent with mechanism:
- Raw magnitude economically relevant versus plausible costs:
- Development route or explicit one-shot reason:
- Horizon justified:
- Component controls defined:
- Payoff archetype frozen:
- Explored-cell registry and multiplicity family:
- Untouched-block, purging, embargo, and no-backward-transfer evidence:
- Separate exact human approval for each outcome-bearing phase:

## Files expected to change

- `<path>`: <reason>

## Milestones

### M1 — <name>

- Action:
- Acceptance criteria:
- Verification command or inspection:
- Expected artifact path:
- Failure response:

### M2 — <name>

- Action:
- Acceptance criteria:
- Verification command or inspection:
- Expected artifact path:
- Failure response:

## Validation commands

Commands must come from the checked-out repository. Do not invent them.

| Command | Why required | Expected result | Actual result |
|---|---|---|---|

## Artifact paths

| Artifact | Path | Manifest/hash requirement | Retention/handoff |
|---|---|---|---|

## Risk and rollback

- User work at risk:
- Data or output mutation risk:
- Rollback method that preserves evidence:
- Dirty-checkout recovery and restoration evidence:
- Remote collision/overwrite policy:

## Decision log

| UTC | Decision | Evidence | Consequence |
|---|---|---|---|

## Progress log

| UTC | Milestone | Result | Next action |
|---|---|---|---|

## Completion record

- Acceptance criteria met:
- Files changed:
- Commands actually run:
- Artifacts and SHA-256:
- Economic runs launched:
- Protected outcomes inspected:
- Remote writes:
- Local task archive and manifest:
- Dirty-repository recovery bundle and manifest:
- Isolated worktree, branch, and base commit:
- Verified remote archive path/hash:
- Registries or continuity files updated:
- Unresolved blockers:
- Next self-contained task, if justified:
```

Do not mark the plan complete until every acceptance criterion is either met or explicitly recorded as blocked. The agent owns this record. Preserve the received task, archive context, decisions, progress, validation, manifest, completion, and next action so another task can continue without chat history or human filing.
