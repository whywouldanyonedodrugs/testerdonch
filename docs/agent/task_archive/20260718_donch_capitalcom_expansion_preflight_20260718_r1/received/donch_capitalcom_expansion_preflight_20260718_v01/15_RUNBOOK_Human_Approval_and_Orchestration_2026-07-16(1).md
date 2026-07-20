---
status: current minimal-human approval and orchestration runbook
date: 2026-07-16
revision: 2.0
scope: Donch project, backtesting repository, capture repository, transfers, approvals, and evidence review
authority: operating contract and user master prompt
supersedes: prompts that assume shared files, implicit remote authority, or operator diagnosis of implementation details
provenance: master prompt; access report; continuity brief; backtester and capture audits; approved 2026-07-16 minimal-human workflow
known limitations: the observed backtesting checkout is heavily dirty; repository instructions, supported commands, and the exact approved Drive archive target must be rechecked in an isolated target environment
---

# Human Approval and Orchestration Runbook

## Environment map

| Environment | Primary work | Does not imply |
|---|---|---|
| Donch project | Analysis, planning, verification design, source governance, prompts, continuity. | Repository access or implementation. |
| Backtesting repository | Historical Kraken data and research engineering. | Access to Donch files, capture credentials, protected outcomes, or authority to run economics. |
| Capture repository | Public forward capture, integrity, storage, and execution calibration. | Trading, private endpoints, alpha discovery, or backtesting-repo access. |
| Human operator | Supplies trading sanity checks and broad direction; resolves consequential conflicts; approves frozen economic runs, protected access, live actions, destructive operations, publication, and named external writes. | Routine filing, transfer, log reconciliation, registry maintenance, or implementation diagnosis. |

Each remote task must be self-contained. Include the relevant files or their exact transferred location, not a statement that they “exist in the Donch project.”

Donch supplies task-relevant evaluations, decision IDs, rejected alternatives, source revisions and hashes, evidence gaps, and approval scope in an `archive_context` block. The receiving agent stores that block with the durable task record.

Routine technical administration belongs to the agents. They maintain plans, progress and decision logs, changed-file and command records, tests, manifests, registries, continuity files, packages, and transfer verification. Do not ask the human to copy routine files, reconstruct context from chat, or reconcile technical logs.

## Standard workflow

### 1. Frame the task

Record:

```text
objective
environment
authorized_actions
forbidden_actions
venue
rankable_and_protected_periods
source_authority
input_paths_or_bundle
expected_outputs
acceptance_criteria
verification
failure_response
approval_gates
archive_context
agent_owned_task_archive
approved_Drive_handoff_or_local_retention
```

If a consequential choice changes the mechanism, data boundary, deployment risk, or protected policy, ask the human before proceeding.

### 2. Read-only preflight

The target agent checks:

- repository root and applicable `AGENTS.md` hierarchy;
- Git status, branch, commit, remotes, and uncommitted work;
- current plan, rules, skills, and config layers;
- exact data and artifact paths;
- protected-data and credential boundaries;
- available test commands from repository evidence.

Do not overwrite uncommitted user work. If direct access or isolation is unavailable, create a local ready-to-apply package.

### 3. Plan before change

For each milestone state action, acceptance criteria, verification, and failure response. The agent creates and maintains the plan in the repository task archive. Do not begin an economic run or production action while the plan is still resolving boundaries.

### 4. Implement surgically

- Touch only approved files.
- Preserve repository style and machine contracts.
- Do not refactor unrelated code.
- Keep deterministic work in scripts and contextual judgment in review.
- Record assumptions, deviations, decisions, and rejected alternatives in the task archive.

### 5. Verify

Run the exact repository-supported tests. Record commands, exit codes, counts, failures, and artifacts. A missing test command is a blocker or a request for repository evidence; it is not permission to invent one.

### 6. Independent review

Review boundary, identity, protected data, secrets, machine/prose agreement, evidence claims, and rollback. Use a separate review pass for consequential changes.

### 7. Close the technical record and request any human decision

The agent closes the local task archive, updates affected registries and continuity files, generates the manifest and review record, and performs any exact authorized Drive upload with end-to-end verification. Present the human with the result, material risks, trading sanity-check questions, what remains unavailable, and the smallest exact approval requested. Do not make approval implicit in a technical handoff.

## Approval matrix

| Action | Required authority |
|---|---|
| Read public/local approved sources | Within scoped task, subject to protected exclusions. |
| Create local curation or ready-to-apply files | Within scoped workspace task. |
| New economic screen or backtest | Explicit human approval of frozen hypothesis contract. |
| Protected outcome access | Separate explicit human approval and policy-compliant protocol. |
| Redefine 2026 protected use | Human policy decision plus replacement holdout and access audit. |
| Restart public capture | Human approval after code, fixture/smoke, storage, and verification review. |
| Network-write smoke test | Explicit scope and human approval. |
| Private Kraken endpoint or credential use | Explicit human authorization; current contract does not provide it. |
| Any order or risk change | Not authorized. Requires a new governing policy and explicit approval. |
| Apply local package to the observed dirty repository | Human approval after remote Git and instruction preflight; use an isolated safe worktree. |
| Commit, push, merge, or deploy | Explicit human approval for each workflow step. |
| Upload/replace/delete Donch project sources | Human review and explicit approval. |

## Self-contained remote task template

```text
ROLE
You are working in [backtesting|capture] repository.

OBJECTIVE
[one concrete deliverable]

BOUNDARY
Kraken only.
Rankable history: 2023-01-01 through 2025-12-31.
2026 onward: protected.
Live trading/orders: not authorized.

AUTHORIZED
[exact reads/writes/tests]

FORBIDDEN
[economic run, protected outcomes, capture restart, orders, unrelated refactor, push, etc.]

INPUTS
[attached files, hashes, exact paths, authoritative roots]

CURRENT FACTS
[decision and known blockers that must be preserved]

ARCHIVE_CONTEXT FROM DONCH
[decision IDs, evaluation conclusions, controlling source revisions and hashes, required excerpts or files, rejected alternatives, gaps, human trading questions, and approval scope]

TASK
[ordered implementation requirements]

ACCEPTANCE CRITERIA
[machine-verifiable conditions]

VERIFICATION
[known repo commands or requirement to discover them from repo docs]

OUTPUT
[files, patch, report, manifest, updated registries and continuity, local task archive, verified remote archive or exact blocker]

FAILURE RESPONSE
Stop before consequential action, preserve evidence, and report the exact blocker.
```

## Transfer package

The responsible agent creates a dated, descriptive archive or folder with:

- README first;
- manifest of path, size, and SHA-256;
- source and destination environment;
- source commit/dirty state where applicable;
- included and excluded scope;
- authority and supersession notes;
- application instructions;
- verification commands;
- rollback method;
- secret scan result;
- no credentials or private tokens.

Record end-to-end identity at the receiving environment. An uploaded filename is not proof that the receiver has the same bytes. Once a stable Drive archive root and non-secret write identity have been explicitly approved and verified, the agent performs routine authorized transfers and records local and remote paths, UTC time, size, SHA-256, tool/version, and identity label. If that one-time setup is incomplete, retain the local package and return one bounded setup request; do not shift routine transfer work to the human.

## Repository application checklist

Before applying a ready-to-use package:

1. verify repository root and applicable instructions;
2. capture `git status`, branch, commit, and remotes;
3. preserve uncommitted work or use an isolated branch/worktree;
4. compare packaged assumptions with current paths and contracts;
5. apply only in-scope files;
6. run repository-supported validation;
7. inspect the diff and secret scan;
8. record untested or unavailable checks;
9. request human approval before commit/push/merge.

## Capture-specific handoff

Do not transfer the live host's Google Drive configuration or token to the backtesting machine. Use a distinct least-privilege identity approved for the exact archive root. The responsible agent transfers a compact hash-verified package when authorized. Raw archives remain in place unless a separate storage decision authorizes movement.

## Research-specific handoff

Include the frozen hypothesis contract, family registry row, data-capability rows, authoritative run lineage, multiplicity family, protected-data policy, and allowed claim wording. Do not send only a prose idea and ask the coding agent to choose thresholds.

## Failure and incident handling

If an agent encounters protected outcomes, credentials, an unexpected dirty tree, mismatched hashes, unknown production process state, or a conflict between machine and prose authority:

1. stop the affected action;
2. preserve non-secret evidence;
3. state what was observed and what was not;
4. do not infer a safe state;
5. propose the smallest recovery or request the human decision.

## Completion report

Use:

```text
objective_status
files_changed
commands_and_tests_run
results
originals_or_archives_modified
protected_data_access
economic_runs
capture_or_order_actions
Git_actions
local_task_archive
verified_remote_archive_or_blocker
registries_and_continuity_updated
unavailable_checks
remaining_blockers
rollback
human_trading_sanity_checks_requested
approval_requested
next_self_contained_task_if_justified
```

Completion means acceptance criteria were verified. It does not mean the next consequential step is authorized.
