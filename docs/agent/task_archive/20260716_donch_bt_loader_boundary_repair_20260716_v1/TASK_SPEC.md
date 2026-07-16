# Backtesting Task: Minimal Rankable-Loader Boundary Repair and Readiness Rerun

```text
task_id: donch_bt_loader_boundary_repair_20260716_v1
date_utc: 2026-07-16
target_environment: backtesting Codex
mode: direct_apply_in_isolated_branch_or_worktree
repository_root: /opt/testerdonch
expected_main_and_origin_main_commit: 8cf3e227105fd7626445d27c8caf4c28bccc2ecb
economic_run_authorized: no
protected_outcome_access: no
data_acquisition_authorized: no
capture_access_authorized: no
local_commit_authorized: yes, one task-scoped commit after all gates pass
push_or_merge_authorized: no
```

## Objective

Repair only the demonstrated rankable-loader boundary defect, add focused synthetic tests, and rerun the archived first-wave readiness verification.

The repair must ensure that outcome-bearing protected data cannot be opened by a rankable loader, while preserving the less restrictive authority boundary for pre-2023 and non-Kraken rows.

Do not address any other readiness, package, universe, multiplicity, data-acquisition, feature, or governance gap.

## Current state and authority

Treat governance integration as complete at:

```text
8cf3e227105fd7626445d27c8caf4c28bccc2ecb
```

At task start, independently verify:

```text
repository root
applicable AGENTS.md chain
main and origin/main commit
branch and remotes
working-tree state
machine contracts
supported test and review commands
task-archive convention
archived readiness-task path and manifest
```

The repository must be clean and `main` plus `origin/main` must match the expected commit. Use an isolated task branch or worktree from that exact commit. If identity or cleanliness differs, stop before editing.

Use the archived record for:

```text
donch_bt_first_wave_readiness_20260716_v1
```

as the evidence source for the demonstrated defect. Verify its artifact manifest and record the exact report path and SHA-256.

## Precise boundary semantics

These requirements supersede any broader wording in the earlier readiness prompt for this repair task. A higher-authority machine contract may impose a stricter rule, but only if the agent identifies and cites that exact contract and test.

### 1. Protected 2026+ outcome-bearing data

A rankable loader must reject the input **before invoking the underlying payload reader** when existing file/path/manifest metadata identifies any of the following:

```text
protected period: 2026-01-01 onward
protected strategy outcomes
execution-calibration-only input
prospective-shadow input not authorized for rankable research
mixed rankable/protected file
externally unrankable file
unknown or unprovable rankability/purpose
```

Required observable result:

```text
payload_reader_calls: 0
feature_or_scoring_calls: 0
return_or_ranking_calls: 0
result: explicit fail-closed error
```

Do not open a real protected file to prove this.

### 2. Pre-2023 rows

Pre-2023 rows must be rejected before they reach:

```text
feature selection or construction for rankable use
signal scoring
return analysis
parameter selection
candidate or family ranking
```

This task does **not** require pre-open rejection of an otherwise authorized historical file merely because it contains or may contain pre-2023 rows, unless an identified higher-authority machine contract explicitly requires that.

Tests must therefore assert the downstream boundary, not require a particular reader-call count:

```text
payload_reader_calls: implementation-dependent under existing authority
feature_or_scoring_calls_for_pre_2023_rows: 0
return_or_ranking_calls_for_pre_2023_rows: 0
result: rows rejected or excluded with explicit evidence
```

Do not redesign data partitioning to obtain an earlier rejection.

### 3. Non-Kraken rows

Non-Kraken rows must be rejected before feature selection/construction, scoring, return analysis, parameter selection, or ranking.

As with pre-2023 rows, do not impose a new universal pre-open requirement unless an exact higher-authority machine contract requires it.

Tests must assert that no non-Kraken row reaches the downstream rankable consumer. They must not require the reader spy to remain at zero.

### 4. Mixed or externally unrankable files

Where existing file/path/manifest authority classifies the entire file as mixed, protected, external/unrankable, calibration-only, prospective-only, or unknown, fail closed before the payload reader is called.

Do not build a new catalog, purpose registry, data lake, partitioning system, or generalized classification framework. Use the smallest existing authority hook available in the current loader path.

If the current code provides no file-level metadata hook capable of enforcing this demonstrated case, add the smallest local contract needed at the affected loader boundary. Do not generalize it beyond the active reader path.

### 5. Funding

Apply equivalent protection to every funding reader used by rankable research:

- protected, mixed, calibration-only, external/unrankable, or unknown funding files fail before the underlying funding reader opens them;
- pre-2023 and non-Kraken funding rows cannot reach feature, gate, scoring, return, parameter-selection, or ranking consumers;
- existing exact/mixed/imputed funding restrictions remain intact;
- imputed or otherwise unrankable funding must not activate a signal;
- do not redesign the funding model, exactness taxonomy, or storage layout.

## Required implementation method

### First reproduce the defect

Before changing production code:

1. identify the exact market-data and funding call paths implicated by the archived readiness report;
2. add a focused failing regression test using synthetic metadata and synthetic rows;
3. prove the test fails for the demonstrated reason;
4. record the failure output.

Do not infer a broad loader defect from one code smell. Patch only the call paths reproduced by tests.

### Make the smallest surgical patch

Requirements:

- change the fewest production lines and files that enforce the boundary;
- reuse existing exceptions, metadata structures, reader wrappers, and style;
- no adjacent refactor;
- no new framework;
- no catalog or partition migration;
- no new research features;
- no changes to strategy definitions or economic parameters;
- no documentation/manual/governance cleanup in this task.

Every changed production line must trace directly to a reproduced failing test.

## Required synthetic tests

Use temporary files, synthetic manifests/metadata, synthetic tabular rows, and spies/mocks. Do not read any real protected payload.

At minimum cover the following for both the affected market-data loader and affected funding loader.

### A. Protected file

Synthetic metadata says the file is outcome-bearing and begins on or after `2026-01-01`.

Assert:

```text
reader_spy.call_count == 0
downstream_spy.call_count == 0
explicit_fail_closed_error
```

### B. Mixed rankable/protected file

Synthetic metadata says the file spans rankable and protected periods or has mixed purpose.

Assert:

```text
reader_spy.call_count == 0
downstream_spy.call_count == 0
explicit_fail_closed_error
```

### C. Execution-calibration, prospective, external/unrankable, and unknown-purpose files

Use the existing repository labels or equivalent authority fields. Test every relevant label handled by the affected loader.

Assert zero reader and downstream calls.

### D. Pre-2023 row in an otherwise authorized synthetic file

Assert:

```text
pre_2023_row_never_reaches_downstream
no feature/scoring/return/ranking call for that row
```

Do not assert that the reader must or must not be called.

### E. Non-Kraken row in an otherwise authorized synthetic file

Assert:

```text
non_kraken_row_never_reaches_downstream
no feature/scoring/return/ranking call for that row
```

Do not assert that the reader must or must not be called.

### F. Valid Kraken rankable input

Synthetic Kraken rows wholly inside:

```text
[2023-01-01, 2026-01-01)
```

with an approved rankable purpose must pass to the expected downstream consumer.

Assert the reader and downstream calls occur exactly as the current valid contract expects.

### G. Funding-specific cases

In addition to A-F:

- protected/mixed/unrankable funding file: funding reader spy remains zero;
- pre-2023/non-Kraken funding row: no funding feature/gate/scoring consumer call;
- valid exact Kraken 2023-2025 funding row: permitted under the existing contract;
- imputed or otherwise signal-ineligible funding: cannot activate a signal or ranking path.

Do not test economic returns.

### Spy requirement

Use both:

```text
underlying payload-reader spy
first downstream feature/scoring/return/ranking boundary spy
```

Where several downstream functions exist, spy on the earliest common rankable consumer that proves prohibited rows cannot influence research. Add narrower spies only when needed to demonstrate the defect.

## Scope

### Files allowed to change

Only:

- the directly affected rankable market-data loader or its smallest shared boundary helper;
- the directly affected funding loader or smallest existing shared boundary helper;
- focused unit/regression tests;
- this task's archive and machine-required defect/repair record, if the active repository contract mandates one.

Do not edit any file until the agent has identified the exact call path and listed the proposed files in `PLAN.md`.

### Files and systems excluded

Do not change:

- hypothesis, family, catalyst, universe, or data-capability registries, except a machine-required defect record;
- strategy or family runners beyond the directly affected loader invocation if unavoidable;
- C01, C02, or C03 code;
- package-protocol artifacts;
- lifecycle or PIT-universe code;
- multiplicity or canonical-episode infrastructure;
- data-acquisition code;
- capture code or archives;
- Donch instructions;
- `AGENTS.md`, skills, manuals, runbooks, repository maps, or stale-state prose;
- dependency versions or broad configuration;
- raw data.

Do not access network services, capture roots, private endpoints, or protected payloads.

## Non-goals and deferred work

Explicitly defer:

```text
all external-review package protocol gaps
raw verification extracts
missing reproducibility hashes
test-count reconstruction outside this patch
PIT lifecycle reconstruction
spot/reference acquisition
C01/C02/C03 implementation
effective-trial registry work
canonical cross-family episode work
funding-model redesign
data catalog or partition framework
documentation restructuring
economic research
```

A later separate governance-cleanup task should simplify Donch and backtesting instructions and remove stale repository-state language. Record this only as a deferred note in `COMPLETION.md`; do not edit governance files now.

## Milestones

### 1. Preflight and defect trace

Acceptance:

- exact repository and governance state verified;
- archived readiness report and manifest verified;
- affected loader and funding call paths identified;
- proposed changed files listed;
- no real protected payload opened.

Failure response: stop with a bounded report.

### 2. Failing synthetic regression tests

Acceptance:

- reader and downstream spies reproduce the demonstrated defect;
- failure is attributable to the current boundary logic;
- no economic code path or real data used.

Failure response: do not patch production code if the defect cannot be reproduced.

### 3. Surgical repair

Acceptance:

- minimal boundary patch;
- no new framework or unrelated changes;
- authority distinction in this task is preserved exactly.

Failure response: revert task changes and report why a larger design would be required. Do not expand scope.

### 4. Focused and repository-supported tests

Run:

1. new focused tests;
2. directly related existing loader/funding tests;
3. the smallest repository-supported non-economic regression suite required by the active instructions.

Record exact command, working directory, start/end time, exit code, test count, pass/fail count, and artifacts.

Acceptance:

```text
focused_tests: pass
related_regressions: pass
protected_reader_spies: zero calls
pre_2023_and_non_kraken_downstream_spies: zero calls
valid_rankable_fixture: passes
real_protected_payloads_opened: zero
```

### 5. Independent review

Use a separate review pass on the actual diff and test evidence.

Review:

- scope;
- authority distinction;
- reader/downstream spy validity;
- market-data and funding equivalence;
- no hidden stronger pre-open rule for pre-2023/non-Kraken;
- no protected payload read;
- no economic behavior;
- no unrelated refactor;
- rollback.

Acceptance: `approve`.

### 6. Rerun first-wave readiness

After the patch and review pass, rerun the archived task:

```text
donch_bt_first_wave_readiness_20260716_v1
```

Use the current governance and transferred authority sources.

For the rerun, apply the corrected authority interpretation from this task:

- pre-open exclusion is mandatory for protected/mixed/external-unrankable/unknown-purpose files;
- pre-2023 and non-Kraken rows must be blocked before any rankable downstream use, but pre-open rejection is not a new universal requirement.

The readiness rerun may inspect and report other readiness items, but it must not repair them. Preserve all deferred blockers exactly and recommend only the next task justified by the rerun.

No economic output, candidate returns, or protected payload reads are permitted.

## Required durable records

Use the active repository archive convention. Retain at least:

```text
TASK_SPEC.md
DONCH_ARCHIVE_CONTEXT.md
PLAN.md
DECISIONS_AND_PROGRESS.md
DEFECT_REPRODUCTION.md
CHANGED_FILES.md
COMMANDS_AND_RESULTS.md
VALIDATION.md
READINESS_RERUN_REPORT.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

`DEFECT_REPRODUCTION.md` must show:

```text
affected_call_paths
original_behavior
synthetic_fixture_identity
reader_spy_result
downstream_spy_result
failing_test_before_patch
passing_test_after_patch
authority_rule_enforced
```

The manifest must include path, bytes, SHA-256, purpose, provenance, and authority status.

## Git and rollback

Create an isolated task branch or worktree from:

```text
8cf3e227105fd7626445d27c8caf4c28bccc2ecb
```

Suggested branch:

```text
fix/rankable-loader-boundary-20260716
```

After all tests and independent review pass, one local task-scoped commit is authorized.

Do not push, merge, rebase `main`, or publish the commit.

If any gate fails:

1. restore all production and test files changed by this task;
2. verify the task worktree is clean or preserve the failed branch without merging;
3. retain the task archive and failure evidence;
4. report the smallest exact blocker.

## Acceptance criteria

The task passes only if all are true:

```text
starting_commit_verified: 8cf3e227105fd7626445d27c8caf4c28bccc2ecb
protected_market_file_reader_calls: 0
protected_funding_file_reader_calls: 0
mixed_or_unrankable_file_reader_calls: 0
pre_2023_rows_reaching_rankable_downstream: 0
non_Kraken_rows_reaching_rankable_downstream: 0
valid_Kraken_2023_2025_fixture_passes: yes
funding_signal_ineligible_rows_activate_signal: no
real_protected_payloads_opened: 0
economic_outputs_computed: 0
unrelated_files_changed: 0
focused_and_related_tests: pass
independent_review: approve
readiness_rerun_completed: yes
```

## Final response format

```text
status:
repository_root:
starting_commit:
task_branch_or_worktree:
AGENTS_and_machine_contract_chain:
archived_readiness_report_path_and_sha256:
demonstrated_defect:
affected_market_loader:
affected_funding_loader:
authority_interpretation_applied:
files_changed:
production_lines_changed:
tests_added_or_changed:
failing_test_before_patch:
focused_tests_after_patch:
related_regression_tests:
reader_spy_results:
downstream_spy_results:
real_protected_payloads_opened: no
candidate_returns_computed: no
economic_run_launched: no
capture_accessed: no
data_acquisition_performed: no
readiness_rerun_status:
remaining_readiness_blockers:
local_commit:
push_or_merge_performed: no
task_archive:
artifacts_and_sha256:
registries_changed:
deferred_governance_cleanup_note:
rollback:
next_recommended_task:
human_approval_required:
```
