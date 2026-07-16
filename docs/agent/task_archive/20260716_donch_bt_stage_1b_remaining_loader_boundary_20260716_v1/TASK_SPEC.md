# Stage 1B — Remaining Rankable Reader Boundary Closure

```text
task_id: donch_bt_stage_1b_remaining_loader_boundary_20260716_v1
repository: /opt/testerdonch
continue_branch: fix/rankable-loader-boundary-20260716
required_ancestor_commit: fc1113d61683e41e7cf9aa76b75b87933a70897c
economic_run_authorized: no
protected_payload_access: no
capture_access: no
data_acquisition: no
```

## Objective

Close the remaining demonstrated rankable-reader boundary bypasses and bind the existing file-authority metadata through `data_paths()`.

Then rerun the first-wave readiness check and report the next justified task.

Do not address unrelated package, lifecycle, multiplicity, C01/C02/C03, documentation, or governance work.

## Start condition

Continue on the existing repair branch/worktree.

Do not require an exact future commit hash. Proceed if:

- the current branch HEAD is `fc1113d61683e41e7cf9aa76b75b87933a70897c` or a clean descendant containing only this loader-repair lineage;
- the worktree is clean;
- local repository identity and applicable `AGENTS.md` chain are verified;
- no unexpected strategy, economic, data, or protected-outcome changes are present.

Record the actual starting commit. Stop only for a material scope or safety mismatch.

## Scope

Patch only the demonstrated remaining paths:

```text
load_symbol_signal_bars()
load_symbol_rank_close_window()
a1_load_symbol_bars_window()
data_paths() authority-metadata binding needed by those readers
```

Reuse the authority helper introduced in commit:

```text
fc1113d61683e41e7cf9aa76b75b87933a70897c
```

Do not create a new catalog, partition system, generalized loader framework, or documentation layer.

## Required behavior

### Protected or unrankable files

When existing path/manifest authority identifies a file as:

```text
2026+ protected
mixed rankable/protected
execution_calibration_only
prospective_only
external_or_unrankable
unknown_or_unprovable
```

the affected reader must fail before opening the payload.

Assert:

```text
payload_reader_calls == 0
rankable_downstream_calls == 0
explicit_fail_closed_error
```

### Pre-2023 and non-Kraken rows

These rows may be opened only where the existing authorized file contract permits it, but they must be removed before any rankable downstream use.

Assert:

```text
feature_or_scoring_calls_for_invalid_rows == 0
return_or_ranking_calls_for_invalid_rows == 0
```

Do not impose a new universal pre-open rule.

### Funding

Do not alter the completed funding repair except where the new reader paths invoke it. Preserve:

```text
protected/mixed/unrankable funding file -> zero payload-reader calls
pre-2023/non-Kraken funding rows -> zero rankable downstream calls
imputed or signal-ineligible funding -> cannot activate a signal
```

## Tests

Use synthetic fixtures and spies only. Do not open real protected payloads.

For each remaining reader path test:

1. protected file;
2. mixed file;
3. unknown/unrankable file;
4. authorized file containing pre-2023 rows;
5. authorized file containing non-Kraken rows;
6. valid Kraken 2023–2025 rows.

Also test that `data_paths()` supplies the expected existing authority metadata to each affected reader.

Run:

- new focused tests;
- all loader-boundary tests;
- directly related regression tests;
- repository guard tests;
- compile/type checks required by repository instructions.

Record exact commands, counts, and exit codes.

## Minimal-change requirement

Before editing, list the exact proposed production and test files.

Every production-line change must trace to a failing synthetic test.

No adjacent refactor. No manual, `AGENTS.md`, registry, package, data-acquisition, strategy, or research-feature changes.

## Readiness rerun

After tests and independent review pass, rerun the archived first-wave readiness task.

For the rerun:

- verify the loader boundary is closed across all known rankable reader paths;
- report external-review package gaps, U2 lifecycle, multiplicity/canonical-episode, C02 spot-data, and C03 PIT-context gaps separately;
- do not treat unrelated legacy package gaps as a reason to repeat this loader task;
- recommend the smallest next outcome-free task for C01 readiness if the loader boundary is closed;
- do not implement that next task.

## Git authorization

If all tests and independent review pass:

1. create one task-scoped local commit on the existing branch;
2. fast-forward `main` to the reviewed branch only if `main` has not changed unexpectedly;
3. push `main` to `origin/main` without force.

This authorization is limited to the verified Stage 1B diff and task archive.

If `main` or `origin/main` changed unexpectedly, stop before integration and report the actual commits. Do not rebase, merge unrelated changes, or force-push.

## Acceptance criteria

```text
all_known_rankable_reader_paths_use_authority_boundary: yes
protected_or_unrankable_reader_calls: 0
invalid_rows_reaching_rankable_downstream: 0
valid_Kraken_2023_2025_fixtures: pass
real_protected_payloads_opened: 0
economic_outputs_computed: 0
focused_and_related_tests: pass
independent_review: approve
readiness_rerun_completed: yes
main_and_origin_main_updated: yes, unless an unexpected remote change blocks integration
```

## Deferred work

Record but do not change:

- external-review package protocol gaps;
- U2 lifecycle authority;
- effective-trial and canonical-episode work;
- C01/C02/C03 implementation;
- spot/reference acquisition;
- Donch/backtesting instruction simplification and stale-state cleanup.

## Final response

```text
status:
actual_starting_commit:
branch:
reader_paths_repaired:
data_paths_authority_binding:
files_changed:
production_lines_changed:
tests_before_and_after:
reader_spy_results:
downstream_spy_results:
real_protected_payloads_opened: no
economic_run_launched: no
readiness_rerun_status:
remaining_blockers_by_category:
next_recommended_task:
local_commit:
main_commit_after_integration:
origin_main_updated:
task_archive:
artifacts_and_sha256:
rollback:
human_approval_required:
```
