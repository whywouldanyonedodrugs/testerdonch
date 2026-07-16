# Backtesting Agent Harness Report

Date: 2026-07-16

Status: `ready_for_remote_agent_reconciliation`

Target repository root: `/opt/testerdonch` — verified read-only in the backtesting VS Code Remote SSH window.

## Outcome

A local, repository-relative backtesting Codex harness was created under:

```text
G:\My Drive\01 Trading\Donch review\DONCH_RESEARCH_REORGANIZATION_20260716\05_backtesting_agent_harness
```

The ready tree can be reconciled in a safe repository worktree, or applied through `BACKTESTING_AGENT_HARNESS_DIFF.patch` after repository-native authority is checked. A read-only VS Code inspection verified `/opt/testerdonch` as the open Git repository on branch `main`, with 239 pending changes and at least 139 staged. The package now includes `docs/agent/DIRTY_REPOSITORY_RECOVERY.md` and an exact remote-agent task specification. They require the backtesting agent to inventory and preserve every staged, unstaged, untracked, conflicted, submodule, large, and excluded-sensitive item; validate a recovery bundle; and create a verified isolated worktree without changing the original checkout. The human is not asked to interpret the diffs or choose routine Git commands.

Nothing in the remote repository was changed, pushed, or merged during the local build. No economic screen ran and the harness build did not inspect protected outcomes. Task-wide disclosure: an earlier generic workbook schema preflight unintentionally surfaced a limited first-row view from outcome-bearing Current Results; values were discarded and not used, period unknown, and no further outcome inspection followed. Transfer of the closed instruction package is recorded separately in `transfer_20260716/TRANSFER_MANIFEST.json`; transfer does not mean repository application.

The harness deliberately creates no `.codex/config.toml` and names no repository test command. Read-only UI inspection did not establish the current commit, remotes, repository instruction chain, or supported commands, so these remain repository-dependent prerequisites. The dirty `main` checkout blocks direct overlay but does not require human code review. The supplied recovery and isolation workflow is the authorized technical route.

## Evidence used

The harness content was built from local authority and metadata. A later read-only target inspection supplied application-state evidence:

1. User master prompt:
   `C:\Users\riman\.codex\attachments\d3693171-c466-47fd-8d05-6d5f875a7e89\pasted-text.txt`
2. Backtesting audit:
   `G:\My Drive\01 Trading\Donch review\backtesterreport.md`
3. Current master brief supplied for this task:
   `G:\My Drive\01 Trading\Donch review\QLMG_Project_Master_Continuity_Brief_2026-07-16_rev7.md`
4. Extracted review-package metadata under:
   `G:\My Drive\01 Trading\Donch review\Review packs\qlmg_external_review_full_20260716_v1\phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1`
5. Read-only access record:
   `G:\My Drive\01 Trading\Donch review\DONCH_RESEARCH_REORGANIZATION_20260716\00_audit\ACCESS_AND_BOUNDARY_REPORT.md`

Metadata inspected was limited to `decision_summary.json`, `package_manifest.csv` summary, `package_size_report.md`, `package_sha256.json`, `redaction_and_secret_scan.md`, `engineering/environment_manifest.json`, `engineering/test_execution_matrix.csv`, `engineering/reproducibility_matrix.csv`, `registry/authoritative_run_registry.csv`, and `registry/root_supersession_map.csv`. Row-level strategy ledgers and protected outcomes were not opened.

The extracted package reports:

- root paths below `/opt/testerdonch`;
- 386 manifest records;
- passing recorded hash validation;
- zero protected-period rows and zero secret findings;
- status `blocked_by_protocol_issue` and release-ready `false`;
- missing raw event-window trade, mark, index/spot, and exact-funding verification extracts;
- blank test and failure counts;
- one missing source snapshot;
- five family lineages with unrecorded reproducibility hashes.

The harness preserves the distinction between integrity of present files and release readiness.

## Design decisions

### Root instructions and documentation

The root `AGENTS.md` is concise and routes detail to seven required documents. It encodes:

- Kraken-only active research;
- the rankable interval `[2023-01-01, 2026-01-01)`;
- the sealed protected period from 2026 onward;
- no paid historical vendor data, live trading, or unauthorized economic screens;
- last/trade, mark, index, and funding roles;
- point-in-time universe and listing controls;
- actual-exit non-overlap and no maximum-hold preblocking;
- frozen candidate and real control identities;
- no summary-row trades, placeholder controls, event sampling, same-bar heroics, artificial closes, or imputed-funding signals;
- immutable historical roots and complete run provenance;
- repository-discovered commands rather than invented commands.

It also routes dirty or uncertain repository state to `DIRTY_REPOSITORY_RECOVERY.md`. That document forbids reset, restore, clean, destructive stash use, deletion, or committing unknown work merely to make the tree clean. It requires an independent recovery bundle, a verified base commit, a separate worktree and branch, overlap analysis, and agent-owned archive records.

`/opt/testerdonch` is now verified as the open repository root. Every deeper repository path or command that was not directly inspected remains marked unverified or `REPO_DEPENDENT_PLACEHOLDER`.

### Skills

Four recurring skills were created:

1. `qlmg-plan-and-preflight`
2. `qlmg-code-change-verification`
3. `qlmg-rankable-backtest-contract`
4. `qlmg-review-package-verification`

The candidate data-acquisition skill was omitted because the current recurring safety and planning requirements are covered by the plan skill plus `DATA_AND_PROTECTED_PERIOD_RULES.md`. Creating a separate skill without repository-specific acquisition entry points would duplicate routing and invite invented commands. Review-package verification and remote handoff were combined because a handoff is the final gated stage of the same recurring package workflow.

No skill resources were added. The workflows need current repository authority, not bundled scripts, reference copies, or assets. Each skill contains only `SKILL.md` and the initializer-created `agents/openai.yaml`.

### Review bundle and Google Drive handoff

The harness requires a closed ZIP named:

```text
qlmg_<specific-content-slug>_<YYYYMMDD>_vNN.zip
```

The UTC date and content slug must describe the bundle. The upload stage requires explicit task authorization, an exact remote and folder, a confirmed non-secret write identity, a collision policy, a verified local ZIP, and a supported remote content check. A readable or configured remote is not authorization. Size equality alone is insufficient. Missing prerequisites produce `remote_handoff_blocked`; the agent must retain the verified local ZIP and must not guess a remote or upload identity.

## Files created

Ready-to-apply repository files:

```text
ready_to_apply/AGENTS.md
ready_to_apply/docs/agent/REPOSITORY_MAP.md
ready_to_apply/docs/agent/DIRTY_REPOSITORY_RECOVERY.md
ready_to_apply/docs/agent/EXECUTION_PLAN_TEMPLATE.md
ready_to_apply/docs/agent/CODE_REVIEW.md
ready_to_apply/docs/agent/RUN_AND_ARTIFACT_CONTRACT.md
ready_to_apply/docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md
ready_to_apply/docs/agent/KNOWN_FAILURE_PATTERNS.md
ready_to_apply/.agents/skills/qlmg-plan-and-preflight/SKILL.md
ready_to_apply/.agents/skills/qlmg-plan-and-preflight/agents/openai.yaml
ready_to_apply/.agents/skills/qlmg-code-change-verification/SKILL.md
ready_to_apply/.agents/skills/qlmg-code-change-verification/agents/openai.yaml
ready_to_apply/.agents/skills/qlmg-rankable-backtest-contract/SKILL.md
ready_to_apply/.agents/skills/qlmg-rankable-backtest-contract/agents/openai.yaml
ready_to_apply/.agents/skills/qlmg-review-package-verification/SKILL.md
ready_to_apply/.agents/skills/qlmg-review-package-verification/agents/openai.yaml
```

Evaluation and delivery files:

```text
ready_to_apply/skill_evals/qlmg-plan-and-preflight.md
ready_to_apply/skill_evals/qlmg-code-change-verification.md
ready_to_apply/skill_evals/qlmg-rankable-backtest-contract.md
ready_to_apply/skill_evals/qlmg-review-package-verification.md
BACKTESTING_AGENT_HARNESS_REPORT.md
BACKTESTING_AGENT_HARNESS_DIFF.patch
BACKTESTING_AGENT_HARNESS_MANIFEST.json
```

The manifest is the authoritative file list with byte sizes and SHA-256 values. It excludes its own hash because an exact self-hash is recursive; the manifest records that limitation explicitly.

## Initialization and verification performed

### Skill creation

The required initializer was run once for each skill:

```text
C:\Users\riman\.codex\skills\.system\skill-creator\scripts\init_skill.py
```

All four initializations succeeded and created `SKILL.md` plus `agents/openai.yaml`. No optional resource directories were requested.

### Skill validation

The required validator was run for every skill:

```text
python C:\Users\riman\.codex\skills\.system\skill-creator\scripts\quick_validate.py <skill-directory>
```

Results:

```text
qlmg-code-change-verification: Skill is valid!
qlmg-plan-and-preflight: Skill is valid!
qlmg-rankable-backtest-contract: Skill is valid!
qlmg-review-package-verification: Skill is valid!
```

### Additional checks

Successful local checks:

```text
STRUCTURE_OK ready_files=20 root_and_agent_docs=8 skills=4 evals=4
FRONTMATTER_OK skills=4
OPENAI_YAML_TOKEN_OK skills=4
EVAL_COUNTS_OK each_positive>=3 each_negative>=3
FINAL_LINKS_OK markdown_files=16
TODO_SCAN_OK
YAML_PARSE_OK skills=4 frontmatter_keys=name,description openai_interface_constraints=pass
POLICY_ASSERTIONS_OK required_terms=18
MINIMAL_SKILLS_OK skills=4 optional_resource_dirs=0
CONFIG_SAFETY_OK config_toml_created=0
ROOT_LABEL_OK opt_testerdonch_mentions=4
PATCH_APPLY_CHECK_OK files=20 core.autocrlf=false
PATCH_ROUNDTRIP_HASH_OK files=20
MANIFEST_JSON_OK entries=22 evidence_sources=12
MANIFEST_FILE_HASHES_OK entries=22
MANIFEST_SOURCE_HASHES_OK entries=12
MANIFEST_SELF_SIZE_OK self_hash_status=not_embedded
```

The patch was applied in a new temporary directory and all 20 applied files matched the ready tree by SHA-256. `core.autocrlf=false` was used for the byte-for-byte test because the package is normalized to UTF-8 with LF line endings.

Diagnostic check failures were corrected and rerun: an early evaluation-count script passed a file object incorrectly, a root-label assertion used too narrow a context window, and the first patch hash comparison allowed Git to convert LF to CRLF. These were check-harness issues, not unresolved artifact defects.

## Verification not performed

- No fresh task was run inside the remote backtesting repository.
- Read-only VS Code inspection verified repository root `/opt/testerdonch`, branch `main`, and 239 pending changes, including at least 139 staged. Quick Open found no exact `AGENTS.md` and no exact proposed `DATA_AND_PROTECTED_PERIOD_RULES.md` collision, but this is not a complete filesystem search or instruction-chain audit.
- The current commit, remotes, full staged/unstaged file list, repository authority revisions, and supported commands were not verified.
- No repository unit, integration, lint, format, or documentation command was run because none was verified from the repository.
- No economic, backtest, validation, or protected-period run was performed.
- Repository-side `rclone` access and the configured write identity remain for the backtesting agent to verify. The exact approved transfer folder is `DONCH_AGENT_TRANSFER_20260716`, ID `1eWdmin0E7QKYJ5AvFcpRmvvDG3YiWlBz`; the local package transfer is recorded by its sidecar manifest.
- No patch was applied to `/opt/testerdonch`.

## Application prerequisites

Before applying the package, the authorized repository agent must:

1. recheck `/opt/testerdonch`, branch `main`, and the working tree immediately before application because the observed 239-change state may have moved;
2. inventory staged, unstaged, untracked, conflicted, submodule, large, and excluded-sensitive state and validate an independent recovery bundle;
3. compare these files with existing `AGENTS.md`, `docs/agent`, and `.agents/skills` content;
4. verify current machine contracts and any continuity brief later than rev7;
5. discover supported repository commands and replace no placeholder with an unverified command;
6. create an isolated branch/worktree from a verified commit, record overlap, and leave the dirty `main` checkout unchanged;
7. review the patch and run repository-supported documentation and skill checks;
8. verify the configured non-secret Drive identity against the exact approved folder and use unique names without overwrite.

## Final status

```text
remote_repository_accessed: true_read_only_via_vscode
remote_repository_modified: false
target_root_verified: true_/opt/testerdonch
observed_branch: main
observed_pending_changes: 239
observed_staged_changes: at_least_139
direct_application_blocker: dirty_main_checkout_and_unverified_commands
authorized_resolution: agent_owned_recovery_bundle_and_verified_isolated_worktree
ready_to_apply_tree: complete
ready_to_apply_patch: generated_and_local_roundtrip_verified
skills_initialized: 4
skills_validated: 4
skill_evals: 4_with_3_positive_and_3_negative_each
repository_commands_invented: false
economic_runs_launched: false
protected_outcomes_inspected_by_harness_build: false
task_wide_schema_preflight_exception: limited unintended first-row view from outcome-bearing Current Results; values discarded and not used; period unknown; no further outcome inspection
google_drive_transfer_destination: DONCH_AGENT_TRANSFER_20260716_folder_id_1eWdmin0E7QKYJ5AvFcpRmvvDG3YiWlBz
git_pushes_or_merges: false
human_technical_review_required: false
human_role: trading_sanity_checks_broad_direction_authority_conflicts_and_consequential_approvals
remaining_approval_boundaries: economic_runs_protected_outcomes_live_or_capture_actions_push_merge_remote_overwrite_or_destructive_operations
```
