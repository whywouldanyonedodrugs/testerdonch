# Backtesting Task: Capital.com Expansion Read-Only Repository Preflight

Status: proposed task; ready to send; no repository or economic authorization implied  
Date: 2026-07-18  
Revision: 1  

## Task identity

```text
task_id: DONCH_CAPITALCOM_EXPANSION_PREFLIGHT_20260718_R1
target_environment: backtesting Codex
mode: read_only
drive_handoff: approved_default
```

## Objective

Produce a repository-grounded, minimal plan for expanding the historical research infrastructure from Kraken perpetuals to:

1. Kraken perpetuals;
2. Capital.com instruments actually acquired by the separate downloader; and
3. explicit directed cross-platform research.

This task must determine the smallest safe repository and documentation changes. It must not implement them.

## Human role for this task

```text
trading_sanity_check_requested: none during preflight
broad_direction_decision_requested: none unless a material architecture conflict is found
approval_requested: none for read-only work
technical_filing_or_manual_transfer_expected_from_human: none
```

## Current state to verify

```text
repository_root: discover and verify; prior read-only observation was /opt/testerdonch
branch_commit_remotes: verify
working_tree: verify; prior observation reported main with 239 pending changes and at least 139 staged
applicable_AGENTS_chain: discover
machine_contracts_and_authority_files: discover and rank
supported_test_lint_format_commands: discover from repository evidence
existing_task_archive_convention: discover
configured_Drive_archive_root_and_write_identity_label: read docs/agent/DRIVE_HANDOFF_TARGET.md if present
```

Do not assume the prior repository observation is still current.

## Archive context from Donch

```text
donch_task_id: DONCH_CAPITALCOM_EXPANSION_PREFLIGHT_20260718_R1
project_decision_ids:
  - CURRENT_KRAKEN_SCOPE_20260716
  - PROPOSED_CAPITALCOM_EXPANSION_DIRECTION_20260718
  - GLOBAL_PROTECTED_PERIOD_PRESERVATION_PROPOSAL_20260718
evaluation_and_review_conclusions:
  - Current applied Donch authority remains Kraken-only until exact replacement sources/settings are approved and applied.
  - The desired architecture is one shared research/evidence core, source-specific Kraken and Capital.com adapters, and a small relationship layer for cross-platform work.
  - Capital.com is an OTC CFD platform and must not be modeled as mechanically equivalent to Kraken.
  - Raw and normalized data should remain source-separated.
  - Existing Kraken family decisions must be preserved; Capital.com translations require new identities.
  - No broad all-instrument economic scan should be proposed as the first task.
required_excerpts_or_transferred_files: all files listed in the attached manifest below
tested_or_rejected_translations:
  - Use the current research decisions and 214-row hypothesis registry.
  - Do not reopen closed Kraken translations by renaming or threshold search.
same_sample_actions_forbidden:
  - no economic screen
  - no protected outcome access
  - no threshold, symbol, asset-class, or horizon selection from outcomes
  - no re-ranking of current families
unresolved_evidence_gaps:
  - actual Capital.com data manifest and schema are not yet available in this package
  - historical Capital.com financing, corporate actions, lifecycle, and trading-rule history are unknown
  - current repository structure and safe worktree state must be verified
human_trading_observations_or_questions: none
approval_scope: read-only repository inspection and local/Drive task archive only
actions_still_forbidden:
  - repository file changes
  - economic runs
  - protected outcome inspection
  - orders or account mutation
  - Git commit, push, merge, PR, or publication
  - external writes other than the approved default Drive handoff
```

## Authority

Use this order:

1. machine-enforced contracts and finalized run manifests;
2. authoritative roots, hashes, ledgers, and audit artifacts;
3. current continuity brief;
4. current manuals and capability reports;
5. current audits and agent reports;
6. research sources as priors;
7. superseded material as provenance.

The attached proposed expansion plan is a planning input, not applied authority.

## Boundaries

```text
current_applied_research_venue: Kraken only until replacement approval/application
proposed_future_research_platforms: Kraken perpetuals; Capital.com acquired instruments
rankable_interval_proposal: 2023-01-01 inclusive to 2026-01-01 exclusive
protected_period: 2026-01-01 onward
Kraken_July_2026_capture: execution_calibration_only
Capital.com_2026_plus_default: data_engineering_only
paid_historical_vendor_data: prohibited
economic_run_authorized: no
protected_outcome_access: no
live_trading_orders_private_endpoints: no
Capital.com_order_position_account_mutation_endpoints: no
capture_restart_or_risk_change: no
Git_push_merge_publication: no
external_writes: approved default Drive task archive only
```

## Scope

### Files allowed to inspect

- repository instructions and authority files;
- source code, configuration, schemas, tests, registries, documentation, manifests, and task-archive conventions;
- directory/file metadata for data roots;
- small schema-only or metadata-only samples when needed and when protected outcome values are not exposed.

### Data allowed to read

- repository metadata and manifests;
- 2023-2025 schema/field definitions as needed;
- no economic outcome tables;
- no 2026+ strategy outcome rows;
- no protected review-package outcomes.

### Files allowed to change

None in the repository. The agent may create its own task archive outside decision-bearing source paths according to the verified archive convention.

### External services allowed

- configured Google Drive handoff only, using `drive_handoff: approved_default` and `docs/agent/DRIVE_HANDOFF_TARGET.md`;
- no GitHub or other external write.

## Non-goals

- no code implementation;
- no data import;
- no backtest or screen;
- no hypothesis prioritization from outcomes;
- no final wording for replacement Donch project instructions;
- no Capital.com downloader changes;
- no attempt to build a universal execution simulator.

## Required inspections

1. Verify repository identity, Git state, instructions, machine contracts, commands, archive convention, and Drive handoff configuration.
2. Map current Kraken data loaders, manifests, period guards, instrument identity, universe/lifecycle logic, event ledger, controls, run identities, and reporting.
3. Determine whether current code already has a source/venue abstraction and whether extending it is simpler than adding one.
4. Identify the minimal compatible representation for Capital.com bid/ask OHLC, instrument epics/types, calendars, expiries, and metadata.
5. Identify where platform/source identity must be added to prevent accidental mixing.
6. Identify the smallest protected-period guard that can apply before load to both sources.
7. Identify exact documentation/registry files that would need revision inside the repository.
8. Determine whether a small instrument map and cross-platform directed-contract layer can be additive without refactoring existing Kraken logic.
9. Identify tests required to prove existing Kraken behavior is unchanged.
10. Identify stop conditions and rollback for a later implementation task.

## Deliverables

Use the repository's verified task-archive convention. If none exists, propose but do not apply a stable convention.

Required files:

```text
READ_FIRST.md
TASK_SPEC.md
DONCH_ARCHIVE_CONTEXT.md
REPOSITORY_PREFLIGHT.md
CURRENT_ARCHITECTURE_MAP.md
MINIMAL_EXPANSION_ARCHITECTURE.md
DOCUMENTATION_AND_REGISTRY_IMPACT.md
PROPOSED_IMPLEMENTATION_TASK_SEQUENCE.md
TEST_AND_ROLLBACK_PLAN.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

### `MINIMAL_EXPANSION_ARCHITECTURE.md` must answer

- What can stay unchanged?
- What must become platform-aware?
- What should remain source-specific?
- Is a shared core plus two adapters feasible without a major refactor?
- What is the smallest instrument identity/mapping layer?
- Where should Capital.com data eventually live?
- How will cross-platform source/target time and execution semantics be represented?
- How will protected rows fail before load?
- What repository files would a later patch touch?

### `DOCUMENTATION_AND_REGISTRY_IMPACT.md` must classify each affected item

```text
must_change_before_import
must_change_before_economic_run
can_remain_Kraken_specific
new_file_required
not_applicable
```

### `PROPOSED_IMPLEMENTATION_TASK_SEQUENCE.md`

Use one observable outcome per future task. Keep the sequence minimal. Each task must include acceptance, verification, failure response, rollback, and Drive handoff.

## Tests and inspections

Discover commands from repository evidence. Because this task is read-only, do not run commands that write caches, mutate state, download data, or execute economics. Safe read-only tests/inspections may run only if their side effects are understood.

Record every command, exit code, and material result. If a command is unavailable or unsafe, report it as such.

## Acceptance criteria

1. Repository root, applicable instructions, branch, commit, remotes, and working-tree state are verified.
2. The actual current architecture and period guard are mapped from code, not inferred from Donch prose.
3. The report identifies a minimal expansion path and explicitly rejects unnecessary abstractions.
4. No repository decision-bearing file is changed.
5. No economic metric, candidate return, protected outcome, or strategy ranking is viewed or generated.
6. Existing Kraken decisions and data semantics are preserved in the proposal.
7. Capital.com is treated as OTC CFD data with bid/ask and platform-specific costs/rules.
8. Cross-platform research is represented as directed source-to-target contracts.
9. A later implementation patch has a bounded file scope, tests, rollback, and stop conditions.
10. The complete local task archive is closed, manifested, secret-scanned, uploaded to the approved default Drive root, and verified by content hash; if setup is unavailable, retain the local package and report one bounded blocker.

## Drive handoff

```text
authorized: yes
mode: approved_default
configuration_authority: docs/agent/DRIVE_HANDOFF_TARGET.md
approved_archive_root: DONCH_BACKTESTING_HANDOFFS
approved_archive_root_url: https://drive.google.com/drive/folders/1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl
collision_policy: do not overwrite; create a new dated descriptive task folder/package
package_name: qlmg_capitalcom_expansion_preflight_20260718_v01.zip
manifest_and_archive_sha256: required
remote_content_verification_method: repository-supported hash or round-trip verification
local_copy_retention: required
```

Never expose credentials or include raw protected outcome payloads.

## Risk and rollback

```text
pre_change_snapshot: Git identity/status and task start record
risks:
  - accidental protected outcome exposure
  - dirty-tree confusion
  - commands with hidden write side effects
  - over-generalising the current architecture
rollback_steps:
  - no repository changes are permitted
  - remove only agent-created temporary files outside the durable task archive if safe
rollback_verification:
  - compare Git status before and after; no new repository changes attributable to the task
stop_conditions:
  - protected rows would need to be opened
  - repository identity or instruction chain is ambiguous
  - secret material is encountered
  - a read-only inspection would mutate or run economics
  - Drive target or write identity is not verified
```

## Final response format

```text
status:
objective_result:
repository_identity_and_state:
files_changed:
commands_and_inspections_run:
tests_and_results:
economic_run_status:
protected_outcome_status:
minimal_architecture_decision:
proposed_later_patch_scope:
artifacts_and_sha256:
local_task_archive:
verified_remote_archive_or_blocker:
registries_and_continuity_files_proposed_for_update:
review_result:
unverified_items:
prohibited_actions_not_performed:
rollback:
approvals_required:
next_self_contained_task_if_justified:
```

## Attached source manifest

- `DONCH_CAPITALCOM_EXPANSION_PLAN_2026-07-18_rev1.md` — 23975 bytes — SHA-256 `ff6cd94a18567f2f32207eb215bf16aafa472cacf6d3f672b26ca8f6da1f1d70`
- `00_READ_FIRST_Project_Source_Map.md` — 5675 bytes — SHA-256 `88e1dbfca157bead77d6f15daa002687edd929bfd5d88fa13576c9dca9d09daf`
- `01_AUTHORITY_QLMG_Operating_Contract_2026-07-16.md` — 8168 bytes — SHA-256 `cc7556ae37ba17391e64b8fc9026e118b40c575597967b78bf2e39e8dfe3f379`
- `02_STATE_Master_Continuity_Brief_2026-07-16_rev8.md` — 9739 bytes — SHA-256 `eff5cfce197e50ba19830545747adf9c423c6f57c6b4ce2afd80614a38613cda`
- `03_STATE_Current_Research_Decisions_2026-07-16.md` — 9588 bytes — SHA-256 `e8482c77a2974914dbc1b3ad30a5972c97273f7d61f5ffc16308a555a97ac0b7`
- `04_REGISTRY_Hypothesis_and_Family_Status_2026-07-16.csv` — 347739 bytes — SHA-256 `f5cebfea208905cfd8714389568a15329e5165ae1326f06fbd5610780e318b53`
- `05_REGISTRY_Kraken_Data_and_Evidence_Capability_2026-07-16.csv` — 11300 bytes — SHA-256 `d4a197e31c048542eae7d5d03b4964e5866da2d8bdd4250198de580692c54244`
- `07_AUDIT_Backtester_and_Evidence_Readiness_2026-07-16.md` — 8915 bytes — SHA-256 `fb4c430f4b8bf480709a789036a7d8d04a7bf58088ef1acaecfd97d86ad4b6f1`
- `12_MANUAL_Test_and_Evidence_Standards_2026-07-16.md` — 10638 bytes — SHA-256 `f1fe267aaf416ec219ec09130a8c52ac11acaafce32c2d991f4c8f2157fa1544`
- `13_GUIDE_Backtest_Claims_and_Review_2026-07-16.md` — 6156 bytes — SHA-256 `ea46a7b09369e2e5f1f2e8c5f9a09a8e57fa8bd66b55abc88c415c69b17bdab3`
- `15_RUNBOOK_Human_Approval_and_Orchestration_2026-07-16(1).md` — 10431 bytes — SHA-256 `d2eec99ffd0e71fc2aca70cfabef4eeb2fbc5c2a259d53742763859ba7db05ce`
- `16_INDEX_Provenance_and_Supersession_Map_2026-07-16.md` — 11981 bytes — SHA-256 `21fa2cdf5977c36b4c99edad1b9694972b23e8b15dc55d0f3772be9e70839be9`
- `99_Karpathy.md` — 2518 bytes — SHA-256 `6e22cc54cb02a5e98ae42d06d9d7292db0c1b43894831b32879beb0166b2aea7`
- `DONCH_PROJECT_INSTRUCTIONS_FULL.md` — 32818 bytes — SHA-256 `ad1a2ca0fe1919f47e9c8aa5d32a1fb719e199a0770a918b3b23e7a6f41ce15d`
- `BACKTESTING_AGENT_OPERATING_AND_ARCHIVE_INSTRUCTIONS.md` — 8992 bytes — SHA-256 `75765e8d96ecdf3905f5caafff7437c81bf53266939493ddcfd92d9f86e6ca57`
- `DONCH_TO_BACKTESTING_TASK_ARCHIVE_TEMPLATE.md` — 6016 bytes — SHA-256 `a8a7d05346e1eb314404c3f3af9e86e90c608cb0e3c383819c9298e4040facbd`
