# Donch-to-Backtesting Task and Archive Template

Status: approved Donch supplemental source  
Date: 2026-07-16  
Revision: 1  
Scope: self-contained task construction, Donch context transfer, agent-owned documentation, and approved Drive archive handoff  
Authority: Donch project instructions revision 3 and the approved minimal-human workflow  
Supersedes: backtesting prompts that depend on prior chat, unstated project sources, or human-maintained technical logs  
Provenance: self-contained task template, backtesting harness, and human workflow decision  
Known limitations: repository paths, commands, task-archive convention, and Drive target must be verified in the backtesting environment

## Use

Donch fills this template for one coherent backtesting outcome. Attach or transfer every required file. The backtesting agent archives the completed specification and owns all routine technical records. The human provides trading sanity checks, broad direction, and approvals—not filing or implementation diagnosis.

```text
TASK TITLE

task_id:
date_utc:
target_environment: backtesting Codex
mode: read_only | ready_to_apply | direct_apply

OBJECTIVE
One observable outcome.

HUMAN ROLE FOR THIS TASK
trading_sanity_check_requested:
broad_direction_decision_requested:
approval_requested:
technical_filing_or_manual_transfer_expected_from_human: none

CURRENT STATE TO VERIFY
repository_root:
branch_commit_remotes:
working_tree:
applicable_AGENTS_chain:
machine_contracts_and_authority_files:
supported_test_lint_format_commands:
existing_task_archive_convention:
configured_Drive_archive_root_and_write_identity_label:

ARCHIVE_CONTEXT FROM DONCH
donch_task_id:
project_decision_ids:
evaluation_and_review_conclusions:
controlling_sources_with_revision_and_sha256:
required_excerpts_or_transferred_files:
tested_or_rejected_translations:
same_sample_actions_forbidden:
unresolved_evidence_gaps:
human_trading_observations_or_questions:
approval_scope:
actions_still_forbidden:

AUTHORITY
1. machine-enforced contracts and finalized run manifests
2. authoritative roots, hashes, ledgers, and audit artifacts
3. current continuity brief
4. current manuals and capability reports
5. current audits and agent reports
6. research sources as priors
7. superseded material as provenance

BOUNDARIES
venue: Kraken only
rankable_interval: 2023-01-01 inclusive to 2026-01-01 exclusive
protected_period: 2026-01-01 onward
July_2026_capture: execution_calibration_only
paid_historical_vendor_data: prohibited
economic_run_authorized: no unless this field names the frozen contract and approval
protected_outcome_access: no
live_trading_orders_private_endpoints: no
capture_restart_or_risk_change: no
Git_push_merge_publication: no unless exact scope is approved
external_writes: exact named targets only

SCOPE
files_allowed_to_inspect:
files_allowed_to_change:
data_allowed_to_read:
data_excluded:
external_services_allowed:

NON_GOALS

FORBIDDEN ACTIONS

ASSUMPTIONS_AND QUESTIONS
Use verified | inferred | proposed | unavailable | blocked.

DELIVERABLES
For each: exact path, format, required fields, authority status.

MILESTONES
For each:
- action
- acceptance criteria
- verification command or inspection
- failure response

TESTS AND INSPECTIONS
Commands must be discovered from the repository.
Separate non-economic tests from any authorized economic run.

BACKTESTING CONTRACT, IF RELEVANT
mechanism:
translation:
event_identity_and_freeze_hash:
control_identity_and_freeze_hash:
historical_universe_authority:
listing_lifecycle_treatment:
rankable_period_guard:
protected_period_audit:
last_mark_index_funding_treatment:
cost_bad_wick_same_bar_boundary_rules:
multiplicity_and_clustered_confidence_gates:
data_config_code_funding_output_provenance:
same_sample_actions_forbidden:
economic_run_contract_hash_and_approval: none unless explicit

AGENT-OWNED TASK ARCHIVE
archive_root_or_proposed_fallback:
TASK_SPEC_and_ARCHIVE_CONTEXT:
PLAN_with_decision_and_progress_logs:
CHANGED_FILES_and_diff_reference:
COMMANDS_TESTS_AND_RESULTS:
VALIDATION_and_protected_period_evidence:
ARTIFACT_MANIFEST:
INDEPENDENT_REVIEW:
COMPLETION_and_NEXT_ACTION:
registries_and_continuity_files_to_update:

DRIVE HANDOFF
authorized: yes | no
exact_remote_and_folder:
non_secret_write_identity_label:
collision_policy:
package_name: qlmg_<specific-scope>_<YYYYMMDD>_vNN.zip
manifest_and_archive_sha256:
remote_content_verification_method:
local_copy_retention:
If setup is incomplete: retain local package and return one bounded setup request.

REVIEW
reviewer_or_fresh_task:
actual_diff_and_artifacts:
domain_invariants:
required_decision: approve | approve_with_notes | changes_required | blocked

RISK AND ROLLBACK
pre_change_snapshot:
risks:
rollback_steps:
rollback_verification:
stop_conditions:

ACCEPTANCE CRITERIA
Numbered and externally observable.

FINAL RESPONSE
status:
objective_result:
files_changed:
commands_and_inspections_run:
tests_and_results:
economic_run_status:
protected_outcome_status:
artifacts_and_sha256:
local_task_archive:
verified_remote_archive_or_blocker:
registries_and_continuity_files_updated:
review_result:
unverified_items:
prohibited_actions_not_performed:
rollback:
human_trading_sanity_checks_requested:
approvals_required:
next_self_contained_task_if_justified:
```

## Donch checks before sending

- The task has one outcome.
- Every controlling source is named and available to the agent.
- `archive_context` contains the needed Donch evaluations and decisions.
- The human is asked only for trading judgment, direction, conflict resolution, or a consequential approval.
- Repository identity, state, commands, and archive convention are discovery items when not verified.
- Economic-run and protected-period status are explicit.
- External writes name the exact destination and collision policy.
- Deliverables, validation, rollback, and final fields are machine-checkable.
- The agent, not the human, owns plans, logs, manifests, registries, packaging, and transfer records.
