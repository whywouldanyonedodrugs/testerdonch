# Backtesting Task: First-Wave Research Readiness Verification

```text
task_id: donch_bt_first_wave_readiness_20260716_v1
date_utc: 2026-07-16
target_environment: backtesting Codex
mode: direct_apply
economic_run_authorized: no
protected_outcome_access: no
```

## Objective

Perform an outcome-free, repository-grounded readiness verification for the approved first-wave Kraken research program.

Produce a verified go/no-go decision for the next data-and-eligibility task. Do not implement a hypothesis generator, acquire new data, calculate strategy returns, change research code, or launch an economic run.

You may create only the task archive and its reports/manifests under the repository's verified task-archive convention. Do not change strategy code, data pipelines, machine contracts, current registries, manuals, or existing run artifacts.

## Human-approved research direction

The first-wave order is:

1. `C01_debetaed_residual_shock_path_bifurcation`
2. `C02_spot_led_vs_perp_led_impulse`
3. `C03_PIT_cohort_breadth_and_dispersion_context`

Approved planning choices:

```text
initial_universe_route:
    U2 — begin C01 on a narrow, independently verified continuously eligible
    anchor cohort while broader PIT lifecycle reconstruction proceeds

C01_directional_prior:
    smooth positive residuals primarily test continuation
    jump-dominated positive shocks require completed failure before shorting
    preserve all four sign/path branches as registered diagnostics
    never pool branches to rescue expectancy

initial_horizon:
    intraday, approximately 1–24 hours

execution_boundary_for_later_mechanism_proof:
    conservative all-taker
    next executable bar
    trade and mark paths kept separate
    frozen slippage scenarios
    no passive-fill claim
    no multi-lot, partial-exit, or add requirement in the first proof

C14_funding_boundary:
    preserved as a later alternate
    not part of the first wave
```

Minimum signal frequency, maximum concurrent positions, and other business-usefulness thresholds are intentionally not fixed in this task. Do not invent them.

## Archive context from Donch

```text
donch_task_id:
    DONCH-NEXT-RESEARCH-20260716-STAGE1

project_decision_ids:
    FIRST-WAVE-C01-C02-C03
    UNIVERSE-U2
    C01-DIRECTIONAL-PRIOR-B
    HORIZON-INTRADAY-FIRST
    C14-DEFERRED-ALTERNATE

evaluation_and_review_conclusions:
    - No current strategy is validation-grade or live-ready.
    - Completed failure is a reusable component, not permission to retune
      RFBS, Backside, LFBS, or failed-breakdown families.
    - Mandatory retest-only entry, session-only directional searches,
      residual-laggard buying, hard prior-high/relative-strength gates,
      broad LFBS fanout, and chart-only liquidation translations remain
      closed in their tested forms.
    - Historical Kraken trade/last and mark bars are the currently established
      2023–2025 foundation for the acquired roster.
    - Historical spot/index, OI, basis, typed forced flow, CVD, spread, depth,
      and observed slippage must be treated as unavailable unless current
      repository evidence proves acquisition.
    - Exact historical funding is materially incomplete before 2025-06-26.
    - The broad historical universe is not yet proven survivorship-free.
    - July 2026 capture is execution_calibration_only.

controlling_sources:
    - QLMG_Next_Research_Plan_2026-07-16_v1.md
      revision: 1
      sha256: 00065c0770984b716c42dc4a0affa3e87b8051ca6aac100e6ee270d0f876f91a
    - 01_AUTHORITY_QLMG_Operating_Contract_2026-07-16.md
      revision: 1.0
    - 02_STATE_Master_Continuity_Brief_2026-07-16_rev8.md
      revision: rev8
    - 03_STATE_Current_Research_Decisions_2026-07-16.md
      revision: 1.0
    - 04_REGISTRY_Hypothesis_and_Family_Status_2026-07-16.csv
    - 05_REGISTRY_Kraken_Data_and_Evidence_Capability_2026-07-16.csv
    - 12_MANUAL_Test_and_Evidence_Standards_2026-07-16.md
      revision: 1.0
    - 13_GUIDE_Backtest_Claims_and_Review_2026-07-16.md
      revision: 1.0
    - BACKTESTING_AGENT_OPERATING_AND_ARCHIVE_INSTRUCTIONS.md
      revision: 1
    - DONCH_TO_BACKTESTING_TASK_ARCHIVE_TEMPLATE.md
      revision: 1

source_identity_rule:
    Compute and record the exact local SHA-256 of every controlling source
    actually used. If a source is absent or differs from the stated revision,
    stop or report the bounded conflict. Do not ask the human to reconstruct it
    from memory.

same_sample_actions_forbidden:
    - no RFBS, Backside, LFBS, failed-breakdown, A1, H43, prior-high,
      relative-strength, retest, session, or C2 threshold rescue
    - no renamed nearby pattern on the same 2023–2025 tape
    - no outcome-driven residual model, path measure, cohort, window, or cutoff
    - no protected-period observation used for ranking or rule selection

unresolved_evidence_gaps_to_verify:
    - current external-review package protocol/release status
    - effective-trial registry support
    - canonical cross-family episode identity support
    - protected-data fail-closed behavior
    - exact status of historical Kraken spot/reference data
    - defensibility of a continuously eligible U2 anchor cohort
    - readiness of causal residual/path features and PIT cohort context

approval_scope:
    read-only repository/data inspection, non-economic tests, and creation of
    this task's archive/report files only

actions_still_forbidden:
    economic screening; return computation for candidate ranking; protected
    data reads; new public-data acquisition; code/config/manual/registry changes;
    capture access; private endpoints; orders; Git commit/push/merge/publication;
    destructive or unapproved external writes
```

## Authority

Use this order:

1. machine-enforced contracts and finalized run manifests;
2. authoritative roots, hashes, ledgers, and audit artifacts;
3. current operating contract and continuity brief;
4. current manuals and capability registries;
5. current audits and agent reports;
6. research sources as priors;
7. superseded material as provenance.

Machine evidence outranks this prompt when the two conflict. Record the conflict and stop if the hierarchy does not resolve it.

## Binding boundaries

```text
venue: Kraken only
rankable_interval: [2023-01-01, 2026-01-01)
protected_period: 2026-01-01 onward
July_2026_capture: execution_calibration_only
paid_historical_vendor_data: prohibited
economic_run_authorized: no
protected_outcome_access: no
capture_access: no
live_trading_orders_private_endpoints: no
Git_commit_push_merge_publication: no
```

Do not open 2026 payloads to “check” the firewall. Verify through code, tests, manifests, date guards, fixtures, or non-outcome metadata only.

## Start-of-task verification

Before any other work, verify and record:

```text
repository_root
applicable_AGENTS.md_chain
branch
commit
remotes
working_tree_state
machine_contracts_and_authority_files
supported_test_lint_format_commands
existing_task_archive_convention
current_plan_or_task_records
current_registry_paths
```

The operator reports that the repository is synchronized and clean. Treat this as a claim to verify, not as machine evidence.

If the tree is dirty, the instruction chain is ambiguous, the repository root is uncertain, or user work could be overwritten, stop before writing anything except a bounded blocker report in a safe location.

## Scope of inspection

Inspect only what is needed to verify readiness:

- repository instructions, contracts, tests, and supported commands;
- current hypothesis, data-capability, attempt/multiplicity, run, defect,
  supersession, and continuity registries;
- external-review package manifests, status records, verification notes, and
  source/test matrices;
- historical-data manifests, schemas, coverage records, and symbol/lifecycle
  authorities;
- code and tests implementing venue/date/protected-data guards;
- code and tests for candidate identity, canonical episode identity, causal
  feature construction, and control registration;
- existing Kraken spot/reference datasets or acquisition records, if present.

Do not inspect candidate returns, strategy summaries for selection, protected data payloads, or 2026 outcome-bearing artifacts.

## Required verification questions

### A. Repository and archive readiness

1. Is the repository clean and safely writable for a later task?
2. Which exact `AGENTS.md` files and machine contracts govern research work?
3. What task-archive convention and supported commands are authoritative?
4. Is routine Drive handoff already configured and approved? Record it, but do
   not upload in this task unless standing machine instructions explicitly
   require the task archive upload and identify the exact target and collision
   policy.

### B. Package and evidence status

5. Does the external-review package still report
   `blocked_by_protocol_issue` / `release_ready=false`?
6. If the status changed, identify the exact higher-authority artifact, diff,
   tests, and hashes that changed it.
7. Are raw verification extracts, test/failure counts, missing source snapshot,
   reproducibility hashes, and TSMOM path coverage now resolved, explicitly
   capped, or still open?
8. Does any current narrative overstate package verification?

Do not repair package gaps in this task.

### C. Protected-data firewall

9. Verify that rankable loaders fail before reading:
   - pre-2023 rows;
   - non-Kraken rows;
   - 2026+ protected rows;
   - execution-calibration files.
10. Record exact tests, fixtures, commands, exit codes, and evidence.
11. Verify that funding is partitioned before protected rows enter rankable
    processing and that mixed/imputed funding cannot activate a signal.

Stop if fail-closed behavior cannot be demonstrated without opening protected outcomes.

### D. Research-degrees-of-freedom and episode identity

12. Is there a current effective-trial or attempt registry that can record every
    C01 definition, dimension, bound, killed candidate, and follow-up?
13. Is one canonical cross-family episode ID available and tested?
14. Can exact and nearby episode overlap be distinguished?
15. Can C01 be registered as a materially new multiplicity family while linking
    its nearest prior families and same-sample prohibitions?

### E. U2 universe readiness

16. Can the repository prove a narrow continuously eligible anchor cohort
    throughout the intended C01 interval using official Kraken evidence?
17. List candidate symbols only as an eligibility inventory; do not inspect or
    compare their returns.
18. For each candidate symbol record:
    - official identity;
    - verified eligibility start;
    - verified or conservatively bounded end/status;
    - data coverage;
    - known lifecycle uncertainty;
    - inclusion/exclusion reason.
19. State the maximum claim permitted under U2.
20. If U2 cannot be proven, identify the smallest outcome-free lifecycle task
    required.

Do not present bar existence or a current active roster as proof of continuous
eligibility.

### F. C01 readiness

21. Classify each item as `ready`, `partial`, `missing`, or `blocked`:

```text
causal BTC residual feature
causal BTC+ETH residual feature
fixed rolling estimation window support
path-efficiency measure
jump/largest-bar contribution measure
parent-market state
realized-volatility state
causal feature hashing
candidate identity freeze
canonical episode identity
control-generation harness
attempt/multiplicity registration
year/symbol/sign/path candidate-count reporting without returns
trade/mark separation
next-executable-bar all-taker simulation support
```

22. Identify the closest prior family/code path for each component and the risk
    of accidental same-sample reuse.
23. State the smallest generator-only implementation task that could follow,
    without writing that implementation now.

### G. C02 readiness

24. Verify whether official public Kraken spot/reference history for 2023–2025
    is already acquired.
25. If present, report source, path, hashes, resolution, symbols, coverage,
    timestamp semantics, gaps, and PIT status.
26. If absent, define the smallest outcome-free acquisition scope needed for
    BTC, ETH, and any U2 alt cohort. Do not make network requests in this task.
27. Identify existing synchronization and symbol-identity infrastructure.

### H. C03 readiness

28. Verify current PIT lifecycle, cohort, breadth, and dispersion support.
29. Distinguish:
    - verified PIT membership;
    - current-roster/bar-existence proxy;
    - dated stable sector;
    - current-only or retrospective narrative tags.
30. State whether C03 can later be applied as a continuous context layer to a
    frozen C01/C02 tape without cutoff searching.

### I. Recommended next task

31. Recommend exactly one next self-contained task:
    - `Stage_2A_U2_and_common_feature_foundation`, or
    - a narrower prerequisite if a blocker requires it first.
32. Give the proposed task's objective, allowed files/data, deliverables, tests,
    and stop conditions.
33. Do not create its implementation prompt or begin its work.

## Deliverables

Use the repository's verified task-archive convention. If none exists, stop and
propose a location rather than inventing one.

Required files:

```text
TASK_SPEC.md
DONCH_ARCHIVE_CONTEXT.md
PLAN.md
DECISIONS_AND_PROGRESS.md
COMMANDS_AND_RESULTS.md
FIRST_WAVE_READINESS_REPORT.md
FIRST_WAVE_DEPENDENCY_MATRIX.csv
PROTECTED_BOUNDARY_VERIFICATION.md
PACKAGE_PROTOCOL_STATUS.md
NEXT_TASK_RECOMMENDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
```

`FIRST_WAVE_DEPENDENCY_MATRIX.csv` must include:

```text
capability_id
mechanism
requirement
status
verified_path_or_artifact
test_or_inspection
authority_level
protected_data_risk
same_sample_reuse_risk
blocking
smallest_remedy
permitted_next_action
```

Do not update substantive project registries in this task. Instead, list any
proposed registry corrections in the readiness report with exact evidence.

## Milestones

### 1. Preflight and archive setup

**Acceptance:** repository identity, instructions, state, commands, and archive
convention verified.

**Failure response:** stop with one bounded blocker report.

### 2. Authority and package verification

**Acceptance:** current package status and every open/resolved protocol item
traced to exact artifacts and hashes.

**Failure response:** retain `blocked` or `unavailable`; do not infer closure.

### 3. Protected-boundary verification

**Acceptance:** fail-closed behavior demonstrated through supported tests or
non-outcome fixtures; no protected payload opened.

**Failure response:** stop; classify new economic work as blocked.

### 4. First-wave dependency audit

**Acceptance:** U2, C01, C02, and C03 capabilities classified with exact paths,
tests, gaps, and smallest remedies.

**Failure response:** do not substitute proxies for missing data.

### 5. Independent review and closure

**Acceptance:** a separate review pass checks authority, protected boundary,
same-sample prohibitions, evidence wording, artifacts, and manifest.

**Failure response:** mark `changes_required` or `blocked`; do not recommend
Stage 2 as ready.

## Tests and inspections

Discover commands from repository evidence. Do not invent commands.

Record:

```text
command
working_directory
purpose
start/end time
exit_code
tests_run
passes
failures
artifact_paths
```

Permitted tests are non-economic:

- instruction and configuration checks;
- unit/fixture tests for venue/date/protected guards;
- schema and manifest validation;
- deterministic identity/feature tests;
- read-only registry/package consistency checks.

Prohibited:

- candidate-return generation;
- PnL, expectancy, MAE/MFE, or post-entry outcome comparison;
- parameter sweep;
- economic control evaluation;
- protected-period content reads.

## Risk and rollback

Only new task-archive files may be created.

Before writing, record the clean state and planned paths. If any file outside the
task archive changes, stop, restore only changes made by this task, verify the
restore, and report the incident. Preserve all pre-existing work.

## Final response format

```text
status:
repository_root:
branch_commit:
working_tree_verified:
AGENTS_and_contract_chain:
package_status:
package_release_ready:
open_protocol_items:
protected_firewall_status:
protected_payloads_opened: no
effective_trial_registry:
canonical_episode_identity:
U2_anchor_cohort_readiness:
C01_readiness:
C02_spot_reference_status:
C03_PIT_context_readiness:
recommended_next_task:
files_changed:
commands_and_inspections:
tests_and_results:
artifacts_and_sha256:
local_task_archive:
verified_remote_archive_or_blocker:
registries_changed: no
economic_run_launched: no
candidate_returns_computed: no
protected_outcomes_inspected: no
capture_accessed: no
Git_commit_push_merge: no
unverified_items:
prohibited_actions_not_performed:
rollback:
human_decisions_required:
```
