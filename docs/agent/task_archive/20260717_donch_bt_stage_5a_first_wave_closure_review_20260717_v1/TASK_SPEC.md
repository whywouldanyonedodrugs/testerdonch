# Stage 5A — First-Wave Closure and Next-Wave Review Bundle

```text
task_id: donch_bt_stage_5a_first_wave_closure_review_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no new computation
protected_outcome_access: no
capture_access: no
new_market_data_acquisition: no
```

## Objective

Close the first research wave, create one comprehensive analyst-ready decision bundle, update durable project records, and recommend the smallest justified next-wave preflight.

This is a synthesis and evidence-packaging task. Do not run new economics, regenerate candidates, inspect protected outcomes, or rescue failed definitions.

## Binding first-wave decisions

Preserve exactly:

### C01

```text
family:
    C01_debetaed_residual_shock_path_bifurcation

decision:
    level3_no_primary_pass_stop

authoritative run root:
    results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227

closure commit:
    b4e6c24c7c1f6c54dea931d824383590d087b819
```

### C02

```text
family:
    C02_positive_resolution_aware_spot_led_continuation_v1

decision:
    level3_no_primary_pass_stop

authoritative run root:
    results/rebaseline/phase_kraken_c02_positive_spot_led_level3_20260717_v1_20260717_161958

closure commit:
    4c0608195f9f8c942d5de394a5189a1c734130bf
```

### C03

```text
family:
    C03_PIT_cohort_breadth_and_dispersion_context

decision:
    C03_PIT_authority_unavailable

authoritative authority root:
    results/rebaseline/phase_kraken_c03_pit_cohort_breadth_20260717_v1_20260717_173151

handoff commit:
    8681d8fa92809b673da3e67b1bce21cbaadf88ba
```

C03 is an evidence-availability result, not a negative economic result.

## Start checks

Verify:

- repository root and applicable `AGENTS.md` chain;
- clean synchronized `main`;
- current commits/remotes;
- all authoritative run roots and task archives;
- hashes and manifests;
- current family, attempt, data-capability, run, supersession, and continuity records;
- approved Drive handoff target.

Record the actual starting commit. Stop for divergent or missing authoritative roots.

## Source scope

Use repository-local machine evidence and the current closed external-review package.

Read:

- C01 Stage 2A through 2E task archives and final run root;
- C02 Stage 3A through 3E task archives and final run root;
- C03 Stage 4A task archive and authority root;
- the prior 14-family external-review package and its narrowed disposition;
- current hypothesis/family, attempt/multiplicity, data-capability, run, defect, and supersession registries;
- current continuity and research-decision records.

Do not open:

- protected-period data or outcomes;
- capture payloads;
- omitted raw market archives;
- any economic object outside named authoritative roots;
- superseded result roots except for provenance identity.

## Required bundle

Create a closed package with stable machine-readable and analyst-readable outputs.

### 1. Read-first and authority map

```text
READ_FIRST.md
AUTHORITY_AND_SCOPE.md
SOURCE_AND_RUN_ROOT_INDEX.csv
```

State:

- venue and period;
- authority order;
- no strategy is validation-grade or live-ready;
- package purpose and claim boundary;
- exact authoritative roots, commits, manifests, and hashes;
- protected data status;
- unavailable evidence.

### 2. First-wave decision matrix

```text
FIRST_WAVE_DECISION_MATRIX.csv
FIRST_WAVE_EXECUTIVE_REVIEW.md
```

For C01, C02, and C03 store separately:

```text
hypothesis_and_mechanism
translation
data_and_universe_authority
generator_contract
event_identity
economic_definition_count
events_and_executed_trades
entry_exit_execution_semantics
cost_and_funding_treatment
Level3_gate_result
evidence_level
reproducibility_status
validation_status
deployment_status
decision
claim_supported
claim_not_supported
same_sample_actions_forbidden
permitted_future_use
```

### 3. Detailed C01/C02 mechanics and economics

```text
C01_DETAILED_REVIEW.md
C02_DETAILED_REVIEW.md
DEFINITION_METRICS_COMPARISON.csv
ENTRY_EXIT_EXECUTION_MATRIX.csv
EVENT_EPISODE_AND_OVERLAP_MATRIX.csv
COST_FUNDING_AND_EXECUTION_EVIDENCE.md
CONCENTRATION_AND_UNCERTAINTY_REVIEW.md
```

Include exact:

- feature and event definitions;
- candidate/onset/event/episode counts;
- branch identities;
- entries, exits, skips, invalid rows, non-overlap;
- gross, base-net, stress-net mean/median;
- bootstrap intervals and gates;
- symbol/year/episode concentration;
- funding partitions and exactness limits;
- execution approximations;
- protected-row counts;
- tests and review results.

Do not calculate new variants or aggregate definitions into a portfolio.

### 4. C03 authority review

```text
C03_AUTHORITY_FAILURE_REVIEW.md
C03_MEMBERSHIP_AUTHORITY_SUMMARY.csv
C03_LIFECYCLE_GAP_REGISTER.csv
```

Explain:

- 479 observed PF identities;
- zero verified full-day eligible members under the frozen standard;
- unknown versus ineligible counts;
- source coverage;
- why sparse point snapshots, current roster, and bar existence cannot establish continuous PIT status;
- unresolved resumptions and identity collision;
- why Phase B was correctly omitted;
- what exact authority would be needed to retry.

Make clear that this is `unavailable`, not evidence that historical breadth has no value.

### 5. Prior-research integration

```text
PROGRAM_DECISION_REGISTER.csv
PRIOR_AND_FIRST_WAVE_LINEAGE_MAP.csv
NEGATIVE_AND_BLOCKED_DECISIONS.md
SAME_SAMPLE_PROHIBITIONS.md
```

Integrate the prior 14 decision-bearing family lineages with C01, C02, and C03.

Preserve the narrowed external-package claim:

```text
package_role:
    strategic_and_continuity_review_only

independent_row_level_reproduction:
    unavailable

package_release_ready:
    false
```

Never convert missing prior hashes, test counts, raw extracts, source snapshots, TSMOM coverage, or causal episode starts into pass.

### 6. Engineering and data assets

```text
REUSABLE_INFRASTRUCTURE_AND_DATA_ASSETS.md
DATA_CAPABILITY_AFTER_FIRST_WAVE.csv
ENGINEERING_TEST_AND_DEFECT_AUDIT.md
```

Record reusable assets, including:

- protected-safe loader boundaries;
- lifecycle and reference-panel authority;
- C01 residual and onset infrastructure;
- verified PF candle-volume semantics and causal liquidity proxy;
- official Kraken spot history, pair mappings, and gap masks;
- resolution-aware spot/PF leadership;
- canonical episode and actual-exit non-overlap;
- economic runners, funding partitions, bootstrap, and gates;
- defects found and repaired;
- tests, commands, resource use, and remaining limitations.

Separate “infrastructure verified” from “strategy evidence negative.”

### 7. Next-wave decision

Review only the pre-existing candidate set:

```text
C16 ETF/regulated-access flow transmission
C14 exact-slice funding-boundary/premium survival
C17 executed catalyst state
C13 listing-age asymmetry
C12 lifecycle/status reconstruction
prospective C08/C09/C11/C15
```

Do not invent new hypotheses from C01/C02 outcomes.

Create:

```text
NEXT_WAVE_READINESS_MATRIX.csv
NEXT_WAVE_RECOMMENDATION.md
NEXT_TASK_SPECIFICATION.md
```

For each candidate report:

```text
mechanism
distinctness_from_closed_work
required_data
currently_verified_data
missing_authority
historical_period_available
sample_and_opportunity expectations
overlap_and_multiplicity risk
implementation cost
protected-data route
smallest non-economic preflight
stop condition
recommended_priority
```

The recommendation must select exactly one smallest next non-economic preflight or recommend a deliberate pause.

C03/lifecycle rule:

- Do not recommend an open-ended lifecycle crawl merely because C03 failed.
- Treat broad lifecycle reconstruction as a separate long-term infrastructure program.
- Recommend it now only if a bounded source-coverage feasibility audit shows a plausible path to the frozen PIT threshold and the expected value exceeds the next directional preflight.
- Current-survivor or bar-existence breadth may not replace PIT breadth.

Default strategic prior, subject to verified current data:

```text
C16 official ETF/regulated-access flow-data authority preflight
```

because it is externally observed, relatively low-overlap, and does not depend on C03. This is a prior for review, not a mandated result.

## Outside-analyst usability

Create:

```text
OUTSIDE_ANALYST_REVIEW_GUIDE.md
OUTSIDE_ANALYST_QUESTIONS.md
```

The guide must explain the project in plain language while retaining exact technical references.

Questions should cover:

- whether the mechanisms were translated faithfully;
- whether event identity and non-overlap are defensible;
- whether cost/funding/execution treatment changes the bounded decision;
- whether negative conclusions are scoped correctly;
- whether any next-wave candidate is genuinely distinct;
- whether missing evidence is being mistaken for zero effect.

## Durable records

Update, only where facts changed:

- family/hypothesis registry;
- attempt/multiplicity registry;
- run registry;
- data-capability registry;
- defect/repair registry;
- supersession map;
- continuity brief;
- next-action record.

Preserve old records and provenance. Do not rewrite historical decisions.

## Validation

Verify:

- all named roots and manifests;
- metric and gate values copied exactly from authoritative artifacts;
- zero protected-data access;
- no new economic computation;
- no pooled portfolio calculation;
- no missing evidence converted to pass;
- all terminal decisions and same-sample prohibitions present;
- package manifest, sizes, SHA-256;
- secret scan;
- independent review of the actual bundle and registry diffs.

## Drive handoff

Use:

```text
drive_handoff: approved_default
```

Package the closed review bundle, upload to the configured `DONCH_BACKTESTING_HANDOFFS` root, round-trip verify every file, and retain the local package.

Large raw data and local Parquet tapes remain excluded. Include only decision-bearing derived evidence, manifests, source indexes, and compact tables.

## Final status

Return exactly one:

```text
first_wave_closed_next_preflight_ready
first_wave_closed_deliberate_pause
blocked_by_missing_authoritative_root
```

No economic work is authorized.

## Final response

```text
status:
actual_starting_commit:
C01_decision:
C02_decision:
C03_decision:
authoritative_roots_verified:
prior_lineages_integrated:
registries_and_continuity_updated:
review_bundle:
bundle_manifest_and_sha256:
independent_review:
next_wave_ranking:
recommended_next_preflight:
economic_outputs_computed: no
protected_outcomes_opened: no
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
human_approval_required:
```
