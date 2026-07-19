# Stage 8C — KDA01 v2 Frozen Level-3 Economic Execution

```text
task_id: donch_bt_stage_8c_kda01_level3_economic_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
economic_run_authorized: yes — exact frozen KDA01 v2 Level-3 contract only
candidate_returns_authorized: yes — exact frozen definitions only
controls_authorized: no
protected_outcome_access: no
new_data_acquisition: no
Capitalcom_payload_access: no
commit_authorized: yes — task-scoped reviewed commits
push_authorized: yes — non-force under the standing workflow
```

## Human approval

The human explicitly approves implementation and one deterministic economic execution of the frozen KDA01 v2 Level-3 contract:

```text
contract version:
    kda01_level3_contract_v2_20260719

contract hash:
    d2112d8fb4f0666890761b70f7fa71c9e8cb4f43c704a04cdf9470e949a177b3

definitions:
    16

primary definitions:
    8

robustness-only definitions:
    8

controls:
    7 frozen but NOT authorized for execution in this task
```

No threshold, feature, branch, side, horizon, cluster, cost, execution delay, overlap rule, bootstrap rule, gate, or control may be changed after outcome access.

## Objective

Implement and execute the exact 16-definition KDA01 v2 Level-3 economic contract on Kraken rankable data. Produce one terminal Level-3 decision:

```text
KDA01_level3_primary_pass_controls_required
KDA01_level3_no_primary_pass_stop
blocked_with_exact_mechanical_remedy
```

Do not execute controls, Level 4, validation, protected data, Capital.com data, portfolio pooling, or another hypothesis.

## Current authority to verify

```text
expected starting commit:
    55c75ef0564b004413c19d670625e44a1838a537

Stage 8B1 task archive:
    docs/agent/task_archive/20260719_donch_bt_stage_8b1_kda01_contract_closure_20260719_v1

Stage 8B task archive:
    docs/agent/task_archive/20260719_donch_bt_stage_8b_kda01_prerun_freeze_20260719_v1

Stage 8A task archive:
    docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1

Stage 8B1 contract JSON:
    KDA01_LEVEL3_DECISION_RULES_V2.json
    file SHA-256: f0488650eed87c714a46acc6e74def362f200240d8733fa09500b679464a56b5

Stage 8B1 definition register:
    KDA01_LEVEL3_DEFINITION_REGISTER_V2.csv
    file SHA-256: ef32ebd1032c79e9f59df71b463d8e6d509b2ccfc8ab3eb75db2449d5d2aac7a

Stage 8B1 event-cluster tape:
    KDA01_V2_EVENT_CLUSTER_IDENTITY.parquet
    SHA-256: 1a2946a2da8c5bf9b1a4cbc9571abad31d968034f246fe94a8e442e959864669

Stage 8B source event tape:
    SHA-256: 7c9e682380681aa6fc83161ae29dafab9a047bdb19fe1fa8ade3ba5a4eb9c2e5

Stage 8B source parent tape:
    SHA-256: ae470d06c7da049d145ac8e083d0c4711c9c2dd6e4ec32fedf79224e8885ddbd

Stage 8B artifact-manifest content hash:
    569bb1c53ba58767091c42dbf96ead63b3df5a45ae7cad1b4f6d4425e5ef61d5

Stage 8B artifact-manifest file SHA-256:
    ee7db729eac30363c6147984658777ed92dc06f1d436527a56efae5bb997f669

analytics data-manifest hash:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

semantic contract hash:
    289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60

cohort hash:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636

rankable interval:
    [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
```

Verify root, `AGENTS.md` chain, branch/commit/remotes, clean worktree, supported commands, source manifests, all named local files/hashes, archive convention, and Drive target before opening price columns.

Stop if any contract, event, cluster, data, schema, purpose, or content identity does not reconcile exactly.

## Archive context

Preserve unchanged:

```text
C01: level3_no_primary_pass_stop
C02: level3_no_primary_pass_stop
C03: C03_PIT_authority_unavailable
C16: C16_flow_authority_unavailable
```

KDA01 v1 remains broad outcome-free provenance and cannot rescue KDA01 v2. KDA02 and KDA03 remain unexecuted.

Same-sample prohibitions remain active for C01, C02, RFBS, Backside, failed-breakdown, LFBS, H43, TSMOM, A1, and every older translation. Do not condition or subgroup KDA01 results using those historical outcomes.

## Exact definitions

Execute only the definitions serialized in the verified Stage 8B1 contract JSON and definition register.

Primary branches, each at 1h and 6h:

```text
positive efficient crowding continuation -> long
negative efficient crowding continuation -> short
positive crowding completed failure -> short
negative crowding completed failure -> long
```

Execute the corresponding eight robustness definitions exactly as serialized. Robustness results are diagnostic and cannot rescue a failed primary definition or branch.

No anticipatory deteriorating-price fade, stop, add, partial exit, dynamic sizing, leverage rule, symbol subset, year subset, session subset, or alternative horizon is authorized.

## Reconstruct accepted execution records

Reuse the reviewed Stage 8B1 timestamp-availability and non-overlap implementation rather than creating new scheduling logic.

The deterministic reconstruction must reproduce before price access:

```text
definition-event records:
    204,272

accepted:
    183,744

rejected/skipped:
    20,528

actual-position overlap:
    20,473

missing exit bar:
    55

entry-delay exceeded:
    0

exit-delay exceeded:
    0

missing entry:
    0

protected timestamps:
    0
```

Entry/exit availability rules:

```text
entry:
    first PF 5m trade-bar open strictly after decision

expected next open:
    first five-minute UTC grid timestamp strictly after decision

maximum entry delay:
    10 minutes

exit target:
    actual entry timestamp + frozen 1h or 6h timeout

exit:
    first PF 5m trade-bar open at or after target

maximum exit delay:
    10 minutes

non-overlap:
    definition-local and symbol-local using actual eligible exit timestamp
```

Do not proceed to prices if the exact reconstruction differs.

## Price and return contract

Use only official rankable Kraken PF trade-bar `open` at the already frozen `entry_ts` and `exit_ts`.

Mark data affected event construction but is not a fill price. Do not use close, high, low, VWAP, midpoint, spot, mark, index, or another venue for entry or exit.

Reject a record if either exact open is absent, non-finite, or non-positive. Report the rejection without replacing or delaying the price.

Direction mapping is frozen by branch:

```text
positive efficient continuation: +1
negative efficient continuation: -1
positive completed failure: -1
negative completed failure: +1
```

Apply the same branch-side mapping to robustness definitions.

For each accepted record:

```text
gross_bps = side * (exit_open / entry_open - 1) * 10,000
base_net_bps = gross_bps - 14
stress_net_bps = gross_bps - 32
```

The 14-bps and 32-bps values are complete round-trip cost scenarios. Do not add another fee or slippage deduction.

Use fixed notional and no compounding. Do not construct a portfolio or capital-allocation simulation.

## Funding

Reuse only the repository's existing tested Kraken funding authority and signed funding implementation.

Classify each executed trade as:

```text
zero_boundary
fully_exact
mixed
fully_imputed
```

Report counts and funding-inclusive diagnostics separately. Funding is excluded from every Level-3 gate and cannot rescue or reject a definition.

Do not use funding state to select events, branches, symbols, years, horizons, or thresholds.

If existing funding identity cannot be reconciled without changing the contract, report funding as unavailable for that record while leaving the frozen no-funding Level-3 gates intact.

## Primary equal-market-day inference

For each definition independently:

1. Aggregate its accepted trade returns inside each `market_day_cluster_id` using an equal-weight arithmetic mean.
2. Treat each resulting market-day cluster return as one equally weighted primary observation.
3. Compute the primary base/stress mean and median across equal-weight market-day returns.
4. Bootstrap the equal-weight market-day returns with replacement:

```text
resamples: 10,000
seed: 20260719
statistic: arithmetic mean of sampled market-day returns
lower bound: percentile 2.5%
```

Also report trade-level summaries, but they are descriptive and do not replace the equal-market-day gates.

Sensitivity-only inference:

```text
market_6h_cluster_id
parent_episode_id
```

Use the same cluster-mean/bootstrap method. Sensitivities cannot override the primary market-day decision.

## Concentration

Under fixed notional, aggregate `base_net_bps` contributions by:

```text
market_day_cluster_id
symbol
year
```

For each grouping:

```text
maximum positive group contribution /
sum of all positive group contributions
```

Undefined or non-positive denominators fail closed.

Frozen limits:

```text
market day <= 10%
symbol <= 25%
year <= 70%
```

Report gross and stress concentration descriptively, but only the base-net concentration gates decide Level 3.

## Frozen Level-3 gates

Apply to each primary definition independently:

```text
executed trades >= 100
executed trades in each of 2023, 2024, 2025 >= 20
equal-market-day base-net mean > 0 bps
equal-market-day base-net median > 0 bps
market-day bootstrap 95% lower bound >= -5 bps
maximum positive market-day contribution <= 10%
maximum positive-symbol contribution <= 25%
maximum positive-year contribution <= 70%
equal-market-day stress-net mean >= -10 bps
```

All gates must pass. No rounding may convert a failure into a pass. Store full-precision values and a displayed rounded value.

Robustness definitions:

- report every corresponding metric and gate;
- cannot create a primary pass;
- cannot rescue a failed primary horizon, direction, or branch.

No pooling across definitions, branches, horizons, directions, or attempts.

## Terminal decision

Return:

```text
KDA01_level3_primary_pass_controls_required
```

when at least one primary definition passes every frozen gate. Freeze every passing primary definition; do not select only the best one. Controls remain unexecuted and require a separate human-approved task.

Return:

```text
KDA01_level3_no_primary_pass_stop
```

when no primary definition passes. Close this exact translation. Do not reverse direction, alter thresholds, change horizons, select years/symbols, or use robustness to continue.

Return:

```text
blocked_with_exact_mechanical_remedy
```

only for a verifiable implementation/data/authority failure before a valid economic decision.

## Required outputs

```text
KDA01_LEVEL3_RUN_CONTRACT_SNAPSHOT.json
KDA01_LEVEL3_ACCEPTED_EXECUTION_RECORDS.parquet
KDA01_LEVEL3_TRADE_TAPE.parquet
KDA01_LEVEL3_DEFINITION_METRICS.csv
KDA01_LEVEL3_GATE_MATRIX.csv
KDA01_LEVEL3_MARKET_DAY_RETURNS.parquet
KDA01_LEVEL3_BOOTSTRAP_SUMMARY.csv
KDA01_LEVEL3_BOOTSTRAP_DISTRIBUTIONS.parquet
KDA01_LEVEL3_CONCENTRATION.csv
KDA01_LEVEL3_YEAR_MONTH_SYMBOL_SUMMARY.csv
KDA01_LEVEL3_FUNDING_PARTITIONS.csv
KDA01_LEVEL3_CLUSTER_SENSITIVITY.csv
KDA01_LEVEL3_DEFINITION_DECISIONS.csv
KDA01_LEVEL3_DECISION.md
KDA01_LEVEL3_CLAIM_BOUNDARY.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large Parquet outputs and bootstrap distributions remain local and hash-manifested. The Drive handoff must include compact complete metrics, gates, decision, contract snapshot, validation, review, and manifest.

## Required reporting

For every definition report at least:

```text
branch
attempt
side
timeout
candidate count
accepted count
price rejection count
trades by year
symbols
market-day clusters
trade-level gross/base/stress mean and median
equal-day gross/base/stress mean and median
market-day bootstrap interval
market-day/symbol/year concentration
gross/base/stress quantiles
win rate
yearly and monthly summaries
funding partition counts and diagnostics
all gate values and pass/fail
terminal definition decision
```

Show primary definitions first. Robustness results must be visually and structurally separated.

## Tests

Use synthetic fixtures for:

1. long and short return arithmetic;
2. exact open-price lookup;
3. no close/high/low/mark fill substitution;
4. 14/32-bps cost deduction;
5. reconstruction of accepted/rejected scheduling;
6. definition-local actual-exit overlap;
7. equal-market-day aggregation;
8. fixed-seed cluster bootstrap;
9. market-day/symbol/year concentration;
10. undefined concentration denominator fails closed;
11. primary/robustness separation;
12. no pooling;
13. funding partition exclusion from gates;
14. exact threshold boundaries and no rounded pass;
15. no pre-2023 or 2026+ price/outcome row;
16. deterministic run and artifact hashes.

Run focused tests and relevant Stage 8A/8B/8B1, source-authority, loader, lifecycle, analytics, funding, overlap, protected-boundary, archive, and decision regressions.

Require an independent review of actual code, contract identity, input hashes, price fields, accepted executions, return arithmetic, costs, clustering, bootstrap, concentration, funding separation, gate decisions, claims, artifacts, and prohibited actions.

## Integration and handoff

After deterministic execution and independent review:

- update factual KDA01 family, attempt, run, evidence, multiplicity, capability, and continuity records;
- preserve all previous roots and negative findings;
- create task-scoped reviewed commits;
- non-force push under the standing workflow;
- use `drive_handoff: approved_default`;
- round-trip verify all compact uploaded files by size and SHA-256;
- retain the full local run and handoff archive.

## Stop conditions

Stop and preserve evidence if:

- contract or source identity differs;
- accepted execution reconstruction differs;
- protected data might be opened;
- a price source or field is ambiguous;
- an outcome-dependent change is proposed;
- controls or another family enter the task;
- Git state is unsafe;
- deterministic verification fails.

## Final response

```text
status:
actual_starting_commit:
run_root:
contract_hash_verified:
source_and_event_hashes_verified:
accepted_execution_reconciliation:
price_fields_and_execution:
primary_definition_results:
robustness_definition_results:
market_day_bootstrap_results:
concentration_results:
funding_partition_results:
primary_pass_count:
passing_primary_definitions:
controls_executed: no
protected_rows_opened: no
economic_outputs_computed: yes
tests_and_review:
data_and_artifact_manifest_hashes:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
terminal_decision:
human_approval_required:
```
