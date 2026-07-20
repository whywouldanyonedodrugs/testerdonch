# Stage 18 — Repair Funding Economics, Build a Protected-Safe Funding Source, and Regenerate the Campaign Packet

```text
task_id: donch_bt_stage_18_funding_semantics_protected_safe_packet_20260720_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default

outcome_free_funding_engineering_authorized: yes
rankable_exact_funding_payload_read_authorized: yes — only through the protected-safe contract below
Parquet_footer_and_row_group_metadata_read_authorized: yes
protected_funding_payload_read_authorized: no
forward_strategy_price_or_return_read_authorized: no
phase_2_or_later_economics_authorized: no
Telegram_real_message_authorized: no
Capitalcom_payload_access: no
new_network_data_acquisition: no
orders_or_live_trading: no

funding_semantic_replacement_authorized: yes — exact scope below
campaign_packet_regeneration_authorized: yes
historical_terminal_decision_change_authorized: no
commit_authorized: yes
push_authorized: yes — non-force
```

## Objective

Resolve the Stage-17 global blocker without acquiring historical index/spot data and without opening protected payload rows.

Replace the invalid relative-rate/boundary-notional funding contract with a complete contract based on:

1. Kraken's exact absolute `fundingRate` where safely available in rankable history;
2. continuous time-weighted funding accrual over each held portion of an hourly funding period;
3. a frozen, direction-independent adverse funding allowance for periods without exact absolute funding;
4. exact-funding results as diagnostics rather than a calendar-dependent selection privilege.

Build a physically rankable-only exact-funding package using Parquet metadata and safe row-group reads. Regenerate all funding-dependent economic addresses, the campaign manifest and the non-authorizing approval packet. Obtain independent pre-outcome acceptance.

Do not launch the economic campaign.

## Current authority to verify

```text
expected origin/main and task start:
    91b9d8829a21e3c4eb429186c1075bddeb52523a

Stage-17 terminal status:
    blocked_preoutcome_common_funding_and_protected_read_defect

Stage-17 terminal root:
    /opt/testerdonch/results/rebaseline/phase_kraken_derivatives_campaign_stage17_20260720_v5

Stage-17 artifact manifest:
    c4b40b626c32d1721501afaab0a895e0bd8833d8a85be2aed0b08d4eb9c5f9eb

Stage-17 external approval:
    fe57d5c1efca3af3cb83c3e07b399e03c51f5dbe635b03bd48201944506c6853

Stage-16 campaign manifest:
    cc07499c671cf39b8ceaee91156f141dcc2c5532142af29a38a4f6830b73f23d

Stage-16 approval packet:
    c01281e50f40f95b922a04ed01c5b3d28ed325577891eed8e2ca5d32286965ca

Stage-16 economic translation registry:
    9c07f8695b117afe61f03354e0b6ab39a4c11bd4f0335f9f066cca33560ff1f8

Stage-16 search/bin specification:
    1cc994543371a8db86f428957a27d3a62d93f390e30f99460aa3e1d973b8c67b

Stage-16 estimator/rule inventory:
    b30b7c115d6d1ed765542c44d791c44117a0387f71ae4b33ec1219d4243a3b

Stage-16 inner-fold map:
    6ee8cbc53eac9326f904eec760ae82861a533b8165be6e1ccfc440dd0fa32ba0

Stage-16 utility/Pareto contract:
    8b9d6afa6c1ea6676cdac9b1703223ef1d47d929988c8e419fc1c112e4a3f093

Stage-16 boundary contract:
    460eb30f45c232c79729765b8724ddd206ed09671bb98f485b2175bf83cc5e75

Stage-16 funding contract, superseded only after this task is reviewed:
    d95c3b21c495a712efb6f9834200c6d3aa95f4bc7a9fda5492e2230a739fa04a

Stage-16 Telegram/supervision contract:
    f23736649f4523add8618720744673a5554d51246b9bcb01bcccf0a64591e70e

analytics manifest:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

authorized cohort:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636
```

Verify root, `AGENTS.md`, branches/remotes/worktrees, clean status, task archives, raw data manifests, exact funding source paths, supported commands and Drive target.

If `origin/main` differs only by an expected verified Drive/publication descendant, reconcile explicitly. Stop for substantive divergence.

## Stage-17 incident preservation

Preserve these facts exactly:

```text
protected non-outcome funding payload opened: yes
protected strategy-price outcomes opened: no
protected funding rows used for fitting or economics: no
exact count: unavailable and must not be recomputed by reopening protected payload
campaign economics executed: no
Telegram messages sent: no
```

Create and register:

```text
PROTECTED_ACCESS_INCIDENT_20260720_STAGE17.md
PROTECTED_ACCESS_INCIDENT_20260720_STAGE17.json
```

This is a bounded protected-access incident. It does not become a zero-access claim and does not rewrite the protected-period policy.

## Official funding interpretation to bind

Kraken's linear perpetual specification distinguishes:

```text
absolute funding rate:
    signed USD funding per one contract/base unit per hour

absolute payout:
    position quantity * absolute funding rate * fraction of funding period held

relative funding rate:
    intermediate percentage relative to the spot/index price
```

Funding accrues continuously and is realized at the hourly boundary or when the position changes. Therefore a trade entering or exiting inside an hour receives a prorated amount.

The replacement contract must use the exact absolute `fundingRate` field for safely available exact rows. It must not require or synthesize a boundary index, spot price or notional ratio.

Do not:

- set `boundary_notional / entry_notional = 1`;
- substitute mark or trade price for the Kraken spot/index inside the relative-rate formula;
- derive spot/index from future basis for funding cashflow;
- continue using imputed `relativeFundingRate` as though it were signed cashflow bps.

Preserve the old relative-rate model as provenance/diagnostic only. It may not enter selection, candidate ranking, route rescue or primary/stress economics.

# Part A — Protected-safe exact funding source

## 1. Metadata-before-payload contract

Create a source guard that verifies before any funding payload deserialization:

```text
platform
source manifest and file SHA-256
schema and timestamp column
rankable purpose
protected start = 2026-01-01T00:00:00Z
Parquet footer identity
row-group timestamp statistics
```

For every source row group:

```text
max_timestamp < protected_start:
    safe_rankable_row_group — may be read

min_timestamp >= protected_start:
    protected_row_group — skip without payload read

min_timestamp < protected_start <= max_timestamp:
    mixed_row_group — skip without payload read

missing, invalid or untrusted timestamp statistics:
    unknown_row_group — skip without payload read
```

Do not issue a broad file/table read with a filter applied after deserialization.

Instrument the low-level reader with an audit log proving exactly which row groups were read. Tests must fail if a protected, mixed or unknown row group is requested.

## 2. Physical rankable-only package

Read only `safe_rankable_row_group` payloads and write a new immutable package, for example:

```text
results/rebaseline/phase_kraken_rankable_exact_funding_safe_20260720_v1/
```

Required package:

```text
RANKABLE_EXACT_FUNDING.parquet or partitioned parquet
SOURCE_FILE_AND_ROW_GROUP_LEDGER.csv
SKIPPED_ROW_GROUP_LEDGER.csv
SCHEMA.json
COVERAGE.csv
PROTECTED_AUDIT.json
ARTIFACT_MANIFEST.json
README.md
```

The output must be physically rankable-only. Verify its maximum timestamp from both manifest and payload after the source guard has admitted the new file.

The new package becomes the only permitted exact-funding source for later campaign work. The old mixed raw paths remain immutable provenance and are forbidden to campaign loaders.

If no safe row groups are available or exact rows become too sparse, continue with the adverse-allowance contract below; do not open mixed/protected groups.

# Part B — Exact funding period and cashflow semantics

## 3. Funding-period timestamp adjudication

Using official documentation, existing source manifests, timestamp sequences and synthetic fixtures only, freeze whether each exact funding row timestamp denotes:

```text
funding-period start
or
funding-period end/realization
```

Create:

```text
FUNDING_PERIOD_SEMANTICS.json
```

It must define:

- hourly interval represented by a row;
- interval closure, e.g. `[start,end)`;
- rate availability time;
- DST/timezone treatment: UTC only;
- duplicate timestamp handling;
- missing hour handling;
- partial-period accrual;
- entry exactly at period start;
- exit exactly at period end;
- position change before an hourly boundary.

If historical row timestamp meaning cannot be resolved without inference beyond current authority, stop with the exact source needed. Do not run economics.

## 4. Exact absolute-rate formula

For a constant base-unit position during one funding period:

```text
position_sign:
    +1 long
    -1 short

held_fraction:
    overlap_seconds([entry_ts, exit_ts), funding_period) / 3600

exact_funding_cashflow_bps:
    -position_sign
    * absolute_funding_rate_usd_per_base_unit_per_hour
    * held_fraction
    / entry_trade_open_usd_per_base_unit
    * 10000
```

Sum across every funding period overlapping `[entry_ts, exit_ts)`.

Requirements:

- `absolute_funding_rate` retains its sign;
- quantity cancels only because the same base-unit quantity defines cashflow and entry notional;
- do not use boundary mark, trade, spot or basis inside the exact formula;
- partial first and last periods are included;
- no funding period overlap means exactly zero funding only after the hourly calendar proves it;
- missing exact funding does not imply zero and routes to the adverse allowance.

Create exact arithmetic fixtures for long/short, positive/negative rate, half-hour/full-hour, entry/exit at boundaries and multi-period holds.

# Part C — Frozen adverse allowance for non-exact funding

## 5. Calibration field

For each safely rankable exact row with an authorized contemporaneous mark at the funding-period start, compute only for cost calibration:

```text
absolute_hourly_funding_bps_on_mark_notional
    = abs(absolute_funding_rate) / mark_open_at_funding_period_start * 10000
```

This field does not claim actual trade funding. It calibrates a direction-independent adverse hourly allowance.

Use no forward strategy returns, candidate results, directions or contexts.

## 6. Quantile contract

Use empirical quantiles with an exact serialized method. Default:

```text
quantile method:
    Hyndman-Fan type 7 / linear interpolation

symbol minimum exact observations:
    720 valid hourly rows

base adverse allowance:
    symbol q95 of absolute_hourly_funding_bps_on_mark_notional

stress adverse allowance:
    symbol q99

fallback when symbol has fewer than 720 rows:
    equal-symbol-weighted pooled q95/q99 across eligible symbols
```

Requirements:

- calculate per-symbol distributions before the pooled fallback so high-row-count symbols do not dominate;
- finite nonnegative values only;
- q99 must be at least q95;
- no favourable sign or credit;
- no candidate-specific or fold-specific recalibration;
- one frozen allowance table is used for the entire programme-exposed campaign;
- report the late-slice/calendar limitation explicitly.

Create:

```text
FUNDING_ADVERSE_ALLOWANCE_TABLE.csv
FUNDING_ADVERSE_ALLOWANCE_CONTRACT.json
FUNDING_CALIBRATION_SUMMARY.md
```

## 7. Non-exact funding formula

For each funding period without safe exact absolute funding:

```text
base_missing_funding_cost_bps
    = -base_adverse_allowance_bps_per_hour * held_fraction

stress_missing_funding_cost_bps
    = -stress_adverse_allowance_bps_per_hour * held_fraction
```

This replaces the old `min(model_signed_cashflow_bps,-32/-64)` rule.

The allowance:

- cannot activate, rescue, rank or differentiate candidates except through holding duration and symbol;
- never becomes a credit;
- is applied identically to long and short positions;
- preserves missing-funding provenance;
- must cover every campaign symbol or fail the affected event before outcome.

# Part D — Revised economic metrics and information preservation

## 8. Primary and stress metrics

Replace the funding-dependent formulas with:

```text
base_funding_adjusted_net_bps:
    gross_return_bps
    - 14
    + exact absolute funding cashflow for safe exact periods
    + base adverse allowance cost for every non-exact period

stress_funding_adjusted_net_bps:
    gross_return_bps
    - 32
    + exact absolute funding cashflow for safe exact periods
    + stress adverse allowance cost for every non-exact period
```

However, to prevent calendar-specific exact funding from granting a selection advantage, freeze one of the following and justify it before outcomes:

```text
preferred selection contract:
    apply the base adverse allowance to every held funding period for every event,
    regardless of whether exact funding exists;
    use exact funding only as a diagnostic partition
```

This preferred contract is the default unless independent pre-outcome review demonstrates a stronger non-discriminatory alternative.

Under the preferred contract:

```text
primary selection metric:
    gross_return_bps - 14 - sum(base adverse allowance * held_fraction)

stress metric:
    gross_return_bps - 32 - sum(stress adverse allowance * held_fraction)

exact_funding diagnostic:
    replace the allowance with exact absolute cashflow only for events whose every overlapped period is safely exact

pre-funding diagnostic:
    gross_return_bps - 14
```

No exact or imputed funding result may rescue a candidate that fails the primary scenario. Report cost sensitivity rather than erasing information.

## 9. Funding evidence partitions

Report at minimum:

```text
pre_funding
uniform_adverse_allowance_primary
uniform_adverse_allowance_stress
fully_exact_diagnostic
mixed_exact_and_allowance_diagnostic
```

Selection uses only the frozen primary scenario. Exact/mixed partitions explain cost sensitivity and calendar dependence.

Retire these old selection partitions:

```text
relative-rate imputed primary
relative-rate imputed severe
favourable imputed funding
```

# Part E — Packet and identity regeneration

## 10. Economic-address changes

Funding contract identity is part of every economic address. Regenerate, do not edit in place:

```text
FUNDING_COST_AND_COVERAGE_CONTRACT.json
FUNDING_PERIOD_SEMANTICS.json
FUNDING_ADVERSE_ALLOWANCE_CONTRACT.json
BOUNDARY_AND_MISSINGNESS_CONTRACT.json
DEVELOPMENT_METRIC_CONTRACT.json
UTILITY_AND_PARETO_CONTRACT.json
ECONOMIC_TRANSLATION_REGISTRY.json
CAMPAIGN_MANIFEST.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md
RESOURCE_PROJECTION.json
```

Search bins, rule grammar, inner/outer folds, candidate beam, instruments, sides, entries and exits remain unchanged unless the funding period integration requires a purely mechanical hash-bound field. Any broader semantic change is outside scope.

The new packet must supersede Stage 16/17 only as a future non-authorizing packet. Historical task records remain immutable.

## 11. Coverage and resource projection

Coverage gates become:

```text
adverse allowance coverage:
    100% of campaign symbols and held period fractions

safe exact funding coverage:
    diagnostic; report by symbol, year and fold

protected-safe source coverage:
    all read row groups must be proven rankable before payload access
```

Re-benchmark the funding integration and full campaign projection outcome-free. Retain four workers, four-hour wall cap and 5 GiB limits when still supported; otherwise regenerate and request later approval.

# Part F — Tests and independent review

## 12. Mandatory protected-read tests

Use synthetic Parquet files containing:

- rankable-only row group;
- protected-only row group;
- mixed boundary row group;
- missing timestamp statistics;
- corrupt or false metadata;
- row group whose footer is safe but payload timestamp violates it.

Prove:

- only safe row groups are requested;
- mixed/protected/unknown groups are never deserialized;
- payload-level post-read assertion catches footer/payload contradiction and globally stops;
- no broad `read_parquet` or table load bypass exists;
- campaign loaders accept only the new physical rankable-only package.

## 13. Mandatory funding arithmetic tests

Test:

- absolute field sign and units;
- partial-hour accrual;
- long/short symmetry;
- positive/negative rates;
- multiple funding periods;
- exact-to-allowance transition;
- missing exact source;
- q95/q99 calculation and fallback;
- no favourable allowance;
- exact diagnostic cannot affect selection;
- economic-address hash changes with funding contract;
- no boundary index/spot/basis dependency remains.

## 14. Campaign semantic canary

Run synthetic end-to-end Phase 2–5 canary with the new funding contract. Prove candidate selection, freeze, evaluation and rolling replication are deterministic and require no new semantic choice.

No real strategy outcome reader may open.

## 15. Independent adversarial review

The reviewer must explicitly determine:

```text
protected-safe loader accepted or rejected
absolute funding period/cashflow semantics accepted or rejected
adverse allowance accepted or rejected
calendar/exact-funding selection privilege eliminated
packet complete without index/spot acquisition
no economic semantics remain to be invented after approval
```

Any residual ambiguity blocks packet publication.

# Part G — Documentation and Git

## 16. Durable updates

Update surgically:

- current continuity;
- current research decisions;
- test/evidence manual;
- protected-data incident register/guide;
- Kraken funding and analytics learnings;
- campaign registry and packet lineage.

Do not change strategy/family terminal decisions.

## 17. Git and handoff

Start from a clean verified repository. Keep unrelated work out of scope.

Create separate reviewed commits for:

1. protected-safe source/loader and tests;
2. funding semantics and campaign packet regeneration;
3. documentation, archive and manifest closure;
4. Drive verification.

Non-force push only. Finish with clean main and task worktree.

Complete `drive_handoff: approved_default`, round-trip verify every compact file and retain the local archive.

# Required outputs

```text
PROTECTED_ACCESS_INCIDENT_20260720_STAGE17.md
PROTECTED_ACCESS_INCIDENT_20260720_STAGE17.json
RANKABLE_EXACT_FUNDING_SOURCE_MANIFEST.json
SOURCE_FILE_AND_ROW_GROUP_LEDGER.csv
SKIPPED_ROW_GROUP_LEDGER.csv
PROTECTED_SAFE_FUNDING_READER_CONTRACT.json
FUNDING_PERIOD_SEMANTICS.json
FUNDING_ADVERSE_ALLOWANCE_TABLE.csv
FUNDING_ADVERSE_ALLOWANCE_CONTRACT.json
FUNDING_CALIBRATION_SUMMARY.md
FUNDING_COST_AND_COVERAGE_CONTRACT.json
BOUNDARY_AND_MISSINGNESS_CONTRACT.json
DEVELOPMENT_METRIC_CONTRACT.json
UTILITY_AND_PARETO_CONTRACT.json
ECONOMIC_TRANSLATION_REGISTRY.json
CAMPAIGN_MANIFEST.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md
RESOURCE_PROJECTION.json
CAMPAIGN_PACKET_LAUNCH_READINESS.md
SYNTHETIC_CANARY.json
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large physical funding data may remain local and hash-manifested.

# Stop conditions

Stop globally for:

- source/hash divergence;
- any protected/mixed/unknown row-group payload read;
- unresolved exact funding row timestamp semantics;
- inability to verify the absolute `fundingRate` field;
- allowance calibration requiring strategy outcomes;
- economic outcome access;
- protected/Capital.com access;
- packet or canary nondeterminism.

If safe exact funding is unavailable, the adverse allowance may still support a complete packet when its calibration fallback is valid. Do not weaken the allowance or use zero funding to force readiness.

Do not launch the campaign.

# Final response

```text
status:
actual_starting_commit:
Stage17_authority_verified:
incident_registered:
protected_payload_rows_opened_in_Stage18: no
safe_row_groups_read:
skipped_protected_mixed_unknown_row_groups:
rankable_exact_funding_package_hash:
funding_period_semantics:
absolute_funding_formula_verified:
adverse_allowance_method_and_hash:
base_and_stress_allowance_summary:
selection_funding_contract:
old_relative_imputation_selection_retired:
economic_translation_registry_hash:
replacement_campaign_manifest_hash:
replacement_approval_packet_hash:
cell_count_and_resource_projection:
synthetic_canary:
launch_readiness_review:
economic_outputs_computed: no
protected_strategy_outcomes_opened: no
Capitalcom_payload_opened: no
Telegram_messages_sent: no
tests_and_reviews:
files_and_commits:
origin_main_updated:
original_checkout_final_status:
task_worktree_final_status:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
human_approval_required:
```
