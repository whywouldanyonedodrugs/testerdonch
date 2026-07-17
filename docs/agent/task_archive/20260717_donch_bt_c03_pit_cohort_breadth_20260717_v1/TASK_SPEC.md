C03_PIT_cohort_breadth_and_dispersion_context
```

Do not use C03 to rescue the closed C01 or C02 translations.

## Archive context

Current terminal decisions:

```text
C01:
    level3_no_primary_pass_stop
    run root:
      results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227

C02:
    level3_no_primary_pass_stop
    run root:
      results/rebaseline/phase_kraken_c02_positive_spot_led_level3_20260717_v1_20260717_161958
    closure commit:
      4c0608195f9f8c942d5de394a5189a1c734130bf
```

Preserve both decisions. No opposite-direction test, subset rescue, new horizon, threshold change, or Level-4 control run is authorized.

C03 was pre-registered before those outcomes. This task may assess context-feature authority and coverage, but it may not inspect or condition on C01/C02 economic results.

## Authority boundary

Use this source order:

1. repository machine contracts and finalized manifests;
2. existing Stage 2A/2A1 lifecycle authority and source ledgers;
3. Stage 2C1 PF semantic histories and official terminal-event ledger;
4. archived official Kraken instrument snapshots/notices;
5. official opening/listing, suspension, wind-down, delisting, settlement, migration, and resumption notices;
6. trade/mark manifests only as operational bar-availability evidence.

Current roster membership or bar existence alone is not lifecycle proof.

Public read-only acquisition is authorized only for missing official Kraken lifecycle, identity, or specification evidence. Do not acquire new economic/context datasets from other venues or vendors.

## Phase A — PIT membership authority

Build the full observed Kraken PF identity inventory from:

- every archived/current official instrument snapshot already retained;
- official terminal/delisting/settlement ledger;
- official opening-date and symbol-migration authority;
- existing trade and mark manifests.

For every UTC day in `[2023-01-01, 2026-01-01)`, classify each identity:

```text
verified_eligible
verified_ineligible
unknown
```

Required fields:

```text
date
canonical_asset_id
PF_symbol
identity_version
opening_or_start
terminal_or_end
status_intervals
status_class
status_confidence
source_ids_and_hashes
trade_bar_available
mark_bar_available
unknown_reason
```

Rules:

- unknown remains unknown;
- absence from a current roster is not proof of historical ineligibility;
- absence from a terminal table is not proof of uninterrupted eligibility;
- bar presence supports operational availability only;
- known suspension/post-only/settlement/migration intervals fail closed;
- no 2026 outcome-bearing payload may be opened.

## PIT feasibility gate

Before breadth construction, report daily and yearly:

```text
potentially_eligible_identity_count
verified_eligible_count
verified_ineligible_count
unknown_count
unknown_share
trade_mark_covered_eligible_count
```

Primary PIT breadth is feasible only if all predeclared conditions pass:

```text
at least 30 verified eligible and trade/mark-covered members per usable day
unknown share <= 20% per usable day
at least 80% usable days in each of 2023, 2024, and 2025
no unresolved identity collision in an included member
```

These are measurement-feasibility thresholds, not alpha parameters. Do not relax them from the result.

If the gate fails, stop after the authority package and return:

```text
C03_PIT_authority_unavailable
```

Do not substitute current-survivor breadth as the primary C03 panel.

## Phase B — Cohorts and causal features

Proceed only if Phase A passes.

### Cohorts

Build only:

```text
listing age:
    <90 days
    90-365 days
    >365 days

lagged liquidity:
    prior-day causal Top-100
    prior-day liquidity quintiles

fixed group:
    BTC
    ETH
    eligible liquid alts
```

Use the existing verified `close_based_usd_volume_proxy` and prior-day ranking contract.

Do not use narrative sectors or current taxonomy.

### Feature windows

Compute from completed Kraken PF trade bars with matching mark availability:

```text
primary horizon:
    1 hour

robustness horizon:
    6 hours
```

At each completed timestamp store:

```text
eligible_member_count
known_unknown_excluded_counts
positive_return_fraction
negative_return_fraction
signed_breadth
cross_sectional_return_median
cross_sectional_dispersion
top_1_absolute_move_share
top_5_absolute_move_share
participation_concentration
```

All features end at the timestamp. No post-timestamp return may be read or stored.

Require at least:

```text
20 valid members per cohort/timestamp
80% bar coverage among verified eligible cohort members
```

Otherwise mark the feature unavailable.

### Current-survivor and aggregate controls

Compute separately:

```text
current_roster_survivor_breadth_control
aggregate_bar_existence_breadth_control
```

These are bias/measurement controls, never PIT truth.

Report correlations and distributional differences among feature panels without joining to later outcomes.

## Frozen-event coverage diagnostics

Using safe identity/timestamp columns only, attach C03 feature availability to:

- all frozen C01 onset events;
- all frozen C02 Stage 3C resolution-aware events.

Do not read their trade ledgers, returns, gates, or decisions during the join.

Report:

```text
event coverage
missing-context reasons
denominator size
PIT versus current-survivor feature difference
```

This is coverage and measurement evidence only. It does not reopen C01/C02 or authorize an interaction test.

## Required outputs

```text
C03_LIFECYCLE_SOURCE_LEDGER.csv
C03_PIT_IDENTITY_INVENTORY.csv
C03_DAILY_MEMBERSHIP_AUTHORITY.parquet
C03_PIT_FEASIBILITY_REPORT.md
C03_PIT_FEASIBILITY_MATRIX.csv
C03_COHORT_CONTRACT.md
C03_BREADTH_FEATURE_SCHEMA.json
C03_BREADTH_FEATURE_PANEL.parquet
C03_CURRENT_SURVIVOR_CONTROL_PANEL.parquet
C03_AGGREGATE_BAR_EXISTENCE_CONTROL_PANEL.parquet
C03_PANEL_COMPARISON_REPORT.md
C03_FROZEN_EVENT_COVERAGE.csv
C03_NEXT_CONTRACT_RECOMMENDATION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large Parquet files remain local and hash-manifested; omit them from Drive.

## Tests

Use synthetic fixtures for:

- daily PIT identity/status transitions;
- unknown state preservation;
- opening, suspension, migration, resumption, delisting, and settlement boundaries;
- no current-roster or bar-existence inference;
- feasibility thresholds at exact boundaries;
- prior-day-only liquidity cohort membership;
- 1h/6h causal breadth arithmetic;
- denominator and missing-member rules;
- current-survivor control separation;
- safe C01/C02 event coverage joins;
- no post-event, protected, pre-2023, non-Kraken, or outcome-derived rows;
- deterministic hashes and replay.

Run relevant lifecycle, loader, protected-boundary, identity, cohort, and archive regressions. Require independent review.

## Acceptance criteria

```text
C01 and C02 terminal decisions preserved: yes
economic outputs computed: zero
protected outcomes opened: zero
PIT statuses preserve unknowns: yes
current-survivor control never presented as PIT: yes
PIT feasibility gate: pass or explicit unavailable
feature panels produced only after gate pass: yes
independent review: approve
```

## Integration and handoff

After review:

- create task-scoped commit(s);
- non-force push under the standing reviewed-task workflow;
- upload with `drive_handoff: approved_default`;
- verify remote size and SHA-256;
- retain local full artifacts;
- update factual lifecycle, data-capability, family, multiplicity, and continuity records only.

## Final status

Return exactly one:

```text
ready_for_C03_context_contract_review
C03_PIT_authority_unavailable
blocked_with_exact_non_economic_remedy
```

No C03 economic/context run is authorized.

## Final response

```text
status:
actual_starting_commit:
C01_C02_terminal_record_status:
identity_inventory:
official_sources_acquired:
daily_membership_summary:
PIT_feasibility_gate:
cohort_contract:
breadth_feature_summary:
current_survivor_control_comparison:
frozen_event_coverage:
tests_and_review:
protected_outcomes_opened: no
economic_outputs_computed: no
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
next_status:
human_approval_required:
```
