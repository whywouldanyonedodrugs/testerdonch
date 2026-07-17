# Stage 3D — C02 Positive Spot-Led Pre-Run Contract

```text
task_id: donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
new_market_data_acquisition: no
```

## Objective

Freeze one approval-ready Level-3 economic contract for the only mechanically supported C02 branch:

```text
positive resolved spot-led continuation
```

Do not compute post-decision returns, entries, exits, PnL, MAE/MFE, controls, or promotion statistics.

## Accepted authority

```text
Stage 3C task:
    donch_bt_stage_3c_c02_alignment_adjudication_20260717_v1

agent-reported final commit:
    d370df829ea38f6cee0d45ece4fccbb91dafa3de

Drive transfer-manifest commit:
    110e09c9f6bf99452c60185fc44cbfe07fd69a4d

resolution-aware generator contract SHA-256:
    ce65c62edfb80f5fb83e9b8b6bae1d3eb9c981f8e9a1bcad3b285fdce46cca51

Stage 3B source contract SHA-256:
    25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb

spot manifest content hash:
    3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046

Stage 2C PF cohort hash:
    768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15
```

Before work, resolve the `d370df8...` versus `110e09c...` lineage from Git. Proceed only when one is an expected task/handoff descendant of the other and no divergent evidence exists.

Preserve:

```text
Stage 3B one-bar alignment:
    failed

resolved spot-led aggregate:
    mechanically sufficient

positive resolved spot-led:
    489 events
    96 / 152 / 241 in 2023 / 2024 / 2025
    86.9121% same leader under 30m
    mechanically sufficient

negative resolved spot-led:
    78.7879% agreement
    mechanically unavailable

resolved perp-led and completed failure:
    mechanically unavailable
```

No failed branch may enter the economic contract.

## Claim boundary

The proposed test is only:

> After a positive same-asset Kraken spot impulse is observed at least one non-overlapping five-minute interval before the Kraken PF follower impulse, does the PF continue upward after the causal decision time?

Do not claim general spot price discovery, negative-direction symmetry, survivorship-free coverage, continuous spot availability, or perp-led reversal evidence.

Use a distinct lineage:

```text
C02_positive_resolution_aware_spot_led_continuation_v1
```

## Frozen event sets

### Primary event set

All Stage 3C events satisfying:

```text
direction_label = positive
leadership_state = resolved_spot_led
primary lookback = 15m
```

Expected source count before economic eligibility/non-overlap:

```text
489
```

### Robustness event set

The primary events that also satisfy:

```text
leadership_30m = resolved_spot_led
```

Expected count:

```text
425
```

The robustness subset cannot rescue a failed primary definition.

Recompute and hash the event-set identities from the immutable Stage 3C tape. Do not regenerate thresholds or inspect future paths.

## Economic definitions to freeze

Use two fixed horizons:

```text
primary timeout:
    1 hour

longer-horizon robustness:
    6 hours
```

Definitions:

```text
c02_l3_primary_all_1h
c02_l3_primary_all_6h
c02_l3_30m_agreement_1h
c02_l3_30m_agreement_6h
```

The first two are primary definitions. The agreement-subset definitions are robustness-only.

No negative, perp-led, failure, shifted-clock, threshold, or alternate-horizon definition is allowed.

## Entry and exit

For every event:

```text
decision time:
    close/availability time of the completed Stage 3C onset bar

entry:
    next executable Kraken PF five-minute trade-bar open after decision

exit:
    first executable PF five-minute trade-bar open at or after
    entry + the definition timeout
```

Use fixed notional exposure.

For Level 3, use timeout-only exits. Do not add a stop, trailing rule, partial exit, add, passive fill, or leverage choice. This isolates the short-horizon transmission question and avoids a new stop-search dimension.

Fail closed for:

- missing next-open entry or exit;
- non-finite prices;
- lifecycle-invalid intervals;
- duplicate economic address;
- path crossing `2026-01-01`;
- pre-2023 or non-Kraken rows;
- mixed protected files;
- any artificial sample-boundary close.

## Non-overlap

Run each definition independently.

Within symbol and definition:

1. process events chronologically;
2. enter only while flat;
3. ignore later events until the actual timeout exit executes;
4. do not combine definitions into a portfolio.

## Costs and funding

Use existing repository-authoritative assumptions:

```text
base:
    5 bps taker per side
    4 bps round-trip slippage

stress:
    10 bps taker per side
    12 bps round-trip slippage
```

Primary Level-3 gates use full-period base fee/slippage net fixed-notional bps, excluding funding.

Report funding separately:

1. fully exact-funded;
2. mixed;
3. fully imputed;
4. zero-boundary.

Mixed or imputed funding may be sensitivity only and cannot rescue a definition.

## Level-3 gates

A primary definition may earn permission for later controls only if all pass:

```text
executed trades >= 100
trades in each of 2023, 2024, 2025 >= 20
mean base-net bps > 0
median base-net bps > 0
canonical-episode bootstrap 95% CI lower bound >= -5 bps
maximum positive symbol aggregate / total net <= 25%
maximum positive episode aggregate / total net <= 10%
maximum positive-year aggregate / total positive-year aggregate <= 70%
stress-net mean >= -10 bps
```

Use 10,000 canonical-episode bootstrap resamples and deterministic seed `20260717`.

Undefined or non-positive concentration denominators fail closed.

These gates permit controls only. They do not imply validation or promotion.

If neither primary definition passes, C02 stops at Level 3. Robustness definitions cannot rescue it.

## Frozen Level-4 control contract

Pre-register but do not execute:

### Leadership control

For each primary event, select at most one positive `coincident_or_unresolved` C02 event:

```text
same PF symbol
same calendar year
spot z_15m within 0.5
perp z_15m within 0.5
prior-day PF liquidity rank within 10
lagged PF 24h volatility within 20%
outside the treated event's canonical episode
at least 24h from treated onset
nearest timestamp
tie break by timestamp
no caliper widening
```

### Leadership ablation

Use the same positive confirmed-impulse generator and entry/timeout rules without requiring resolved spot leadership.

### Measurement robustness

Report the frozen 30-minute-agreement subset. It cannot substitute for the primary set after outcomes.

Controls run only for a primary definition that passes every Level-3 gate and require separate human approval.

## Required outputs

```text
C02_POSITIVE_SPOT_LED_EVENT_SET.csv
C02_LEVEL3_DEFINITION_REGISTER.csv
C02_FINAL_LEVEL3_ECONOMIC_CONTRACT.md
C02_LEVEL3_DECISION_RULES.json
C02_LEVEL4_CONTROL_CONTRACT.md
C02_PRERUN_APPROVAL_PACKET.md
C02_MULTIPLICITY_AND_LINEAGE_RECORD.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

The approval packet must contain:

- final contract hash;
- all immutable input hashes;
- exact later runner command interface;
- expected fresh result-root pattern;
- rollback;
- forbidden actions.

Do not implement or execute the economic runner.

## Tests

Use synthetic fixtures for:

- exact primary and robustness event membership;
- four definitions retained;
- primary/robustness non-rescue;
- next-open entry and timeout exit;
- actual-exit non-overlap;
- protected/sample-boundary failure;
- fixed-notional base/stress arithmetic;
- funding partitions;
- bootstrap and concentration gates;
- control calipers never widening;
- no economic or protected fields read during this task.

Run relevant Stage 3A–3C, loader, lifecycle, identity, protected-boundary, and archive guards. Require independent review.

## Final decision

Return exactly one:

```text
ready_for_human_C02_Level3_run_approval
```

or:

```text
blocked_with_exact_non_economic_remedy
```

No economic run is authorized.

## Integration and handoff

After tests and review:

- create task-scoped commit(s);
- non-force push under the standing reviewed-task workflow;
- upload using `drive_handoff: approved_default`;
- verify remote size and SHA-256;
- keep local archive;
- omit raw market data and large Parquet tapes from Drive.

## Final response

```text
status:
actual_starting_commit:
lineage_discrepancy_resolution:
final_contract_hash:
primary_event_count:
robustness_event_count:
definition_count:
Level3_gates:
Level4_control_contract:
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
