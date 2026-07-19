# Stage 9 — KDA02 Purge-State Adjudication and Conditional Level-3 Execution

```text
task_id: donch_bt_stage_9_kda02_purge_adjudication_conditional_level3_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
non_economic_adjudication_authorized: yes
conditional_economic_run_authorized: yes — exact KDA02 v2 contract only, after frozen pre-outcome review and mechanical gates
controls_authorized: no
protected_outcome_access: no
new_data_acquisition: no
Capitalcom_payload_access: no
commit_authorized: yes — task-scoped reviewed commits
push_authorized: yes — non-force under the standing workflow
```

## Human decision

KDA01 v2 is closed at its exact repaired steady-state translation:

```text
terminal decision:
    KDA01_level3_repaired_no_primary_pass_stop

Stage 8C1 commit:
    baaa10c224807e1dc7e32bfee7227711cb0c1279

Stage 8C1 run manifest SHA-256:
    2e8d8cce4ed5507bbac11b86d98158e7bcfc3ce60905c82c1cc22224636eff05
```

Do not perform the proposed promotion-fee study. Do not reopen KDA01 through costs, controls, thresholds, directions, horizons, subsets, basis change, or another timing variant.

The next mechanism is KDA02: actual aggregate liquidation intensity plus material OI reset, separating active forced-flow continuation from completed purge reversal.

Sending this task constitutes conditional human approval for one Level-3 economic run only if the outcome-free KDA02 v2 contract passes every mechanical gate and receives independent pre-run approval before outcome access.

## Objective

1. Adjudicate the broad Stage 8A KDA02 generator without changing its provenance.
2. Build a new episode-level KDA02 v2 translation that genuinely requires liquidation and OI reset.
3. Separate the liquidation-free OI-vacuum idea into a different inactive lineage.
4. Freeze the complete 16-definition KDA02 v2 Level-3 contract, controls, costs, clustering, and gates before returns.
5. If all primary branches pass the mechanical gates and independent pre-run review approves, execute the exact frozen contract once.
6. Return a terminal KDA02 Level-3 decision.

No parameter search, outcome-conditioned adaptation, KDA01 rescue, control execution, or protected data is permitted.

## Current authority to verify

```text
expected starting commit:
    baaa10c224807e1dc7e32bfee7227711cb0c1279

Stage 8A published commit:
    41b64b52a9146669eb26dcf25a86523a35219b8d

Stage 8A task archive:
    docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1

Stage 8A KDA02 event tape SHA-256:
    c4d553267e2107beeb042bc22f3280013c41a962d1674b4942d2b7f0de5e2b43

Stage 8A semantic contract hash:
    289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60

Stage 8A shared feature contract hash:
    4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4

Stage 8A generator hash:
    c285af918e317722e91e3c1df5e44eb48abf27c97fedfe251b0881e3aa8f0017

analytics data-manifest hash:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

cohort hash:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636

feature/event cache:
    /opt/parquet/kraken_derivatives/analytics/stage8a_foundation_v1_exact

rankable interval:
    [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
```

Verify repository root, applicable `AGENTS.md`, branch/commit/remotes, clean worktree, source/data hashes, supported commands, local tapes, archive convention, Drive target, and protected boundary before changes.

## Archive context and prohibitions

Preserve unchanged:

```text
C01: level3_no_primary_pass_stop
C02: level3_no_primary_pass_stop
C03: C03_PIT_authority_unavailable
C16: C16_flow_authority_unavailable
KDA01: KDA01_level3_repaired_no_primary_pass_stop
```

No same-sample rescue of C01, C02, RFBS, Backside, failed-breakdown, delayed-flush, LFBS, H43, TSMOM, A1, or KDA01.

The new KDA02 v2 mechanism is allowed because actual historical liquidation and OI data were unavailable to the older chart-only delayed-flush translation. The old translation remains unchanged.

## Stage 8A KDA02 adjudication

Document the exact Stage 8A definitions and why they are not economic-ready:

```text
primary active purge:
    liquidation intensity robust z >= 2
    and any negative one-hour OI change

primary completed purge:
    previous-bar liquidation extreme
    current OI decline
    current five-minute trade-return sign aligned with one-hour mark return

primary OI vacuum:
    any OI decline
    modest one-hour price displacement
    no liquidation requirement
```

Stage 8A counts to reconcile:

```text
primary active purge: 21,241
primary completed purge: 43,946
primary OI vacuum: 1,176,354
robust active purge: 3,089
robust completed purge: 7,602
robust OI vacuum: killed semantic duplicate, zero tape
```

These are valid feasibility attempts, not code defects. Preserve their outputs and attempt identities.

Create two distinct lineages:

```text
KDA02A_v2_liquidation_oi_purge_state_machine
KDA02B_oi_vacuum_without_liquidation_candidate_library
```

KDA02B receives no economic contract or outcome access in this task.

## KDA02 v2 feature extension

Use completed five-minute Kraken trade, mark, OI, and liquidation rows.

Add these causal fields if absent:

```text
trade_return_15m
mark_return_15m
liquidation_base_units_15m
liquidation_to_lagged_oi_15m
liquidation_intensity_15m_robust_z
liquidation_intensity_15m_percentile
oi_log_change_15m
oi_change_15m_robust_z
oi_change_15m_percentile
price_displacement_15m_robust_z
price_displacement_15m_percentile
```

All normalization must use data through the prior UTC day only:

```text
lookback: 60 calendar days
minimum valid days: 30
minimum expected valid observations: 70%
median/MAD robust z
empirical percentile
zero or non-finite scale: fail closed
```

Future or same-day later rows must not change prior features.

Primary cross-symbol liquidation intensity is:

```text
sum of liquidation base units over trailing completed 15 minutes
/
OI close immediately before that 15-minute window
```

The unit cancels under the inferred common base-unit semantics. Preserve raw fields and the inferred-semantic status.

## Frozen parent purge state

For each completed five-minute row define:

```text
d = sign(trade_return_15m)
```

A primary parent purge begins only when:

```text
d != 0
sign(mark_return_15m) = d
liquidation_intensity_15m_robust_z >= 2
oi_change_15m_robust_z <= -2
price_displacement_15m_robust_z >= 1
all exact contiguous 15-minute windows and causal normalizations are valid
```

Robustness parent:

```text
liquidation_intensity_15m_percentile >= 0.95
oi_change_15m_percentile <= 0.05
price_displacement_15m_percentile >= 0.75
same trade/mark direction and validity requirements
```

Do not weaken these boundaries after counts.

Interpret direction only as:

```text
d < 0: long_liquidation_proxy
d > 0: short_liquidation_proxy
```

This remains a price-inferred proxy. Do not claim native long/short liquidation labels.

## Purge episode contract

For each symbol, attempt, and direction:

1. Start at the first qualifying parent row after at least 60 consecutive minutes without that directional parent state.
2. Freeze the onset five-minute trade and mark open/high/low/close.
3. Freeze the initial impulse window as the first three completed bars beginning at onset.
4. Continue the purge episode while either of these hysteresis conditions remains true:

```text
liquidation intensity robust z >= 1
oi change robust z <= -1
```

For robustness use:

```text
liquidation percentile >= 0.75
oi-change percentile <= 0.25
```

5. End after 15 consecutive minutes outside both hysteresis conditions or after six hours, whichever occurs first.
6. Preserve every episode, including episodes with no candidate.
7. Emit at most one active-continuation candidate and one completed-reversal candidate per episode.

Primary and robustness attempts remain separate.

## Active-purge continuation branch

Within 60 minutes of parent onset, while the episode remains active, confirm at the first completed bar satisfying:

```text
d < 0:
    trade close < initial impulse trade low
    and mark close < initial impulse mark low

d > 0:
    trade close > initial impulse trade high
    and mark close > initial impulse mark high
```

Also require at confirmation:

```text
liquidation intensity remains above the attempt's hysteresis threshold
OI remains below its pre-onset level
```

Decision time is the completed confirmation bar availability time.

Trade direction:

```text
d < 0 -> short
d > 0 -> long
```

A touch, wick, trade-only break, or mark-only break is insufficient.

## Completed-purge reversal branch

A reversal candidate requires all of:

1. The episode previously satisfied the parent purge state.
2. Cumulative OI reduction from the pre-onset OI close remains material under the parent attempt.
3. Liquidation intensity has remained below the attempt's hysteresis threshold for at least 15 consecutive minutes.
4. Within six hours of parent onset, the first completed bar satisfies:

```text
d < 0:
    trade close > onset trade open
    and mark close > onset mark open

d > 0:
    trade close < onset trade open
    and mark close < onset mark open
```

Decision time is the completed structural reclaim/failure bar availability time.

Trade direction:

```text
d < 0 -> long
d > 0 -> short
```

Do not create an anticipatory reversal branch.

## Candidate branches

Primary definitions before horizons:

```text
negative active purge continuation -> short
positive active purge continuation -> long
negative completed purge reversal -> long
positive completed purge reversal -> short
```

Create the same four robustness-only branches.

## Timestamp and execution contract

Use the repaired repository convention established by Stage 8C1:

```text
stored candle timestamp:
    interval start

decision_ts:
    completed signal-bar start + 5 minutes

entry:
    first authorized PF 5m trade-bar open at or after decision_ts

maximum entry delay:
    10 minutes after expected entry

exit target:
    actual entry + frozen timeout

exit:
    first authorized PF 5m trade-bar open at or after exit target

maximum exit delay:
    10 minutes
```

No additional five-minute delay is permitted. Add regressions covering this exact issue.

Use definition-local, symbol-local actual-exit non-overlap.

## Market-wide inference identity

Before outcomes add:

```text
market_day_cluster_id:
    hash(translation_id, attempt, UTC parent-onset date)

market_6h_cluster_id:
    hash(translation_id, attempt, fixed UTC six-hour parent-onset block)
```

Primary inference uses equal-weight market-day clusters. Six-hour and symbol-specific episode clusters are sensitivity-only and cannot rescue primary failure.

## Mechanical feasibility gates

Each of the four primary branches must have after timestamp eligibility and non-overlap:

```text
candidate events >= 100
events in each of 2023, 2024, 2025 >= 20
symbols >= 20
maximum symbol event share <= 25%
duplicate event IDs/economic addresses = 0
protected rows = 0
```

A failed branch remains registered and receives no economic definition. Do not adapt it.

If fewer than two primary branches remain feasible, stop before outcomes with:

```text
KDA02_v2_mechanically_unavailable
```

## Conditional frozen Level-3 contract

If mechanical gates pass and independent pre-run review approves, freeze and hash:

```text
timeouts: 1h and 6h
fixed notional
entry/exit as above
no stop
no target
no add
no partial exit
base all-in round-trip cost: 14 bps
stress all-in round-trip cost: 32 bps
funding partitions diagnostic and excluded from gates
bootstrap: 10,000 market-day resamples
seed: 20260719
```

The expected maximum is:

```text
8 primary definitions
8 robustness-only definitions
```

If a primary branch failed feasibility, omit its two horizons rather than replacing it.

## Level-3 economic calculation

Use only official rankable PF five-minute trade-bar opens at frozen entry and exit timestamps.

```text
gross_bps = side * (exit_open / entry_open - 1) * 10,000
base_net_bps = gross_bps - 14
stress_net_bps = gross_bps - 32
```

Mark confirms the mechanism but is not a fill price.

For each definition independently:

1. Equal-weight trades within each `market_day_cluster_id`.
2. Equal-weight the resulting market-day returns.
3. Compute base/stress mean and median.
4. Bootstrap market-day means with replacement, 10,000 times, seed 20260719.
5. Compute contribution concentration by market day, symbol, and year.

No pooling across directions, branches, horizons, or attempts.

## Frozen Level-3 gates

Every primary definition must independently pass all:

```text
executed trades >= 100
trades in each year >= 20
equal-market-day base mean > 0 bps
equal-market-day base median > 0 bps
market-day bootstrap 95% lower bound >= -5 bps
maximum positive market-day contribution <= 10%
maximum positive-symbol contribution <= 25%
maximum positive-year contribution <= 70%
equal-market-day stress mean >= -10 bps
```

Robustness cannot rescue primary.

## Frozen controls — do not execute

Register and freeze:

1. same price/mark displacement and structural break without liquidation or OI requirements;
2. liquidation extreme without material OI reset;
3. material OI reset without liquidation extreme;
4. ordinary high-volume shock with matched displacement;
5. structural continuation/reversal without analytics confirmation;
6. BTC/ETH market-wide stress-day control;
7. Stage 8A KDA02 overlap/ablation;
8. KDA02B OI-vacuum lineage as a separate non-liquidation mechanism.

No control outcome is authorized in this task.

## Terminal decision

Return:

```text
KDA02_level3_primary_pass_controls_required
```

when at least one primary definition passes every gate. Freeze all passing definitions; do not select only the best.

Return:

```text
KDA02_level3_no_primary_pass_stop
```

when no primary definition passes. Close the exact KDA02 v2 translation without threshold, direction, cost, horizon, symbol, year, session, or OI-vacuum rescue.

Return:

```text
KDA02_v2_mechanically_unavailable
```

when the outcome-free branches fail the predeclared feasibility rule.

Return `blocked_with_exact_mechanical_remedy` only for verified authority, code, data, or reproducibility failure.

## Required outputs

### Always before outcomes

```text
KDA02_STAGE8A_ADJUDICATION.md
KDA02_V2_FEATURE_EXTENSION_CONTRACT.md
KDA02_V2_PARENT_EPISODE_TAPE.parquet
KDA02_V2_EVENT_TAPE.parquet
KDA02_V2_COUNT_MATRIX.csv
KDA02_V2_FEASIBILITY_GATES.csv
KDA02_V2_ATTEMPT_REGISTER.csv
KDA02B_OI_VACUUM_LINEAGE_DECISION.md
KDA02_V2_MARKET_CLUSTER_SUMMARY.csv
KDA02_FINAL_LEVEL3_CONTRACT.json
KDA02_LEVEL3_DEFINITION_REGISTER.csv
KDA02_LEVEL4_CONTROL_CONTRACT.md
KDA02_PRERUN_REVIEW.md
```

### If conditional economics execute

```text
KDA02_LEVEL3_ACCEPTED_EXECUTIONS.parquet
KDA02_LEVEL3_TRADE_TAPE.parquet
KDA02_LEVEL3_DEFINITION_METRICS.csv
KDA02_LEVEL3_GATE_MATRIX.csv
KDA02_LEVEL3_MARKET_DAY_RETURNS.parquet
KDA02_LEVEL3_BOOTSTRAP_SUMMARY.csv
KDA02_LEVEL3_CONCENTRATION.csv
KDA02_LEVEL3_FUNDING_PARTITIONS.csv
KDA02_LEVEL3_CLUSTER_SENSITIVITY.csv
KDA02_LEVEL3_DEFINITION_DECISIONS.csv
KDA02_LEVEL3_DECISION.md
KDA02_LEVEL3_CLAIM_BOUNDARY.md
```

### Always

```text
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large Parquet and bootstrap artifacts remain local and hash-manifested. Upload compact complete evidence and manifest.

## Tests and independent review

Test at minimum:

1. Stage 8A KDA02 counts and hash reconciliation;
2. causal 15-minute feature construction;
3. exact contiguous-window failure;
4. future/same-day later invariance;
5. liquidation/OI unit-cancelling ratio;
6. price-inferred side remains a proxy;
7. parent onset, reset, hysteresis, and six-hour cap;
8. one continuation and one reversal per episode;
9. continuation requires trade+mark extreme break;
10. reversal requires liquidation cooldown and trade+mark onset-open reclaim/failure;
11. OI-vacuum branch cannot enter KDA02A;
12. corrected entry is at or after decision time, with no extra bar delay;
13. timestamp availability and actual-exit non-overlap;
14. market-day cluster identity;
15. long/short return and cost arithmetic;
16. bootstrap and concentration;
17. primary/robustness separation;
18. funding excluded from gates;
19. no protected, Capital.com, control, KDA01, or other-family outcome;
20. deterministic replay and manifest hashes.

Require two independent reviews:

```text
pre-outcome review:
    definitions, counts, contract, timestamp semantics, controls, hashes

post-run review if economics execute:
    prices, arithmetic, clustering, bootstrap, gates, claims, artifacts
```

## Integration and handoff

After terminal review:

- update factual KDA02 family, attempt, source, run, evidence, multiplicity, and continuity records;
- preserve KDA01 and all prior roots and findings;
- create task-scoped reviewed commits;
- non-force push under the standing workflow;
- use `drive_handoff: approved_default`;
- round-trip verify all compact uploads by size and SHA-256;
- retain full local data and task archive.

## Stop conditions

Stop and preserve evidence if:

- any source/hash/contract identity fails;
- timestamp semantics regress;
- a protected row may be opened;
- the OI-vacuum branch contaminates the liquidation family;
- an outcome-dependent change is proposed;
- controls, KDA01, KDA03, Capital.com, or capture enter scope;
- Git state is unsafe;
- deterministic replay fails.

## Final response

```text
status:
actual_starting_commit:
KDA01_closure_preserved:
Stage8A_KDA02_adjudication:
KDA02_v2_feature_contract_hash:
KDA02_v2_generator_hash:
parent_episode_counts:
branch_counts_by_year_direction_symbol:
market_day_cluster_counts:
mechanical_feasibility_gates:
frozen_Level3_contract_hash:
definition_count:
conditional_economics_executed:
primary_definition_results:
robustness_definition_results:
primary_pass_count:
passing_primary_definitions:
controls_executed: no
KDA02B_outcomes_executed: no
protected_rows_opened: no
tests_and_reviews:
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
