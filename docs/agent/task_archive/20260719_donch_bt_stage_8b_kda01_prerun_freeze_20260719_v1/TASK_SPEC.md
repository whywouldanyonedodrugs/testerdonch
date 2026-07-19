# Stage 8B — KDA01 Mechanism Adjudication and Pre-Run Freeze

```text
task_id: donch_bt_stage_8b_kda01_prerun_freeze_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
new_data_acquisition: no
commit_authorized: yes — task-scoped reviewed commits
push_authorized: yes — non-force under the standing workflow
```

## Objective

Replace the mechanically broad Stage 8A KDA01 event translation with one episode-level, mechanism-faithful, outcome-free translation and freeze an approval-ready Level-3 contract if its predeclared feasibility gates pass.

Do not calculate forward returns, entries/exits against real paths, PnL, MAE/MFE, economic rankings, control performance, or protected outcomes.

KDA02 and KDA03 remain unchanged and receive no economic contract in this task.

## Authority to verify

```text
Stage 8A published commit:
    41b64b52a9146669eb26dcf25a86523a35219b8d

Stage 8A task archive:
    docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1

semantic contract hash:
    289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60

analytics manifest hash:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

cohort hash:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636

shared feature contract hash:
    4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4

Stage 8A generator hash:
    c285af918e317722e91e3c1df5e44eb48abf27c97fedfe251b0881e3aa8f0017

Stage 8A KDA01 tape SHA-256:
    583c1f940f185cf01417a1f5ba6540c6a1b6545c0532851d1dabf200d8c874ce

feature/event cache:
    /opt/parquet/kraken_derivatives/analytics/stage8a_foundation_v1_exact
```

Verify repository identity, `AGENTS.md` chain, clean synchronized Git state, commands, data roots, task archive convention, and Drive target before changes.

## Archive context

Preserve unchanged:

```text
C01: level3_no_primary_pass_stop
C02: level3_no_primary_pass_stop
C03: C03_PIT_authority_unavailable
C16: C16_flow_authority_unavailable
```

The Stage 8A KDA01 tape is valid outcome-free provenance but is not automatically economic-ready. It contains 2,639,115 attempt-events across six definitions and collapses with KDA02 into 450,408 canonical episodes. It also overlaps 2,321 of 4,887 C01 anchors and 11,869 of 32,686 C02 impulses. Preserve those facts.

Do not use analytics to rescue or retune C01, C02, RFBS, Backside, failed-breakdown, LFBS, H43, TSMOM, A1, or another closed translation.

## Required adjudication

Document explicitly that the Stage 8A KDA01 generator used:

- any positive 1h OI change rather than a material OI-expansion state;
- an extreme absolute basis state without requiring directional coherence in the parent mask;
- sign-based failure confirmation rather than a structural close-through;
- threshold re-entry onsets that can create many members inside one economic episode.

These are not code defects. They are mechanical feasibility definitions that must be narrowed before economics.

Do not overwrite or relabel Stage 8A outputs. Create a new translation:

```text
KDA01_v2_episode_level_crowding_price_progress_bifurcation
```

and register every new attempt in the same multiplicity family.

## Frozen KDA01 v2 parent state

Use completed five-minute rows and existing exact trade/mark/analytics intersections.

For every row, define parent direction:

```text
d = sign(trade_return_1h)
```

A row may enter the parent state only when:

```text
d != 0
sign(mark_return_1h) = d
OI expansion is materially positive
basis is directionally coherent with d
basis is materially extreme
all causal normalization and coverage fields are valid
```

Add causal prior-day 60-calendar-day normalization for `oi_log_change_1h` if not already cached:

```text
minimum valid days: 30
minimum expected coverage: 70%
primary OI expansion: oi_change_robust_z >= 2
robustness OI expansion: oi_change_percentile >= 0.95
```

Directional basis rules:

```text
primary:
    d * basis_decimal > 0
    d * basis_level_robust_z >= 2

robustness:
    positive d: basis_decimal > 0 and basis_level_percentile >= 0.95
    negative d: basis_decimal < 0 and basis_level_percentile <= 0.05
```

Do not weaken these thresholds after counts.

## Price-progress classification

Create:

```text
price_progress_per_oi = abs(trade_return_1h) / oi_log_change_1h
```

only where the denominator is finite and positive under the parent state. Normalize this field causally by the same prior-day method.

At parent onset classify:

```text
efficient_progress:
    price_progress_percentile >= 0.75
    path_efficiency_1h >= 0.50

deteriorating_progress:
    price_progress_percentile <= 0.25
    or path_efficiency_1h <= 0.25

intermediate:
    all other parent onsets
```

Intermediate and deteriorating-onset rows are diagnostics. They are not anticipatory reversal entries.

## Parent episodes and reset

For each symbol and direction:

1. Start an episode at the first qualifying parent row after at least 60 consecutive minutes without that directional parent state.
2. Retain the episode while both OI and directional basis remain above their relaxed hysteresis boundaries:

```text
oi_change_robust_z >= 1
and d * basis_level_robust_z >= 1
```

3. End after 30 consecutive minutes outside the hysteresis state or after six hours, whichever comes first.
4. Emit at most one efficient-continuation event and at most one completed-failure event per parent episode.
5. Preserve all parent episodes, including those producing no economic candidate.

Primary and robustness parent episodes remain separate attempts and cannot be selected from counts.

## Structural completed failure

A failure candidate may occur only after a parent episode has entered `deteriorating_progress` at least once.

Freeze the initial impulse window as the first 60 minutes of the parent episode. Store its trade and mark highs/lows.

Within six hours of parent onset, confirm at the first completed five-minute bar satisfying:

```text
positive parent d:
    trade close < initial impulse trade low
    and mark close < initial impulse mark low

negative parent d:
    trade close > initial impulse trade high
    and mark close > initial impulse mark high
```

Decision time is the availability time of that completed confirmation bar. A wick, touch, sign flip, or trade-only close is insufficient.

The eventual trade direction is opposite the parent direction.

## Candidate branches

Retain exactly these primary mechanism branches, each split by parent direction:

```text
positive efficient crowding continuation -> long
negative efficient crowding continuation -> short
positive crowding completed failure -> short
negative crowding completed failure -> long
```

Repeat the same four branches under the robustness parent threshold as robustness-only attempts.

Do not add an anticipatory deteriorating-price fade.

## Pre-economic feasibility gates

Before freezing the economic contract, each primary branch must have:

```text
at least 100 candidate events total
at least 20 events in each of 2023, 2024, and 2025
at least 20 symbols
maximum event-count symbol share <= 25%
duplicate event IDs/economic addresses = 0
protected rows = 0
```

A failed branch remains registered and is excluded from the economic approval packet. Do not weaken the state to rescue it.

## Frozen Level-3 contract if feasible

For every feasible primary branch freeze two timeout definitions:

```text
1 hour
6 hours
```

For every corresponding robustness branch freeze the same timeouts as robustness-only definitions.

Execution interface:

```text
decision:
    parent onset for efficient continuation;
    structural failure confirmation for failure reversal

entry:
    first executable Kraken PF five-minute trade-bar open strictly after decision

exit:
    first executable trade-bar open at or after entry + timeout

position:
    fixed notional

exit type:
    timeout only
```

No stop, add, partial exit, leverage selection, passive-fill assumption, or portfolio combination.

Run each definition independently with symbol-level actual-exit non-overlap.

Costs:

```text
base: 14 bps round trip
stress: 32 bps round trip
funding: separate exact/mixed/imputed/zero partitions; excluded from Level-3 gates
```

Level-3 gates for primary definitions:

```text
executed trades >= 100
trades per year >= 20
base-net mean > 0
base-net median > 0
canonical-episode bootstrap 95% lower bound >= -5 bps
maximum positive-symbol contribution / total net <= 25%
maximum positive-episode contribution / total net <= 10%
maximum positive-year contribution / positive-year total <= 70%
stress-net mean >= -10 bps
```

Use 10,000 canonical-episode resamples and seed `20260719`. Undefined or non-positive concentration denominators fail closed. Robustness definitions cannot rescue a primary failure.

Do not implement or execute the economic runner.

## Frozen controls for later separate approval

Pre-register but do not execute:

1. same price-progress/path branch without OI or basis requirements;
2. material OI expansion without basis requirement;
3. directional basis extreme without OI requirement;
4. completed structural failure after a price-only parent state;
5. matched parent episodes with ordinary OI and basis;
6. BTC/ETH parent-state control;
7. KDA01 v1 overlap/ablation, never as a rescue.

No caliper widening after outcomes.

## Required outputs

```text
KDA01_V1_ADJUDICATION.md
KDA01_V2_FEATURE_EXTENSION_CONTRACT.md
KDA01_V2_PARENT_EPISODE_TAPE.parquet
KDA01_V2_EVENT_TAPE.parquet
KDA01_V2_COUNT_MATRIX.csv
KDA01_V2_FEASIBILITY_GATES.csv
KDA01_V2_ATTEMPT_REGISTER.csv
KDA01_V2_CANONICAL_EPISODE_REPORT.md
KDA01_V2_OLD_FAMILY_OVERLAP.csv
KDA01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md
KDA01_LEVEL3_DEFINITION_REGISTER.csv
KDA01_LEVEL3_DECISION_RULES.json
KDA01_LEVEL4_CONTROL_CONTRACT.md
KDA01_PRERUN_APPROVAL_PACKET.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

If no primary branch passes the mechanical feasibility gates, omit the approval packet and return the exact blocker.

Large Parquet outputs remain local and hash-manifested.

## Tests and review

Test at minimum:

- causal OI and price-progress normalization;
- future rows cannot change prior states;
- directional basis coherence;
- exact threshold boundaries;
- episode reset and hysteresis;
- one event per episode/branch;
- structural trade+mark close-through;
- no touch-only or sign-flip failure;
- six-hour confirmation deadline;
- primary/robustness separation and no rescue;
- deterministic IDs, episodes, manifests, and replay;
- no pre-2023, 2026+, lifecycle-invalid, non-Kraken, outcome, or protected rows;
- bounded-memory partitioned execution.

Run focused tests and relevant Stage 8A, loader, protected-boundary, lifecycle, identity, archive, C01/C02, and analytics regressions.

Require independent review of the actual definitions, code, event counts, episode reduction, contract, controls, hashes, and claims.

## Decision

Return exactly one:

```text
ready_for_human_KDA01_Level3_run_approval
KDA01_mechanically_unavailable
blocked_with_exact_non_economic_remedy
```

No economic run is authorized.

## Final response

```text
status:
actual_starting_commit:
KDA01_v1_adjudication:
KDA01_v2_feature_contract_hash:
KDA01_v2_generator_hash:
parent_episode_counts:
branch_counts_by_year_direction_symbol:
mechanical_feasibility_gates:
final_Level3_contract_hash:
definition_count:
controls_frozen:
protected_rows_opened: no
economic_outputs_computed: no
tests_and_review:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
next_status:
human_approval_required:
```
