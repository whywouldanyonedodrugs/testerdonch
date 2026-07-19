# Stage 8C1 — KDA01 Forensic Execution Audit and Timestamp Repair

```text
task_id: donch_bt_stage_8c1_kda01_forensic_timing_audit_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
forensic_outcome_access_authorized: yes — existing KDA01 Stage 8C outputs only
conditional_repaired_economic_rerun_authorized: yes — exact 16 frozen definitions only, only if the timestamp defect is independently confirmed before repaired outcomes are opened
controls_authorized: no
reverse_direction_authorized: no
new_thresholds_horizons_subsets: no
protected_outcome_access: no
new_data_acquisition: no
Capitalcom_payload_access: no
commit_authorized: yes — task-scoped reviewed commits
push_authorized: yes — non-force under the standing workflow
```

## Objective

Determine whether Stage 8C implemented an unnecessary extra five-minute delay between causal signal availability and entry. If and only if the defect is confirmed from timestamp semantics and code lineage without using repaired outcomes, make the smallest repair and rerun the exact frozen KDA01 v2 Level-3 contract once.

Also produce a forensic explanation of why the reported losses occurred, separating:

```text
implementation error
definition behavior
cost insufficiency
market-day weighting
```

Do not rescue KDA01 through direction reversal, basis-change substitution, threshold changes, horizons, stops, symbols, years, sessions, controls, or another family.

## Authority to verify

```text
expected starting commit:
    fa919770c13c9268dc67c65647f207ee3faf1e0c

Stage 8C run root:
    results/rebaseline/phase_kraken_kda01_v2_level3_economic_20260719_v1_20260719_121215

Stage 8C terminal decision:
    KDA01_level3_no_primary_pass_stop

Stage 8C run-manifest SHA-256:
    fba91095e84ddf78eb1e218bf11e4542233ea9b974687b85dac6e99d2e54f1c4

Stage 8C Drive ZIP SHA-256:
    756698431ff641fbbfbfc6c9b5e1c07f1d21801e584af1d7dec7fe5f0a562246

frozen Level-3 contract hash:
    d2112d8fb4f0666890761b70f7fa71c9e8cb4f43c704a04cdf9470e949a177b3

Stage 8B1 event-cluster tape SHA-256:
    1a2946a2da8c5bf9b1a4cbc9571abad31d968034f246fe94a8e442e959864669

Stage 8B event tape SHA-256:
    7c9e682380681aa6fc83161ae29dafab9a047bdb19fe1fa8ade3ba5a4eb9c2e5

Stage 8B parent tape SHA-256:
    ae470d06c7da049d145ac8e083d0c4711c9c2dd6e4ec32fedf79224e8885ddbd

analytics manifest hash:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

rankable interval:
    [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
```

Verify repository root, `AGENTS.md`, branch/commit/remotes, worktree, source/data hashes, task archives, commands, Drive target, and protected boundary before reading any outcome or price data.

Preserve C01/C02/C03/C16 and every earlier negative decision unchanged.

## Frozen facts from Stage 8C to reconcile

Before any repair, reproduce from the existing immutable Stage 8C outputs:

```text
schedule records: 204,272
accepted trades: 183,744
rejected schedules: 20,528
actual-overlap rejections: 20,473
missing-exit rejections: 55
price rejections: 0
primary passes: 0/8
controls executed: 0
protected rows: 0
```

Recompute every Stage 8C trade return and all 16 metric/gate rows from the existing local trade tape using an independent implementation. Require byte- or value-exact reconciliation within declared floating precision.

## Mandatory forensic decomposition

Create one table for all 16 definitions containing:

```text
trade-level gross mean and median
equal-market-day gross mean and median
base and stress means/medians
bootstrap interval
branch side and timeout
```

Explicitly report these distinctions:

- completed-failure branches were not uniformly gross-negative;
- continuation branches were materially worse under equal-market-day inference;
- whether costs alone changed any gross-positive definition into a base-negative definition;
- whether the result is robust at trade, market-day, six-hour-cluster, and parent-episode levels.

This is diagnosis only. No diagnostic statistic can promote or rescue a definition.

## Timestamp-semantics audit

Audit the complete lineage for the five-minute PF trade and mark bars:

```text
raw endpoint/schema snapshot
acquisition manifest and interval semantics
parquet `time` convention
Stage 8A feature timestamp convention
Stage 8B event decision convention
Stage 8B1 execution lookup convention
Stage 8C price lookup convention
```

The current lineage appears to do both:

```text
Stage 8B:
    decision_ts = signal_bar_timestamp + 5 minutes

Stage 8B1/8C:
    entry = first bar timestamp strictly after decision_ts
```

Determine whether this creates an unnecessary additional full five-minute delay.

The defect is confirmed only if source and repository authority establish that:

1. the stored candle timestamp identifies the five-minute interval start; and
2. `decision_ts = bar_start + 5 minutes` is the causal availability time of the completed signal bar; and
3. the next bar opens at exactly `decision_ts`.

Do not use return differences to decide whether the defect exists.

If timestamp authority remains ambiguous, do not rerun. Return the exact missing authority.

## Conditional surgical repair

If the defect is independently confirmed, create a new execution-contract version while leaving Stage 8A/8B/8B1 event generation unchanged:

```text
contract version:
    kda01_level3_contract_v3_timestamp_repair_20260719

entry:
    first authorized PF 5m trade-bar open at or after decision_ts

expected entry:
    decision_ts when exactly aligned to the 5m grid;
    otherwise the first 5m grid timestamp after decision_ts

maximum entry delay:
    10 minutes after expected entry

exit target:
    repaired actual entry + unchanged frozen 1h or 6h timeout

exit:
    first PF 5m trade-bar open at or after exit target

maximum exit delay:
    unchanged 10 minutes
```

Do not alter:

```text
102,136 source events
16 definitions
8 primary / 8 robustness separation
branch directions
OI/basis/progress definitions
parent episodes
market-day clusters
costs 14/32 bps
funding separation
fixed notional
non-overlap policy
bootstrap seed/resamples
gates
```

Because the repaired entry changes actual exits and overlap, rebuild scheduling deterministically and report every count delta versus Stage 8C.

## Conditional repaired economic execution

Only after independent pre-run review approves the timestamp finding and exact repair diff may the agent open repaired entry/exit prices and execute one deterministic repaired run.

Use the same return, cost, clustering, concentration, funding, gate, and terminal-decision methods as Stage 8C.

Terminal repaired decision:

```text
KDA01_level3_repaired_primary_pass_controls_required
KDA01_level3_repaired_no_primary_pass_stop
```

At least one primary definition must pass every unchanged gate to receive the first status. Robustness cannot rescue primary.

## Required non-decision diagnostics

Report but do not use for promotion:

### Timing delta

For every original accepted trade that can be paired to a repaired trade:

```text
entry shift minutes
exit shift minutes
original gross bps
repaired gross bps
gross delta bps
```

Summarize by branch, timeout, year, symbol, market day, and event-count decile.

### Inference-estimand explanation

Explain the difference between:

```text
trade-weighted mean
mean of equal-weight market-day means
```

As a diagnostic only, perform a market-day cluster bootstrap of the trade-weighted mean by resampling complete market days and preserving all trades inside each sampled day. This statistic cannot replace the frozen equal-day gates or rescue the result.

### Sign symmetry

Verify mechanically that opposite-side gross equals the negative of frozen-side gross for every trade. Do not report opposite-side performance as a candidate, decision, or next strategy.

## Definition audit without adaptation

Document, without changing definitions:

- `efficient_progress` is based on a trailing one-hour move already completed at parent onset;
- basis uses an extreme directional level, not basis change;
- completed failure waits for a complete first-hour impulse and trade+mark close-through;
- these choices may detect mature/exhausted states rather than early continuation;
- changing any of them would create a new multiplicity attempt, not repair Stage 8C.

Do not implement KDA01 v3 feature logic in this task.

## Prohibited actions

- no reverse-direction economic run;
- no alternative entry delay besides the single causally correct repair;
- no alternative costs;
- no control execution;
- no basis-change or OI-threshold variant;
- no 2024-only, 2025-only, symbol, liquidity, session, or outlier subset;
- no stop, target, add, or alternative timeout;
- no KDA02/KDA03 or older-family economics;
- no protected or Capital.com payload;
- no deletion or overwrite of Stage 8C evidence.

## Required outputs

```text
KDA01_STAGE8C_FORENSIC_RECONCILIATION.md
KDA01_STAGE8C_METRIC_DECOMPOSITION.csv
KDA01_TIMESTAMP_SEMANTICS_AUDIT.md
KDA01_TIMESTAMP_AUTHORITY_LEDGER.csv
KDA01_STAGE8C_INDEPENDENT_RECOMPUTE.csv
KDA01_STAGE8C_SIGN_AND_PRICE_AUDIT.csv
KDA01_DEFINITION_LIMITATIONS.md
KDA01_INFERENCE_ESTIMAND_DIAGNOSTIC.md
KDA01_TIMESTAMP_REPAIR_DECISION.md
```

If no defect is confirmed, also produce:

```text
KDA01_STAGE8C_ORIGINAL_DECISION_REAFFIRMATION.md
```

If a repair is confirmed and rerun:

```text
KDA01_LEVEL3_REPAIRED_CONTRACT.json
KDA01_LEVEL3_REPAIRED_ACCEPTED_EXECUTIONS.parquet
KDA01_LEVEL3_REPAIRED_TRADE_TAPE.parquet
KDA01_LEVEL3_REPAIRED_DEFINITION_METRICS.csv
KDA01_LEVEL3_REPAIRED_GATE_MATRIX.csv
KDA01_LEVEL3_REPAIRED_BOOTSTRAP_SUMMARY.csv
KDA01_LEVEL3_REPAIRED_CONCENTRATION.csv
KDA01_LEVEL3_TIMING_DELTA.parquet
KDA01_LEVEL3_REPAIRED_DECISION.md
```

Always produce:

```text
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large Parquet outputs remain local and hash-manifested. Upload compact evidence and the complete manifest only.

## Tests and independent review

Test at minimum:

1. full independent recomputation of original Stage 8C arithmetic;
2. branch-side mapping;
3. exact open lookup at original and repaired timestamps;
4. bar-start and bar-availability semantics;
5. no price read before timestamp authority decision;
6. repaired entry uses `>= decision_ts`, never a searched delay;
7. exit target follows repaired actual entry;
8. schedule/non-overlap deterministic rebuild;
9. unchanged 16 definitions and gates;
10. original/repaired evidence separation;
11. trade-weighted bootstrap marked diagnostic only;
12. opposite-side symmetry marked non-candidate;
13. no controls, variants, protected rows, or other-family outcomes;
14. deterministic replay and artifact hashes.

Require independent review before any repaired outcome read and again after completion.

## Decision vocabulary

Return exactly one:

```text
Stage8C_execution_timing_correct_original_stop_reaffirmed
KDA01_level3_repaired_primary_pass_controls_required
KDA01_level3_repaired_no_primary_pass_stop
blocked_with_exact_mechanical_remedy
```

## Integration and handoff

After review:

- update only factual KDA01 execution, defect, run, evidence, and continuity records;
- preserve Stage 8C as immutable provenance;
- create task-scoped reviewed commits;
- non-force push under the standing workflow;
- use `drive_handoff: approved_default`;
- round-trip verify uploaded bytes and SHA-256;
- retain complete local evidence.

## Final response

```text
status:
actual_starting_commit:
original_Stage8C_reconciliation:
bar_timestamp_semantics:
extra_five_minute_delay_confirmed:
repair_contract_hash:
original_vs_repaired_schedule_counts:
original_vs_repaired_metric_summary:
timing_delta_summary:
trade_vs_equal_day_diagnostic:
sign_and_price_audit:
primary_pass_count_after_repair:
controls_executed: no
reverse_direction_executed: no
protected_rows_opened: no
economic_outputs_computed:
tests_and_reviews:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
terminal_decision:
human_approval_required:
```
