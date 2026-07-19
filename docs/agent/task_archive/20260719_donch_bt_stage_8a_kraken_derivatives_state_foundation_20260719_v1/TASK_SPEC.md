# Stage 8A — Kraken Derivatives-State Semantic and Generator Foundation

```text
task_id: donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1
target_environment: backtesting Codex
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
new_market_data_acquisition: no
commit_authorized: yes — task-scoped reviewed commits only
push_authorized: yes — non-force under the standing reviewed workflow
```

## Objective

Turn the completed Stage 7C Kraken analytics acquisition into a reviewed, causal, outcome-free research foundation for:

```text
KDA01_crowding_price_progress_bifurcation
KDA02_liquidation_oi_purge_state_machine
KDA03_basis_state_and_dislocation_feasibility
```

Implement the binding internal semantic contract, build reusable causal features, register all attempts, generate deterministic event tapes and counts, and recommend no more than two later frozen economic contracts.

Do not calculate any post-decision return, exit, PnL, MAE/MFE, control performance, threshold performance, or promotion statistic.

## Current state to verify

Verify and record:

```text
repository_root
applicable_AGENTS.md_chain
branch_commit_remotes
working_tree_and_untracked_files
machine_contracts
supported_commands
data_roots
task_archive_convention
Drive_target_and_identity
```

Current Donch sources report a published multi-platform boundary at:

```text
bf7a694c3d0764807cd12015046f633be54c53ab
```

The Stage 7C completion task reports no tracked repository mutation and one pre-existing unrelated untracked Capital.com archive. Treat both as claims to verify. Do not touch, stage, move, hash into this task, or delete the Capital.com archive.

Proceed from the current clean synchronized `main` in an isolated worktree. Do not stop solely because expected routine descendants changed a prior hash. Stop for a material code, data, authority, or protected-boundary conflict.

## Archive context

```text
project_decisions:
    C01 level3_no_primary_pass_stop
    C02 level3_no_primary_pass_stop
    C03 C03_PIT_authority_unavailable
    C16 C16_flow_authority_unavailable

same_sample_prohibitions:
    no C01/C02 reversal, threshold, horizon, symbol, year, funding,
    or analytics-filter rescue
    no RFBS/Backside/failed-breakdown threshold rescue
    no delayed-flush chart proxy presented as liquidation evidence
    no TSMOM/A1/H43/relative-strength subset promotion from new analytics

new_data_decision:
    Stage 7C acquisition complete
    semantic decision is binding internally and explicitly inferred

approval_scope:
    rankable 2023-2025 local Kraken trade, mark, spot where already
    authorized, and Stage 7C analytics reads
    non-economic code, feature cache, event tapes, counts, tests,
    registries, archives, commit, push, and Drive handoff

still_forbidden:
    economic outcomes
    protected data
    new acquisition
    capture
    Capital.com payloads
    private endpoints
    orders
    live or deployment changes
```

## Immutable Stage 7C authority

```text
Drive ZIP SHA-256:
    14d49653a5f2c868b6eb38cff89dcb932d6fc6bc629db56b19ffcfe7ae708bf8

package content hash:
    c5be5d045433e1dc13ef9fbf1e2915d62f101b678088f04eb70630f952c75150

data-manifest content hash:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

frozen symbol inventory SHA-256:
    0e1c35b3348b3187547f188dac47e5e0f5bc3d259207fa3884bc8efccdd36f09

local data root:
    /opt/parquet/kraken_derivatives/analytics/stage7c_v1

rankable interval:
    [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)

five-minute scope:
    frozen 460-identity inventory, endpoint-supported rows

one-minute scope:
    PF_XBTUSD
    PF_ETHUSD

metrics:
    future-basis
    open-interest
    liquidation-volume
```

Before reading analytics values, verify the final manifest, all referenced object identities needed by this task, purpose, interval, schema, and protected boundary.

## Required transferred Donch files

The operator will attach:

```text
DONCH_DECISION_Kraken_Analytics_Inferred_Semantics_2026-07-19_v1.json
SHA-256: c5ccd4f57981dfd949857016410fab87defaecc4635a6951fe5ee3e4965ede48

DONCH_Kraken_Analytics_Semantics_and_Research_Plan_2026-07-19_v1.md
```

Archive both exactly as received and verify their hashes before implementation.

## Binding semantic contract

Implement the JSON decision exactly.

### Future basis

```text
basis_decimal = Decimal(basis_raw)
basis_percent = basis_decimal * 100
basis_bps = basis_decimal * 10000

aggregation:
    closing/last state
sign:
    positive = futures above reference
    negative = futures below reference
```

Preserve exact raw fields. Use within-symbol causal changes, robust scales, and percentiles. Basis is not funding.

### Open interest

```text
value_0_raw = open
value_1_raw = high
value_2_raw = low
value_3_raw = close

unit:
    named Kraken contract base-unit quantity

direction:
    unsigned
```

Do not infer long/short direction from OI. Do not divide by two.

### Liquidation volume

```text
value_raw:
    nonnegative base-unit quantity force-closed in the interval

aggregation:
    sum

direction:
    unavailable in source
```

A price-derived side is allowed only in a separate field named:

```text
price_inferred_liquidation_side
```

with values `long_liquidation_proxy`, `short_liquidation_proxy`, or `ambiguous`.

### Project status

Store each decision as:

```text
semantic_status:
    inferred_authoritative_v1
```

Do not relabel it `verified`.

## Scope and cohort

Build a new data-authority object:

```text
current_roster_analytics_bar_existence_cohort_v1
```

Start from the existing Stage 2C crypto-perpetual mechanism-proof exclusions and authority where available:

- exclude stablecoin, fiat, commodity, index, DTF, pre-IPO, xStock and other non-crypto contracts;
- retain BTC and ETH as a separately reported major cohort;
- use prior-day causal Top-100 liquidity for the alt mechanism-proof cohort;
- require exact event-time trade, mark, and required analytics coverage;
- mask known lifecycle-invalid intervals;
- unknown lifecycle remains a claim cap.

Do not infer lifecycle eligibility from analytics rows. No survivorship-free or continuous-tradeability claim is permitted.

## Implementation architecture

Make the smallest reusable implementation.

Create or extend only the minimal source-neutral/analytics-specific modules needed for:

1. semantic loading and validation;
2. exact trade/mark/analytics alignment;
3. causal daily normalization;
4. feature caching;
5. outcome-free event generation;
6. identities, episodes, attempts, and overlap reports.

Use column projection and predicate pushdown. Never load the full 229.9-million-row panel into pandas or memory. Use DuckDB, PyArrow dataset scans, or the repository's existing bounded columnar path.

Target peak RSS:

```text
< 4 GiB
```

Feature cache and event tapes must be resumable or partitioned. Preserve source/data/feature hashes.

## Shared causal feature contract

Use completed five-minute rows as the broad research grid.

For BTC/ETH only, compute one-minute diagnostics for timing and aggregation validation. Do not use one-minute alt proxies.

Required raw/derived fields:

### Price and path

```text
trade_log_return_5m
trade_return_15m
trade_return_1h
trade_return_6h
trade_return_24h
mark_return_1h
mark_return_6h
realized_vol_1h
realized_vol_24h
path_efficiency_1h
path_efficiency_6h
largest_bar_share_1h
largest_bar_share_6h
```

### Open interest

```text
oi_open_base_units
oi_high_base_units
oi_low_base_units
oi_close_base_units
oi_log_change_15m
oi_log_change_1h
oi_log_change_6h
oi_log_change_24h
oi_range_fraction_1h
oi_notional_usd_proxy
oi_notional_log_change_1h
oi_notional_log_change_6h
```

### Basis

```text
basis_decimal
basis_bps
basis_change_15m
basis_change_1h
basis_change_6h
basis_level_robust_z
basis_change_robust_z
```

### Liquidation

```text
liquidation_base_units_5m
liquidation_base_units_1h
liquidation_base_units_6h
liquidation_mark_notional_usd_proxy_1h
liquidation_to_lagged_oi_1h
liquidation_to_lagged_oi_6h
liquidation_to_lagged_trade_volume_1h
liquidation_intensity_percentile
price_inferred_liquidation_side
```

### Parent and hygiene

```text
BTC_return_1h
BTC_return_6h
ETH_return_1h
ETH_return_6h
prior_day_liquidity_rank
trade_coverage
mark_coverage
analytics_coverage
known_lifecycle_mask
```

Use exact Decimal parsing or a lossless audited conversion. Keep raw strings addressable.

## Causal normalization

Once per UTC day, use data through the prior UTC day only.

Primary lookback:

```text
60 calendar days
```

Minimum:

```text
30 valid days
70% expected valid observations inside the lookback
```

Produce:

- median/MAD robust z-scores;
- empirical percentiles;
- valid-observation counts;
- stale/missing flags.

Zero MAD or non-finite scales fail closed.

Do not normalize from the full sample. Changing future rows must not change past features.

## Family registration

Register before authoritative generator execution:

```text
KDA01_crowding_price_progress_bifurcation
KDA02_liquidation_oi_purge_state_machine
KDA03_basis_state_and_dislocation_feasibility
```

Link nearest prior families:

```text
KDA01:
    FORUM H01 / C04
    RFBS
    Backside
    failed-breakdown
    C01

KDA02:
    delayed-flush reclaim
    PD04 OI vacuum
    FORUM H04 as a stricter typed-flow relative
    RFBS/Backside as price controls

KDA03:
    C08 / local basis-dislocation priors
    C02 spot/perp relation as a control
```

Record every attempted definition, state dimension, lookback, threshold, zero-count branch, and killed branch. Do not drop sparse attempts.

## Outcome-free generator design

The agent owns the exact frozen generator specification, subject to these limits:

### KDA01 required states

At minimum define causal onsets for:

```text
crowding_with_efficient_price_progress
crowding_with_deteriorating_price_progress
completed_failure_after_crowding_deterioration
```

Requirements:

- OI expansion is required;
- basis pressure/state is required in the primary definition;
- price progress is measured separately from crowding;
- both positive and negative parent directions are retained;
- completed failure is causal and trade+mark confirmed;
- no funding field is required;
- no threshold may be chosen from outcomes.

Primary thresholds must be standard, predeclared statistical boundaries such as robust z >= 2 or empirical 95th percentile. One robustness boundary is permitted. Do not run a grid.

### KDA02 required states

At minimum define:

```text
active_liquidation_purge
completed_purge_reclaim_or_failure
OI_vacuum_with_modest_price_displacement
```

Requirements:

- primary liquidation intensity uses `liquidation_to_lagged_oi`;
- OI reset/contraction is a separate component;
- price path and mark confirmation are separate components;
- positive and negative price directions are retained;
- liquidation side remains a price-inferred proxy;
- typed liquidation/assignment/termination claims are prohibited.

Use one primary and one robustness intensity boundary. Do not run a threshold grid.

### KDA03 feasibility states

Produce counts and distributions for:

```text
basis_expansion_with_OI_confirmation
basis_expansion_without_price_confirmation
extreme_basis_with_liquidation_and_OI_reset
```

Do not create an economic entry/exit contract in this task unless KDA01/KDA02 are mechanically unavailable and the independent review approves the substitution. The final recommendation may still rank KDA03 next.

## Event and episode identity

Create deterministic causal identities containing:

```text
family_id
definition_id
attempt_id
event_id
economic_address
symbol
direction
state_start
decision_ts
feature_window_start
feature_window_end
semantic_contract_hash
analytics_data_manifest_hash
trade_and_mark_authority_hashes
cohort_version
feature_version
source_path_refs
protected_row_count
```

Cluster same-symbol overlapping mechanism intervals into family-neutral canonical episodes. Use only causal interval identity; no outcomes.

## Required diagnostics

Report counts by:

```text
year
symbol
major_vs_alt
direction
family
definition
state
```

Also report:

- unavailable-feature counts and reasons;
- coverage by year/symbol/metric;
- data-start truncation, especially OI in 2023;
- duplicate IDs/economic addresses;
- episode sizes and overlap;
- one-minute versus five-minute BTC/ETH state agreement;
- sensitivity to one-minute timing for BTC/ETH only;
- event concentration before outcomes;
- exact overlap with RFBS, Backside, failed-breakdown, LFBS, H43, C01 and C02 where safe causal identities exist.

Read only safe identity/timestamp fields from older families. Do not read their economic columns.

## Required outputs

Use the verified task archive convention and retain at least:

```text
TASK_SPEC.md
DONCH_ARCHIVE_CONTEXT.md
PLAN.md
DECISIONS_AND_PROGRESS.md
ANALYTICS_SEMANTIC_CONTRACT.json
ANALYTICS_SEMANTIC_IMPLEMENTATION.md
ANALYTICS_DATA_AUTHORITY_AND_COVERAGE.md
KDA_SHARED_FEATURE_CONTRACT.md
KDA_SHARED_FEATURE_SCHEMA.json
KDA_FEATURE_CACHE_MANIFEST.json
KDA_FAMILY_AND_ATTEMPT_REGISTER.csv
KDA01_EVENT_TAPE.parquet
KDA01_EVENT_COUNT_MATRIX.csv
KDA02_EVENT_TAPE.parquet
KDA02_EVENT_COUNT_MATRIX.csv
KDA03_FEASIBILITY_MATRIX.csv
KDA_ONE_MINUTE_TIMING_DIAGNOSTICS.csv
KDA_CANONICAL_EPISODE_REPORT.md
KDA_OLD_FAMILY_OVERLAP.csv
KDA_GENERATOR_REVIEW.md
KDA_NEXT_ECONOMIC_CONTRACT_RECOMMENDATION.md
CHANGED_FILES.md
COMMANDS_AND_RESULTS.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large feature/event Parquet files remain local and hash-manifested. Omit them from Drive.

## Tests

Use synthetic fixtures and bounded real-data inspections for:

1. semantic decision hash and version enforcement;
2. future-basis Decimal/bps conversion;
3. OI exact OHLC tuple mapping;
4. liquidation sum semantics and unsigned status;
5. no silent long/short liquidation label;
6. contract-base-unit and mark-notional proxy arithmetic;
7. exact timestamp and source-path alignment;
8. causal prior-day normalization;
9. future rows cannot change past features/events;
10. missing metric windows fail closed;
11. OI pre-retention windows remain unavailable;
12. no pre-2023, 2026+, non-Kraken, mixed, or lifecycle-invalid event;
13. deterministic feature/event/episode IDs;
14. no outcome columns or outcome readers;
15. one-minute/five-minute BTC/ETH aggregation;
16. bounded-memory partitioned execution;
17. registry retains every attempted branch;
18. old-family overlap reads safe identity fields only.

Run focused tests and the relevant repository loader, protected-boundary, lifecycle, identity, attempt, archive, C01/C02, and analytics acquisition regressions.

## Acceptance criteria

```text
semantic contract implemented and versioned: yes
raw fields preserved: yes
economic interpretation status: inferred_authoritative_v1
feature cache causal and deterministic: yes
KDA01 event tape: produced or exact mechanical blocker
KDA02 event tape: produced or exact mechanical blocker
KDA03 feasibility: reported
attempt registry complete: yes
outcome columns/readers: zero
protected rows opened: zero
economic outputs computed: zero
peak RSS: < 4 GiB
independent review: approve
```

Do not weaken a definition after counts. If a branch is sparse, report it.

## Recommendation rule

Recommend no more than two later economic contracts.

Rank using only:

```text
causal clarity
distinctness
coverage
event counts and year distribution
identity quality
mechanical falsifiability
execution compatibility
overlap/multiplicity risk
```

Do not use returns or economic summaries.

Preferred order absent a mechanical blocker:

```text
1. KDA01 crowding-price-progress
2. KDA02 liquidation/OI purge
3. KDA03 basis state
```

Do not implement the economic runner.

## Git, review, and handoff

Use an isolated worktree.

After tests and independent review pass:

- create task-scoped commit(s);
- fast-forward and non-force push under the standing reviewed workflow if `main` has no unexpected material change;
- update factual capability, family, attempt, source, and continuity records;
- preserve every existing negative decision;
- use `drive_handoff: approved_default`;
- verify remote bytes and SHA-256;
- retain the local archive.

If the existing untracked Capital.com archive prevents a clean main checkout, leave it untouched and use an isolated worktree. Do not treat it as a Stage 8A blocker unless it overlaps an approved path.

## Stop conditions

Stop and preserve evidence if:

- Stage 7C manifest/content identity fails;
- analytics/trade/mark alignment is ambiguous;
- a protected row may be opened;
- semantics cannot be implemented exactly as the transferred decision;
- outcome fields enter the pipeline;
- memory/resource use exceeds limits without a bounded remedy;
- scope expands to new acquisition, Capital.com, capture, or economics;
- Git state contains unexpected material changes.

## Final response

```text
status:
actual_starting_commit:
semantic_contract_hash:
analytics_data_manifest_hash:
cohort_version_and_hash:
shared_feature_contract_hash:
KDA01_counts_and_coverage:
KDA02_counts_and_coverage:
KDA03_feasibility:
one_minute_timing_findings:
canonical_episode_and_old_family_overlap:
tests_and_resource_use:
protected_rows_opened: no
economic_outputs_computed: no
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
recommended_next_contracts:
human_approval_required:
```
