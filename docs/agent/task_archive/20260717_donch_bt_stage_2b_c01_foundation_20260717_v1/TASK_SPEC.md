# Stage 2B — C01 Identity, Multiplicity, and Feature Foundation

```text
task_id: donch_bt_stage_2b_c01_foundation_20260717_v1
target_environment: backtesting Codex
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
new_market_data_acquisition: no
```

## Objective

Build the smallest non-economic foundation needed to define and review C01:

```text
C01_debetaed_residual_shock_path_bifurcation
```

Deliver:

1. a registered C01 family and attempt/multiplicity contract;
2. deterministic candidate and episode identity support;
3. causal BTC/ETH residual and path-composition features;
4. a generator-only diagnostic tape containing no forward returns;
5. non-outcome candidate counts and overlap diagnostics;
6. an exact recommendation for the later frozen C01 generator contract.

Do not calculate trade returns, MAE/MFE after decision time, PnL, expectancy, control performance, or any promotion statistic.

## Accepted authority

Stage 2A1 is accepted:

```text
commit:
    9949b29ead0e6d6e17543ddd955bff0234805006

reference_panel_id:
    kraken_c01_reference_panel_v1

reference_panel_hash:
    2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763

members:
    PF_XBTUSD
    PF_ETHUSD

role:
    causal factor/reference series only

coverage:
    [2023-01-01, 2026-01-01)

continuous_tradeability_claim:
    no

survivorship_free_claim:
    no
```

The candidate universe is not U2. Until stronger PIT lifecycle evidence exists, label it exactly:

```text
current_roster_bar_existence_cohort
```

A candidate row is eligible only when the complete causal feature window has valid authorized trade and mark coverage and no known lifecycle/status invalidation applies.

## Start condition

Verify the repository, active `AGENTS.md` chain, clean synchronized `main`, current machine contracts, Stage 2A1 artifacts, and supported commands. Record the actual starting commit.

Proceed from the current clean synchronized state. Do not stop solely because routine task/archive commits changed the previously reported hash. Stop for an unexpected material code, data, authority, or protected-boundary change.

## C01 economic question

An asset-specific move remaining after causal BTC/ETH de-betaing may represent idiosyncratic information or demand.

Proposed path distinction:

```text
smooth residual shock:
    persistence/continuation question

jump-dominated residual shock:
    completed-failure/reversal question
```

This task establishes observable states only. It does not test whether either branch is profitable.

## Same-sample prohibitions

C01 must not become:

- A1/compression continuation under a new label;
- H43 residual-laggard buying after BTC impulse;
- a renamed relative-strength hard gate;
- a generic large-candle fade;
- RFBS, Backside, LFBS, or failed-breakdown threshold recombination;
- a search over the old best prior-high, retest, session, or failure thresholds.

Record these relationships in the family and attempt registry.

## Frozen feature definitions

Use 5-minute Kraken trade bars for returns. Mark bars are required for eligibility/quality and retained separately; do not substitute mark returns for trade returns.

### Returns

```text
r_i,t = log(trade_close_i,t / trade_close_i,t-1)
r_btc,t = PF_XBTUSD 5m log return
r_eth,t = PF_ETHUSD 5m log return
```

All timestamps must be aligned exactly. A missing required factor or candidate bar invalidates that observation.

### Primary residual model

For each candidate symbol and each 5-minute bar:

```text
r_i,t = alpha + beta_btc * r_btc,t + beta_eth * r_eth,t + epsilon_i,t
```

Fit using only observations strictly before `t`.

Primary estimation window:

```text
30 calendar days
minimum valid observations: 70% of expected 5-minute observations
refit cadence: once per UTC day using data through the prior UTC day
coefficients used unchanged during the following UTC day
```

This daily refit is required to keep the implementation causal, deterministic, and inexpensive.

### Robustness residual model

BTC-only:

```text
r_i,t = alpha + beta_btc * r_btc,t + epsilon_i,t
```

Use the same window, minimum coverage, and refit cadence.

This is the only residual-model robustness branch in this task.

### Shock window

Primary shock horizon:

```text
6 hours = 72 five-minute residuals
```

At each decision timestamp:

```text
residual_shock_6h = sum(epsilon over the trailing 72 completed bars)
```

Scale causally using the standard deviation of prior non-overlapping 6-hour residual sums over the trailing 30 calendar days, excluding the current shock window:

```text
residual_shock_z_6h
```

Require at least 80 valid prior 6-hour blocks. Do not fill missing blocks.

### Primary path measure

```text
largest_bar_share =
    max(abs(epsilon_j)) / sum(abs(epsilon_j))
```

over the current 6-hour shock window.

### Robustness path measure

```text
path_efficiency =
    abs(sum(epsilon_j)) / sum(abs(epsilon_j))
```

over the same window.

If the denominator is zero or any required bar is missing, the state is unavailable.

### Parent context diagnostics

Store, but do not use as entry gates in this task:

```text
BTC 6h return
ETH 6h return
BTC 24h realized volatility
candidate 24h realized volatility
candidate lagged trade-bar availability
candidate lagged mark-bar availability
```

Do not add funding, OI, basis, spread, session, catalyst, prior-high, or relative-strength features.

## Predeclared generator-only diagnostic states

These states are for count and feasibility review only. They are not approved trading rules.

Shock activation:

```text
positive shock: residual_shock_z_6h >= +3.0
negative shock: residual_shock_z_6h <= -3.0
```

Path diagnostics:

```text
smooth_primary:
    largest_bar_share <= 0.25
    and path_efficiency >= 0.50

jump_dominated_primary:
    largest_bar_share >= 0.50

intermediate:
    all other eligible shocks
```

Preserve all six sign/path combinations:

```text
positive_smooth
positive_jump
positive_intermediate
negative_smooth
negative_jump
negative_intermediate
```

Do not change these thresholds from candidate counts. If a branch is sparse, report it; do not repair it here.

## Candidate identity

Create deterministic identities before any future outcome calculation.

Minimum fields:

```text
family_id
definition_id
attempt_id
candidate_id
economic_address
canonical_episode_input_start
canonical_episode_input_end
symbol
venue
decision_ts
shock_window_start
shock_window_end
residual_model_version
feature_version
reference_panel_id
reference_panel_hash
candidate_cohort_version
data_authority_hash
trade_path_refs
mark_path_refs
protected_row_count
```

`candidate_id` must hash only causal identity fields.

## Episode identity

Implement the smallest interval-based, family-neutral episode interface required by C01.

For C01:

```text
episode_input_start = shock_window_start
episode_input_end = decision_ts + 24 hours
```

Cluster same-symbol intervals when they overlap. Do not add an arbitrary extra gap in this task.

Store:

```text
canonical_episode_id
episode_cluster_start
episode_cluster_end
episode_member_count
```

The algorithm must be deterministic and independent of outcomes.

### Cross-family preflight

Using only safe candidate/identity/timestamp fields, determine whether the nearest prior family ledgers can be mapped through the same interval interface:

- A1/compression continuation;
- BTC-led alt diffusion/H43;
- relative-strength breakout;
- RFBS/completed-failure;
- repaired Backside.

Do not read return, PnL, MAE/MFE, or protected columns.

If safe identity inputs are unavailable, report the exact blocker. Do not reconstruct them from economic summaries.

Do not perform a broad 14-family backfill in this task.

## Multiplicity and attempt registry

Register before generator execution:

```text
family:
    C01_debetaed_residual_shock_path_bifurcation

planned definitions:
    primary BTC+ETH residual model
    robustness BTC-only residual model

fixed shock horizon:
    6h

fixed activation:
    absolute z >= 3.0

path states:
    smooth
    jump_dominated
    intermediate

sign branches:
    positive
    negative
```

Every combination counts as an attempted definition even if it produces zero candidates.

Record:

```text
attempt_id
family_id
component_dimensions
fixed_values
status
candidate_count
killed_or_retained_for_later_review
reason
source_commit
feature_hash
data/cohort/reference hashes
```

No attempt may be removed because it is sparse or aesthetically weak.

## Outputs

Use repository conventions. Produce at least:

```text
C01_FAMILY_AND_ATTEMPT_REGISTER.csv
C01_FEATURE_CONTRACT.md
C01_FEATURE_SCHEMA.json
C01_GENERATOR_DIAGNOSTIC_TAPE.parquet
C01_CANDIDATE_COUNT_MATRIX.csv
C01_EPISODE_IDENTITY_REPORT.md
C01_CROSS_FAMILY_OVERLAP_PREFLIGHT.csv
C01_GENERATOR_REVIEW.md
C01_NEXT_CONTRACT_RECOMMENDATION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

The diagnostic tape must not contain any future return, exit, PnL, MAE/MFE, promotion, or outcome-derived field.

Candidate counts must be reported by:

```text
year
symbol
residual_model
sign
path_state
```

Also report:

- unavailable-feature counts;
- missing trade/mark window counts;
- duplicate candidate IDs;
- exact duplicate economic addresses;
- episode cluster sizes;
- overlap counts with each safely mappable nearest prior family.

## Tests

Use synthetic fixtures for:

1. no coefficient uses same-day or future bars;
2. daily refit boundary;
3. missing factor/candidate bars fail closed;
4. BTC+ETH and BTC-only residual arithmetic;
5. 6-hour shock and causal scale;
6. largest-bar share and path efficiency;
7. exact threshold boundaries;
8. deterministic candidate IDs;
9. deterministic interval clustering;
10. no outcome columns in outputs;
11. no pre-2023, non-Kraken, or 2026+ candidate;
12. reference-panel and cohort hashes enforced;
13. trade and mark paths remain distinct;
14. changing future bars cannot change past features or candidates.

Run focused tests and the smallest relevant repository regression/guard suite.

## Acceptance criteria

```text
economic outputs computed: zero
protected outcomes opened: zero
C01 family registered: yes
all predeclared attempts retained: yes
causal feature tests: pass
candidate identities deterministic: yes
episode identities deterministic: yes
candidate counts produced without returns: yes
candidate cohort label: current_roster_bar_existence_cohort
survivorship-free claim: no
continuous-tradeability claim: no
independent review: approve
```

## Integration and Drive handoff

After tests and independent review pass:

- create one task-scoped commit;
- integrate and non-force push under the standing reviewed-task workflow;
- upload the closed package using `drive_handoff: approved_default`;
- verify the destination from `docs/agent/DRIVE_HANDOFF_TARGET.md`;
- keep the local package.

Do not include raw market data in the Drive package.

## Deferred work

Do not:

- calculate C01 returns or controls;
- select among definitions by performance;
- change thresholds from counts;
- close the external-review package;
- acquire spot/reference data for C02;
- compute C03 breadth;
- use funding;
- access capture;
- inspect protected data;
- simplify governance/instructions.

## Next decision

At completion, return one of:

```text
ready_for_C01_generator_contract_review
```

or:

```text
blocked_with_exact_non_economic_remedy
```

Do not prepare or run the C01 economic test.

## Final response

```text
status:
actual_starting_commit:
family_and_attempt_registry:
feature_contract_hash:
reference_panel_hash:
candidate_cohort_version:
candidate_counts_summary:
unavailable_feature_summary:
candidate_identity_checks:
episode_identity_checks:
cross_family_overlap_preflight:
tests_and_results:
protected_outcomes_opened: no
economic_outputs_computed: no
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
artifacts_and_sha256:
remaining_limits:
next_status:
human_approval_required:
```
