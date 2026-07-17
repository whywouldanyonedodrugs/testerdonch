# Stage 3B — C02 Spot/Perp Leadership Generator Foundation

```text
task_id: donch_bt_stage_3b_c02_leadership_generator_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
new_market_data_acquisition: no
```

## Objective

Build and independently review a causal, outcome-free C02 generator for:

```text
C02_spot_led_vs_perp_led_impulse
```

Generate deterministic impulse-onset, leadership, and completed-failure identities; counts and synchronization diagnostics only.

Do not calculate post-decision returns, PnL, MAE/MFE, control performance, promotion metrics, or economic rankings.

## Accepted authority

```text
Stage 3A final repository commit:
    2a83432a5ecd94b284b3a9c8f6366e4e0ae8df1f

spot panel:
    Kraken_USD_spot_bar_existence_panel

spot data root:
    /opt/parquet/kraken_spot_reference/official_kraken_time_sales/

spot manifest content hash:
    3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046

spot coverage:
    204 official USD pairs with observed archive bars
    [2023-01-01, 2026-01-01), pair-specific sparse intervals

PF candidate authority:
    Stage 2C mechanism-proof cohort
    current-roster capped; not survivorship-free

C01 terminal decision:
    level3_no_primary_pass_stop
```

C01 is closed. Do not reuse its outcomes, reverse its directions, or retune it.

## Mechanism and distinctness

Hypothesis:

- persistent same-asset Kraken spot inventory pressure may lead and then transmit into the Kraken perpetual;
- a same-asset impulse led mainly by the perpetual may reflect leveraged demand and becomes a distinct reversal question only after a completed failure.

C02 is:

```text
same asset
same Kraken venue family
spot versus perpetual timing
```

It is not:

- BTC-led alt-laggard selection;
- cross-coin diffusion;
- another-venue price discovery;
- a simultaneous ALT/BTC breakout gate;
- funding, OI, index, mark-minus-last, or breadth alpha;
- a relabeling of C01, H43, relative-strength breakout, RFBS, or Backside.

Register the family and all attempted definitions before generation.

## Input and cohort rules

Use only:

- official sparse Kraken USD spot five-minute bars from Stage 3A;
- authorized Kraken PF trade and mark five-minute bars;
- the Stage 2C causal daily PF liquidity cohort;
- existing lifecycle masks and identity authority.

A symbol-day is eligible only when:

1. the PF symbol belongs to the Stage 2C daily Top-100 causal PF-liquidity cohort;
2. its exact canonical asset has an included official Kraken USD spot mapping;
3. all spot, PF-trade, and PF-mark bars required by the feature/event window exist;
4. the prior 30 calendar days contain:
   - at least 20 UTC days with at least 70% exact spot/PF five-minute intersection coverage; and
   - at least 70% exact intersection coverage across the complete 30-day window;
5. no known lifecycle-invalid interval intersects the required window;
6. price-unit equivalence between the spot pair and PF contract is verified from existing authority or fails closed.

Do not fill, interpolate, or forward-fill spot bars. No trade in an interval means no observed spot bar under this contract.

Report BTC and ETH separately from the alt group. They may be C02 same-asset candidates; they are not mixed into alt summaries.

## Frozen causal features

### Completed bars

All features use completed, exactly aligned UTC five-minute bars. At decision time `t`, every source bar must have:

```text
feature_available_ts <= t
```

Use PF trade close for market-return leadership. PF mark is required for eligibility and later failure confirmation; it is not substituted for trade price.

### Fifteen-minute returns

For each market:

```text
spot_r_15m = log(spot_close_t / spot_close_t-15m)
perp_r_15m = log(perp_trade_close_t / perp_trade_close_t-15m)
```

Require three consecutive observed five-minute intervals and the starting close. A gap invalidates the feature.

### Causal scale

Once per UTC day, using data through the prior UTC day only:

```text
lookback: 30 calendar days
series: all complete rolling 15-minute returns
minimum valid observations per market: 2,000
scale: sample standard deviation
```

Zero/non-finite scale fails closed.

```text
spot_z_15m = spot_r_15m / prior_scale_spot
perp_z_15m = perp_r_15m / prior_scale_perp
```

No full-sample normalization.

### Confirmed impulse activation

At completed bar `t`, choose direction `d ∈ {+1,-1}` only when spot and perp returns have the same sign.

A confirmed impulse is active when:

```text
max(d * spot_z_15m, d * perp_z_15m) >= 3.0
min(d * spot_z_15m, d * perp_z_15m) >= 1.5
```

The event onset is the first active bar after 60 consecutive minutes with no same-direction confirmed impulse.

Do not select the strongest or latest row in the episode.

## Leadership classification

For the event direction, define the follower threshold:

```text
d * z_15m >= 1.5
```

Within the specified lookback, identify the first completed bar at which spot and perp each crossed that threshold.

### Primary classification

```text
lookback: 15 minutes

spot_led:
    spot crossing precedes perp crossing by at least 5 minutes

perp_led:
    perp crossing precedes spot crossing by at least 5 minutes

simultaneous:
    crossings occur on the same five-minute bar

ambiguous:
    a first crossing cannot be established within the lookback
```

### Robustness classification

Repeat the same classification with:

```text
lookback: 30 minutes
```

The 15-minute classification is primary. The 30-minute version is robustness only and cannot replace it after counts.

## Diagnostic features, not gates

Store causally:

- spot and PF 15-minute close-based USD-volume proxy;
- each market's volume surprise versus its prior-30-day median;
- `10,000 * log(perp_trade_close / spot_close)`;
- 15-minute change in that price-gap measure;
- PF mark-versus-trade gap;
- BTC and ETH 15-minute returns;
- prior-day PF liquidity rank;
- exact spot/PF intersection coverage.

Do not threshold or rank these fields in this task.

Do not add funding, OI, index, breadth, session, catalyst, prior-high, or C01 residual/path features.

## Branch identities

Preserve the following onset branches separately by sign:

```text
spot_led_continuation
simultaneous_impulse
perp_led_continuation
perp_led_completed_failure
ambiguous_diagnostic
```

The first three have decision time equal to the impulse onset.

### Perp-led completed failure

For a primary-classified perp-led event only, search causally for up to six hours after onset.

Positive impulse failure:

```text
PF trade close < impulse-window low
and PF mark close < impulse-window low
```

Negative impulse failure:

```text
PF trade close > impulse-window high
and PF mark close > impulse-window high
```

Use the first completed bar satisfying both. Its close time is the failure decision time. No confirmation means no failure candidate.

The impulse window is the three completed five-minute bars used for the onset's 15-minute return. No anticipatory failure and no touch-only failure.

## Alignment sensitivity

The exact UTC alignment is primary.

Recompute classification diagnostics only with spot intervals shifted:

```text
-5 minutes
+5 minutes
```

Do not use shifted candidates for economics or choose the shift from counts.

Report:

- same canonical impulse episode;
- same direction;
- same primary leadership state;
- counts changing among spot-led, simultaneous, perp-led, and ambiguous.

Classify the result:

```text
alignment_stable_for_contract_review
```

only when, among events present under exact and both shifted alignments:

```text
same direction and same episode >= 80%
same leadership state >= 70%
```

Otherwise return:

```text
alignment_fragile_requires_review
```

Do not alter thresholds to improve stability.

## Identity and episodes

Create deterministic causal identities containing at least:

```text
family_id
attempt_id
event_id
economic_address
canonical_asset_id
PF_symbol
Kraken_spot_pair
direction
impulse_start
impulse_onset_ts
decision_ts
leadership_state
leadership_lookback
failure_state
reference/data/cohort/feature hashes
spot/PF trade/PF mark path references
protected_row_count
```

Cluster same-symbol intervals into family-neutral episodes using:

```text
episode_start = impulse_start
episode_end = max(onset + 6 hours, failure decision if present)
```

Merge overlapping intervals only. Do not use outcomes or later exits.

## Multiplicity register

Register before generation:

```text
leadership lookbacks:
    15m primary
    30m robustness

directions:
    positive
    negative

states:
    spot-led continuation
    simultaneous impulse
    perp-led continuation
    perp-led completed failure
    ambiguous diagnostic
```

Retain all attempts, including zero-count attempts. Shifted-alignment runs are sensitivity attempts and count in the research record.

## Required outputs

```text
C02_FAMILY_AND_ATTEMPT_REGISTER.csv
C02_GENERATOR_CONTRACT.md
C02_FEATURE_SCHEMA.json
C02_DAILY_ELIGIBILITY_AUDIT.parquet
C02_IMPULSE_EVENT_TAPE.parquet
C02_FAILURE_EVENT_TAPE.parquet
C02_EVENT_COUNT_MATRIX.csv
C02_ALIGNMENT_SENSITIVITY.csv
C02_LEADERSHIP_AGREEMENT_REPORT.md
C02_EPISODE_IDENTITY_REPORT.md
C02_NEAREST_FAMILY_OVERLAP_PREFLIGHT.csv
C02_GENERATOR_REVIEW.md
C02_NEXT_CONTRACT_RECOMMENDATION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

No output may contain post-decision return, exit, PnL, MAE/MFE, expectancy, promotion, or outcome-derived labels.

Counts must include:

- year;
- BTC/ETH versus alt;
- symbol;
- direction;
- primary and robustness leadership;
- branch;
- failure confirmed/unconfirmed;
- eligibility and missing-window reasons;
- canonical episode sizes.

## Nearest-family overlap preflight

Using safe causal identity fields only, compare with:

- H43 BTC-led alt diffusion;
- relative-strength breakout;
- RFBS/completed failure;
- repaired Backside;
- C01 onset episodes.

Do not read economic columns. Preserve exact blockers where causal starts are unavailable.

## Tests

Use synthetic fixtures for:

- exact sparse intersection and no filling;
- prior-day-only eligibility and scale;
- exact 15-minute returns;
- impulse threshold boundaries;
- first-onset/reset behavior;
- 15m and 30m leadership crossing times;
- simultaneous and ambiguous states;
- no peak-row selection;
- completed trade-and-mark failure;
- no touch-only or anticipatory failure;
- ±5m alignment diagnostics;
- deterministic IDs and episodes;
- no pre-2023, 2026+, non-Kraken, lifecycle-invalid, or mixed-file rows;
- no outcome fields or post-decision reads;
- future data changes cannot alter past eligibility or events.

Run focused tests plus relevant spot/PF loader, lifecycle, protected-boundary, identity, and archive guards.

## Acceptance criteria

```text
economic outputs computed: zero
protected outcomes opened: zero
sparse spot gaps filled: zero
C02 family registered: yes
all attempts retained: yes
causal leadership classification: pass
completed-failure classification: pass
deterministic event and episode identity: pass
alignment result: reported under frozen criteria
independent review: approve
```

## Integration and handoff

After tests and independent review:

- create task-scoped commit(s);
- non-force push under the standing reviewed-task workflow;
- upload the closed package using `drive_handoff: approved_default`;
- omit large parquet and raw market payloads from Drive, retaining them locally with hashes;
- update only factual family, capability, multiplicity, and continuity records.

## Final status

Return exactly one:

```text
ready_for_C02_generator_contract_review
alignment_fragile_requires_review
blocked_with_exact_non_economic_remedy
```

No C02 economic run is authorized.

## Final response

```text
status:
actual_starting_commit:
family_and_attempt_registry:
generator_contract_hash:
spot_manifest_hash:
cohort_and_eligibility_summary:
impulse_event_counts:
leadership_counts:
failure_counts:
alignment_sensitivity:
episode_identity:
nearest_family_overlap:
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
