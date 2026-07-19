# Kraken Analytics Semantic Adjudication and Research Plan

```text
date: 2026-07-19
revision: 1
scope: Kraken-only research
economic_run_authorized: no
protected_outcome_access: no
semantic_decision_id: DONCH_KRAKEN_ANALYTICS_SEMANTICS_20260719_V1
semantic_decision_sha256: c5ccd4f57981dfd949857016410fab87defaecc4635a6951fe5ee3e4965ede48
```

## Executive decision

The Stage 7C acquisition is accepted as complete and technically reliable.

The acquired panel is large enough and clean enough to change the Kraken research roadmap:

```text
final normalized rows: 229,869,200
five-minute identities frozen: 460
one-minute coverage: PF_XBTUSD and PF_ETHUSD
metrics: open interest, liquidation volume, future basis
protected rows: 0
economic outputs: 0
local apparent data size: approximately 1.47 GiB
```

The unresolved units no longer block non-economic or later economic research. They are resolved by the versioned internal semantic decision accompanying this report. The adopted meanings remain explicitly marked `inferred_authoritative_v1`, not falsely upgraded to primary-source verification.

The first new research wave should contain exactly two mechanisms:

1. **KDA01 — crowding versus price progress**, using OI and basis as the parent state and completed failure as one possible confirmation.
2. **KDA02 — liquidation/OI purge state**, using actual aggregate liquidation intensity and OI reduction to distinguish ongoing forced pressure from completed deleveraging.

A third mechanism, **KDA03 — basis dislocation and convergence**, should be prepared as a second-wave candidate and control layer, not rushed into the first economic run.

Closed C01/C02 and older price-pattern translations remain closed. Their event tapes may serve as overlap diagnostics and controls; the new data may not be used to select filters that retrospectively rescue them.

## Acquisition assessment

The completion package supports all of the following statements:

- all 1,836 acquisition units and 154,747 requests completed;
- all responses were HTTP 200;
- 229,974,051 raw rows compacted to 229,869,200 unique rows;
- 104,851 equal pagination-boundary duplicates were removed;
- no conflicting duplicate was reported;
- 3,672 final data objects passed size and SHA-256 verification;
- no 2026+ row entered the normalized data;
- peak RSS remained about 1.34 GiB;
- the full normalized and raw evidence remains local and manifest-addressable.

Coverage is strong but not identical across metrics:

| Metric | Resolution | Final rows | Coverage note |
|---|---:|---:|---|
| Future basis | 1m | 3,156,480 | BTC/ETH, full 2023–2025 |
| Liquidation volume | 1m | 3,156,480 | BTC/ETH, full 2023–2025 |
| Open interest | 1m | 2,897,280 | BTC/ETH begins 2023-04-01 |
| Future basis | 5m | 74,376,703 | 456/460 identities returned rows |
| Liquidation volume | 5m | 74,222,329 | 456/460 identities returned rows |
| Open interest | 5m | 72,059,928 | 454/460 identities; earliest 2023-03-07 |

The panel is not a survivorship-free universe. The correct label is:

```text
current_roster_analytics_bar_existence_cohort
```

## Binding semantic adjudication

### Future basis

Adopt:

```text
basis_decimal = Decimal(basis_raw)
basis_percent = basis_decimal * 100
basis_bps = basis_decimal * 10,000
```

`basis_raw` is a signed closing state. Positive means the futures reference is above the spot/reference market; negative means below. Five-minute values equal the final one-minute value in their buckets, confirming a state rather than a flow.

The decimal-ratio scale is the most coherent interpretation. Typical values of `0.0001` to `0.0010` correspond to 1–10 bps; interpreting them as absolute dollars is not coherent across BTC, ETH, and low-priced tokens. Extreme observations must remain visible rather than being silently clipped.

Primary research should rely on causal within-symbol changes, robust z-scores, and empirical percentiles. This makes the research insensitive to modest remaining scale uncertainty.

### Open interest

Adopt:

```text
value_0_raw = open
value_1_raw = high
value_2_raw = low
value_3_raw = close
unit = named Kraken contract base-unit quantity
```

The OHLC ordering is effectively established: 74,957,208 rows had zero structural violations, and every overlapping BTC/ETH five-minute tuple exactly equalled first/max/min/last of the one-minute sequence.

OI remains unsigned. It says exposure is opening or closing, not whether longs or shorts dominate.

Core derived fields:

```text
oi_log_change_h
oi_log_change_6h
oi_log_change_24h
oi_range_fraction
oi_notional_usd_proxy = oi_close_base_units * mark_close
```

Raw cross-symbol OI must not be compared without normalization.

### Liquidation volume

Adopt:

```text
liquidation_base_units = Decimal(value_raw)
aggregation = interval sum
direction = unsigned
```

The series is an interval flow: all values are nonnegative and five-minute values equal the sum of their one-minute components.

Core derived fields:

```text
liquidation_to_lagged_oi =
    liquidation_base_units / lagged_oi_close_base_units

liquidation_mark_notional_usd_proxy =
    liquidation_base_units * mark_close
```

`liquidation_to_lagged_oi` should be the primary cross-symbol intensity measure because the inferred unit cancels.

Long-versus-short liquidation is not present in the source. A price-aligned label may be derived only as:

```text
price_inferred_liquidation_side
```

and must remain a proxy. This dataset does not become a typed liquidation tape.

## What the new data genuinely unlock

### Priority 1: KDA01 / C04 — crowding versus price progress

Mechanism:

> Additional open exposure and futures premium continue to build, but each new unit of crowding produces less price progress. The crowded side becomes vulnerable only after the structure completes a causal failure.

Compelled actor:

- leveraged traders must add, hold, reduce, or liquidate;
- basis-sensitive arbitrageurs respond to premium;
- once marginal leverage stops moving price, modest opposing flow can trigger a reduction cascade.

Distinctness:

- OI and basis define the parent state;
- completed failure is confirmation, not the whole hypothesis;
- this is not a high-funding short;
- it is not a renamed RFBS, Backside, or generic failed breakout.

Required first controls:

1. completed failure without OI/basis crowding;
2. OI change alone;
3. basis level/change alone;
4. price-progress deterioration alone;
5. matched price failures with ordinary OI;
6. parent BTC/ETH state;
7. same event definition with liquidation omitted.

Main falsification:

> The OI/basis-to-price-progress interaction adds no stable information beyond completed failure or price movement alone.

Recommended initial horizons:

```text
parent-state windows: 1h and 6h
economic horizons later: 1h, 6h, and at most 24h
```

Do not begin with dense parameter grids.

### Priority 2: KDA02 — liquidation/OI purge state machine

Mechanism:

> Liquidation flow is forced rather than discretionary. Price can continue while forced flow is active, but a large liquidation burst combined with a material OI reset may mark completion, absorption, or an OI vacuum.

The first research contract should distinguish:

```text
active purge continuation:
    liquidation intensity elevated
    OI has not completed a material reset
    price continues to make progress

completed purge reversal/reclaim:
    liquidation intensity elevated
    OI materially contracts
    price completes a causal reclaim/failure after the burst

OI-vacuum continuation:
    OI contracts sharply
    price displacement is modest
    surviving directional pressure remains
```

Required controls:

1. same price shock without elevated liquidation;
2. OI contraction without liquidation;
3. liquidation burst without OI reset;
4. ordinary high-volume candle;
5. completed price reclaim without analytics confirmation;
6. BTC/ETH-wide stress matched by time.

Main falsification:

> OI reset and liquidation intensity do not separate continuation from completion beyond the price path itself.

This is the legitimate derivatives-state successor to the earlier delayed-flush idea. The old chart-only translation did not test forced deleveraging; the new mechanism does.

### Priority 3: KDA03 / C08 — basis state and dislocation

Mechanism:

> A basis shock can represent local leveraged demand, stress, or temporary dislocation. Its meaning depends on OI, price response, and liquidation state.

Useful branches:

- basis expansion + OI growth + efficient price progress: possible continuation;
- basis expansion + stalled price + rising OI: crowding fragility;
- extreme basis + liquidation/OI reset: convergence/completion question;
- basis shock without spot/mark confirmation: dislocation screen.

This family should remain second-wave because historical spread/depth data are still absent. A bar-level result can support a bounded state study, not a high-fidelity arbitrage claim.

## Older hypotheses: permitted and prohibited use

| Older object | New-data use | What remains prohibited |
|---|---|---|
| RFBS / Backside / failed-breakdown | Controls and overlap tapes for KDA01 failure confirmation | Retuning the old thresholds or selecting OI filters from old winners |
| Delayed-flush reclaim | Re-express the broad mechanism as KDA02 with actual liquidation/OI state | Calling the old price-only tape forced deleveraging evidence |
| C01 residual/path | Parent-market or residual control; episode overlap | OI/basis filtering of C01 losers to rescue the closed translation |
| C02 spot-led | Spot-lead control where same-asset spot exists | Reopening C02 because OI might explain the failed trades |
| TSMOM | Market-weather/context control | New promotion from OI-conditioned subsets on the inspected tape |
| A1/compression | Price-path control | Rebranding smooth crowded continuation as A1 |
| H43/BTC diffusion | Parent-shock and market-wide stress control | Residual-laggard selection |
| Relative strength/prior high | Continuous diagnostics only | New hard-gate threshold searches |
| C2 catalysts | Later pre-event positioning and post-event forced-flow context | Hindsight event-quality labels or timestamp imputation |

The guiding rule is simple:

> Build the new derivatives-state hypothesis independently across all eligible events. Only then compare overlap with older families.

## Research sequence

### Stage 8A — semantics, feature foundation, and outcome-free generators

Implement now:

1. machine-readable semantic contract and fail-closed tests;
2. exact trade/mark/analytics alignment;
3. causal shared derivatives-state feature cache;
4. coverage and cohort authority;
5. family and attempt registration;
6. outcome-free KDA01 and KDA02 event tapes;
7. KDA03 feasibility/state inventory;
8. counts by year, symbol, direction, major/alt, and state;
9. canonical episode identities;
10. overlap with old families using safe identity fields only;
11. one frozen recommendation for no more than two later economic contracts.

No forward returns are permitted.

### Stage 8B — contract freeze

After reviewing Stage 8A:

- freeze definitions, thresholds, entries, exits, timeouts, costs, controls, multiplicity, and kill gates;
- ensure the exact distinct component is testable;
- do not calculate outcomes.

### Stage 8C — Level-3 economics

Only after explicit human approval:

- run one mechanism at a time;
- test all frozen branches;
- stop failed translations;
- run controls only for definitions that pass the predeclared Level-3 gates.

### Stage 8D — controls and interactions

Only after a primary definition passes:

- test price-only, OI-only, basis-only, liquidation-only, and matched-event controls;
- then test whether an older price component adds incremental information;
- do not use old families to discover the new rule.

## One-minute expansion

Do not download one-minute data for all 460 identities now.

The present allocation is efficient:

```text
all supported identities: 5m
BTC/ETH: 1m
```

Use Stage 8A to determine whether five-minute timing creates a material ambiguity. A later expansion is justified only if:

- event ordering or burst completion changes materially inside a five-minute bucket;
- the first economic contract depends on sub-five-minute sequencing;
- the target symbols have sufficient activity and coverage.

Preferred next one-minute scope:

```text
frozen top 50–100 liquid crypto perpetuals
```

chosen before outcomes using prior-day liquidity and coverage. An alternative is one-minute acquisition only around a frozen event list, but the event list must be created without outcome inspection.

All-symbol one-minute acquisition is feasible in principle, but its marginal value for thin contracts is unlikely to justify the larger request count and maintenance burden before a five-minute mechanism survives.

## Pre-2023 data

Do not change the rankable interval.

The claim that pre-2023 crypto was not dominated by algorithms is too broad to use as an authority rule. The stronger reasons to keep pre-2023 outside selection are:

- venue and product structure changed;
- contract coverage and lifecycle authority are weaker;
- 2020–2022 contain distinct crisis and boom regimes;
- the current programme already protects a consistent 2023–2025 research interval.

Older data can later support:

- mechanism history;
- engineering validation;
- a separately registered stress-regime robustness study after rules are frozen.

It should not be added now for parameter selection.

## Immediate action

Send the accompanying Stage 8A prompt to the backtesting agent. It authorizes a comprehensive non-economic implementation and the reviewed repository/Drive handoff, but no economic run.
