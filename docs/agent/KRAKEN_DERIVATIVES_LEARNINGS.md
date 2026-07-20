---
status: active consolidated Kraken learning state after Stage 19 official funding authority
date: 2026-07-20
revision: 1.2
scope: Stage 7C analytics, Stage 8A foundation, KDA01-KDA03, reusable components, defects, and future routes
authority: finalized manifests, reviewed task archives, published commits, and terminal run decisions
supersedes: prior narrative statements that 2023-2025 OI, basis, and liquidation analytics were unavailable; does not supersede finalized manifests or roots
provenance: Stage 7A-7C; Stage 8A-8C1; Stage 9; current operating contract and family decisions
known limitations: analytics units are inferred; universe is current-roster/lifecycle capped; no strategy is validation-grade or live-ready
---

# Kraken Derivatives Analytics and Research Learnings

## Stage 13 readiness note

The outcome-free Stage 8 OI-vacuum onset matrix contains 1,176,354 events across 186 symbols: 143,358 in 2023, 455,948 in 2024, and 577,048 in 2025. These are false-to-true onsets, not durations or outcomes. KDA02C breadth and KDX01 joint-state incidence remain unmaterialized; registration supports no economic conclusion.

## Current state

Kraken historical analytics are now acquired and source-authorized for the rankable period:

```text
normalized rows: 229,869,200
five-minute analytics: broad frozen PF inventory
one-minute analytics: PF_XBTUSD and PF_ETHUSD
metrics: open interest, liquidation volume, future basis
protected rows: 0
```

The authoritative local analytics manifest content hash is:

```text
f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d
```

The acquired universe remains current-roster/bar-existence based. It is not survivorship-free and does not prove uninterrupted historical tradeability.

## Semantic decisions

### Future basis

- signed closing state;
- positive means futures above spot/reference;
- negative means below;
- internally treated as a decimal ratio;
- five-minute value equals the final one-minute value in the bucket;
- use within-symbol causal changes, robust z-scores, and percentiles;
- basis is not funding.

### Open interest

- four-value tuple is structurally OHLC;
- value is treated as instrument base/contract quantity;
- unsigned gross open exposure;
- no long/short direction from OI alone;
- cross-symbol use requires normalization or mark-notional proxy.

### Liquidation volume

- nonnegative interval flow;
- five-minute value is the sum of one-minute values;
- treated as base/contract quantity;
- no native long/short side;
- primary cross-symbol intensity is liquidation divided by lagged OI;
- price-inferred side remains a proxy.

## Engineering lessons

### Absolute funding source layout

Kraken linear perpetual funding is a continuous USD cashflow: the absolute rate is USD per base-unit contract per hour, and an open position accrues it in proportion to time held. The Stage 18 Parquet authority remains unusable because all 305 row groups are mixed or protected. Stage 19 established a separate human-transferred official-export authority with immutable rankable, protected, pre-rankable, and invalid packages. Campaign access is limited to the `[2023,2026)` package; protected rate tails were routed without numeric conversion.

Export row timestamps remain ambiguous between `[t,t+1h)` and `[t-1h,t)`. Do not choose by inference. Compute both and use `min(0,signed_start,signed_end)` for selection. Missing export hours receive nonpositive Decimal type-7 q95/q99 allowances calibrated on rankable rows only. Relative funding, mark price, trade price, or future basis must not replace the exact absolute cashflow; `absolute_rate/relative_rate` is used only as a unit sanity anchor.

### Timestamp semantics

A confirmed defect showed that completed five-minute bars were initially entered one bar too late.

Correct convention:

```text
stored candle time: interval start
decision time: interval start + five minutes
entry: first executable bar open at or after decision time
```

Future tasks must test this explicitly. Repository-wide audit is appropriate before selectively rerunning an older family.

### Point-in-time normalization

Stage 8A review found and repaired a same-day lookahead defect where an entire day's aggregate could affect earlier intraday rows.

Rule:

```text
intraday row scoring uses a distribution ending on the prior UTC day
```

Future-row invariance tests are mandatory.

### Contiguous windows

Lagged features must verify every interior interval. Endpoint comparison alone is insufficient. Missing five-minute bars fail the feature closed.

### Scheduler identity

Economic schedulers must consume only identities in the frozen definition register. Broader feasibility tables may contain omitted or killed branches and cannot define the run universe.

### Inference clusters

Symbol-specific episodes prevent duplicate positions but do not make cross-symbol events independent. Market-day and shorter market-wide clusters must be available before outcomes.

Trade-weighted, market-day-weighted, six-hour-cluster, and episode-weighted estimands answer different questions and must be labelled separately.

### Handoff completeness

A compact package must include:

- the binding artifact manifest;
- identities of binding year, symbol, market day, and episode contributors;
- complete gate matrix;
- frozen contracts and definition register;
- superseded/failed attempts.

A ratio without the contributor identity is insufficient for later mechanism interpretation.

## KDA01 decision and lessons

```text
family: KDA01_v2_episode_level_crowding_price_progress_bifurcation
terminal decision: KDA01_level3_repaired_no_primary_pass_stop
```

### What was tested

- material OI expansion;
- directionally coherent extreme basis;
- price progress relative to OI expansion;
- efficient continuation;
- completed trade-and-mark structural failure;
- both directions;
- one-hour and six-hour timeouts.

### Findings

1. Efficient crowded continuation was materially negative before ordinary costs, especially over six hours.
2. The state likely identified mature or exhausted crowded moves after an already completed one-hour advance.
3. Completed-failure branches contained modest gross information, but did not clear the 14-bps steady-state cost contract.
4. The five-minute timestamp repair improved average gross return by about 1.18 bps but did not change the terminal decision.
5. Basis level and basis change are different hypotheses. A future buildup/acceleration mechanism would need a new identity and independent evidence.
6. KDA01 cannot be reopened through costs, opposite direction, thresholds, years, symbols, or horizons.

## KDA02 decision and lessons

```text
family: KDA02A_v2_liquidation_oi_purge_state_machine
terminal decision: KDA02_level3_no_primary_pass_stop
```

### Mechanically feasible branches

Only completed-purge reversal branches had enough annual coverage. Active-purge continuation was mechanically sparse and was not economically rejected.

### Directional asymmetry

Negative purge followed by completed reclaim behaved much better than the positive-purge analogue.

Interpretation prior:

```text
downside long-liquidation proxy:
    possible overshoot and slower recovery after OI reset

upside short-liquidation proxy:
    apparent structural failure did not support stable short expectancy
```

Because liquidation side is price-inferred, this asymmetry may reflect both real market mechanics and proxy error.

### Horizon

The useful negative-purge reversal effect appeared at six hours, not one hour. The mechanism resembled post-deleveraging recovery rather than an immediate snapback.

### Strongest branch

Primary negative completed-purge reversal, six hours:

```text
accepted trades: 2,643
2023/2024/2025 trades: 101 / 1,063 / 1,479
symbols: 54
market-day clusters: 651
trade-weighted gross mean: +16.1340 bps
trade-weighted base mean: +2.1340 bps
equal-market-day gross mean: +28.4404 bps
equal-market-day base mean: +14.4404 bps
equal-market-day base median: +10.3686 bps
bootstrap lower: -2.6700 bps
stress mean: -3.5596 bps
market-day positive contribution: 2.08%
symbol positive contribution: 11.65%
positive-year contribution: 86.10%
```

It failed only the frozen unconditional positive-year contribution gate.

The original run decision remains binding. Under the aggressive conditional-alpha policy, this object is preserved as:

```text
conditional_context_candidate_unvalidated
post_hoc_context_hypothesis
```

It may not be traded or promoted from the same sample. A future task may ask whether a causal context such as isolated versus systemic purge breadth explains the concentration.

### Estimand clue

For the strongest branch:

```text
parent-episode base mean: +2.1340 bps
six-hour market-cluster base mean: -1.9016 bps
equal-market-day base mean: +14.4404 bps
```

This suggests a possible distinction:

```text
isolated or low-breadth purge:
    more likely to mean-revert

broad systemic cascade:
    more likely to retain pressure
```

This is outcome-derived hypothesis presence, not evidence. Register it separately and require independent validation.

### KDA02B OI vacuum

The Stage 8A OI-vacuum branch contained no liquidation requirement and was separated into a distinct outcome-free candidate-library lineage. It remains untested.

## Stage 11 KDA03 result

KDA03 tested three prospectively frozen basis-shock mechanisms in separate positive and negative directions at one- and six-hour timeouts. Eleven of the twelve primary definitions were `translation_rejected` after 14 bps round-trip costs.

The sole non-rejected primary definition was negative completed-basis impulse rejection at six hours:

```text
equal-market-day gross mean: +23.1570 bps
equal-market-day base mean: +9.1570 bps
equal-market-day base median: +2.9323 bps
bootstrap lower bound: -8.2953 bps
stress mean: -8.8430 bps
```

Its route is `sample_limited_prospective_candidate`, not a pass or production claim. It is not control-eligible under the frozen policy. No same-sample threshold, context, horizon, robustness, or control work may rescue it; the next permitted evidence is independent or prospective validation of that exact frozen candidate.

The limitation was not a literally tiny raw trade count: 1,839 accepted trades spanned 691 market-day clusters, 1,839 parent episodes, and 49 symbols. The problem was weak precision and threshold robustness. Separate evidence-limitation tags are `high_variance`, `wide_cluster_uncertainty`, `threshold_sensitive`, and `not_control_eligible`; they do not alter its route or evidence level.

Stage 11 also established method lessons: most definitions were near-flat gross before the frozen 14-bps cost; standardized extremes lacked a minimum raw-magnitude condition; prior-day normalization can make intraday “extremes” broad; the claimed shock state was very frequent; KDA03A lacked a directly observed complete executable reference leg; OI did not identify actor direction; and positive/negative shocks were asymmetric. The formal response is the staged development protocol, not a KDA03 rerun.

The evidence cohort is also a binding claim cap: the Kraken K0 trade manifest contains 301 PF symbols, while Stage 8A authorized a 187-symbol current-roster/lifecycle-capped intersection and only 55 generated KDA03 candidates after causal coverage and onset gates. This is not a survivorship-free historical universe.

## Reusable cross-family components

### Completed failure

Completed trade-and-mark structural failure remains the most reusable price component. It must be attached to a defined parent mechanism and tested against the same parent without failure confirmation.

### Derivatives-state interactions

OI, basis, or liquidation alone do not provide direction. Their useful role is an interaction with:

- price progress;
- OI reset;
- structural completion;
- market-wide breadth;
- spot/reference confirmation;
- execution state.

### Directional branches

Positive and negative states must remain separate. Symmetry should not be assumed.

### Context before hard gate

Market breadth, purge breadth, BTC/ETH state, liquidity rank, and basis/OI context should usually begin as continuous or categorical diagnostics. They become entry gates only after a frozen component test.

## Gate-policy lesson

The previous universal all-gates rule was appropriate for unconditional promotion but too restrictive for research routing.

A positive-year contribution above 70% means:

```text
unconditional claim failed
```

It does not necessarily mean:

```text
all conditional economic value is absent
```

Future contracts must separate:

- integrity;
- economic relevance;
- generality/context;
- controls;
- independent validation;
- deployment.

## Candidate library after Stage 9

| Candidate | Status | Next permitted route |
|---|---|---|
| KDA01 v2 | Exact translation closed | No same-sample work. |
| KDA02A v2 unconditional | Exact translation closed | No threshold/direction/horizon rescue. |
| KDA02 negative reversal 6h conditional | Post-hoc context candidate, unvalidated | Context/control design, then independent evidence. |
| KDA02 active purge continuation | Mechanically unavailable under strict 5m contract | Preserve; possible future one-minute feasibility only if justified. |
| KDA02B OI vacuum | Outcome-free candidate library | Fresh independent contract if prioritized. |
| KDA02C isolated-versus-systemic purge breadth | Post-hoc hypothesis presence | No same-sample promotion; context study and prospective validation. |
| KDA03 basis-shock family | Level-3 routes assigned; 11/12 primary definitions rejected | Negative completed-basis rejection 6h is sample-limited and requires independent or prospective evidence. |

## Hypothesis-development method after Stage 12

Future families begin with a compelled actor or structural balance sheet and outcome-free measurement validation. Raw magnitude, event frequency/duration, reference authority, exact feature windows, timestamp availability, and plausible cost break-even must be established alongside standardized scores. Any development-fold returns require separate exact human approval, all explored cells count toward multiplicity, and the frozen translation is evaluated only on the next untouched block. Later-fold information cannot flow backward.

## One-minute expansion

Do not download all-symbol one-minute data automatically.

Potentially justified future scope:

- top 50-100 liquid contracts chosen before outcomes;
- event-window acquisition around a frozen candidate tape;
- active-purge sequencing where five-minute aggregation causes mechanical sparsity.

BTC/ETH one-minute data should first test whether the active-purge state materially changes at one-minute resolution.

## Pre-2023 data

Keep `[2023-01-01, 2026-01-01)` as the rankable interval. Pre-2023 data may support mechanism history or a separately registered stress-regime robustness study after rules are frozen. It should not choose current features, thresholds, or priorities.
