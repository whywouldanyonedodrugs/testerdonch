# Next Hypothesis-Development Candidates

Status: non-authorizing, outcome-free comparison. No ordering below uses new or protected returns.

All candidates must start at Phase 0/1 of the [`Hypothesis Development Protocol`](docs/agent/HYPOTHESIS_DEVELOPMENT_PROTOCOL.md). Phase 2 or later requires separate exact human approval.

## KDA02C — isolated versus systemic purge breadth

- Mechanism: an isolated deleveraging shock may exhaust local forced selling and reverse, while a broad cascade may retain market-wide pressure.
- Compelled actor: liquidated or margin-constrained longs plus risk managers reducing correlated exposure; actor side remains price-inferred.
- Available data: pre-existing causal liquidation/OI state, PF trade/mark bars, and cross-symbol event identities on the Stage 8A capped cohort.
- Semantic weaknesses: post-hoc origin; liquidation side proxy; lifecycle-capped breadth denominator; five-minute sequencing; no survivorship-free universe.
- Measurement validation: define PIT breadth denominator, native liquidation/OI magnitude, simultaneous episode clustering, frequency/duration, missing-symbol behavior, and cost break-even before outcomes.
- Development-fold design: predefined chronological development folds; continuous breadth response first, with all breadth cells registered; next-block freeze/evaluation only afterward.
- Main controls: same parent purge without breadth conditioning, matched non-purge breadth, BTC/ETH-only versus broad breadth, and price-path-only control.
- Same-sample restrictions: Stage 9 outcomes may explain provenance only; they cannot choose a breadth threshold, calendar context, horizon, symbol set, or direction.
- Priority rationale: high mechanism relevance and existing causal components, but post-hoc contamination makes untouched forward evidence essential.

## KDA02B — OI vacuum without liquidation

- Mechanism: rapid OI contraction without observed liquidation may represent voluntary risk withdrawal or hedge closure, changing continuation/reversal pressure differently from forced purge.
- Compelled actor: discretionary leverage reducers, hedgers, and arbitrage balance sheets; current observables do not identify their mix.
- Available data: causal OI, basis, trade, and mark features already present in the outcome-free Stage 8A lineage.
- Semantic weaknesses: OI is unsigned; early history is truncated; “without liquidation” is limited by liquidation measurement/side semantics; actor direction is underidentified.
- Measurement validation: raw OI notional change, basis co-movement, event frequency/duration, zero-versus-missing liquidation distinction, liquidity cohorts, and cost break-even.
- Development-fold design: characterize continuous OI contraction magnitude crossed with basis/price progress in registered chronological folds; freeze sign/horizon only after development.
- Main controls: matched OI-stable episodes, liquidation-present contraction, price-move-only control, basis component removal, and same-time market-state control.
- Same-sample restrictions: no inference from KDA02A outcomes and no reuse of its winning-looking direction/horizon.
- Priority rationale: clean outcome-free lineage and inexpensive measurement work, but Phase 0 may end `mechanism_underidentified` unless actor/structural direction is sharpened.

## Downside completed derivatives-state rejection

- Mechanism: after a causal downside leverage/dislocation state completes, trade and mark reclaim may indicate forced supply exhaustion and balance-sheet normalization over a slower horizon.
- Compelled actor: liquidated longs and constrained sellers cease supplying; arbitrageurs or unlevered buyers absorb dislocation after completed reclaim.
- Available data: only pre-existing causal components—liquidation/OI purge state, basis state/change, PF trade/mark completion, and existing market clusters.
- Semantic weaknesses: combining components can disguise reuse of KDA02/KDA03 outcomes; OI and liquidation side remain incomplete proxies; no complete executable reference panel.
- Measurement validation: independently validate each component’s native magnitude, actor consistency, joint-state frequency/duration, overlap with prior economic addresses, reference limitations, and break-even magnitude.
- Development-fold design: pre-register a small component-incrementality matrix in development folds; freeze a single direction, form, horizon, controls, and multiplicity family for the next untouched block.
- Main controls: parent derivatives state without completed reclaim, completed reclaim without derivatives state, single-component ablations, and matched downside price paths.
- Same-sample restrictions: KDA02/KDA03 return signs and six-hour results cannot set the new horizon, weights, thresholds, or selection rule; duplicate economic addresses must remain provenance, not fresh evidence.
- Priority rationale: strongest cross-family causal synthesis, but highest contamination risk; it should proceed only if the fresh contract and actor story can be shown without outcome-derived choices.

## C17 — executed catalyst state × leader/breadth

- Mechanism: an effective change in access, legality, supply, fees, utility, or distribution forces repricing; continuation requires realized state and participation, while failed realization can trap anticipatory positioning.
- Compelled actor: users, issuers, market makers, hedgers, and holders compelled by the effective rule or supply/access change rather than by derivatives positioning.
- Available data: current source-verified catalyst seed plus Kraken price/mark bars and existing leader/breadth building blocks; the catalyst set is not a closed census.
- Semantic weaknesses: publication/effective timestamps, revisions, event completeness, basket identity, subtype heterogeneity, and historical source continuity remain material.
- Measurement validation: independently verify first-public/effective times, event coverage, eligible symbols, mechanism subtype, participation/breadth frequency, native price magnitude, and executable cost break-even.
- Development-fold design: split by event chronology before outcome access; characterize executed versus announced state and continuous participation in development events, then freeze one subtype/translation for untouched later events.
- Main controls: matched non-event dates, announcement versus execution, ordinary trend/breadth, failed-catalyst branch, subtype removal, and cluster-level independence.
- Same-sample restrictions: no retuning of prior C2 episodes, hindsight event-quality labels, imputed timestamps, best subtype selection, or reuse of protected/post-2025 events.
- Priority rationale: genuinely non-derivatives and actor-led, offering diversification from KDA families, but its non-closed event census and timestamp burden place measurement authority ahead of economics.

## Outcome-free sequencing recommendation

Phase 0/1 readiness work can compare all four without ranking returns. KDA02B is the least contaminated technically; KDA02C has the clearest immediate control question but is post-hoc; the combined downside rejection mechanism has the strongest synthesis and the greatest reuse risk; C17 is the freshest family and the most data-authority intensive. This is an implementation/readiness comparison, not an alpha ranking or economic-run approval.
