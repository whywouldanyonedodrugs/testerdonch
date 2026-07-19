---
status: proposed current research-search method; verified application pending
date: 2026-07-19
revision: 1.0
scope: intelligent hypothesis construction, response surfaces, nested walk-forward search, candidate beams and multiplicity
authority: profitability-first policy and test/evidence standards; below machine contracts
supersedes: one-shot hand-written translation as the default research method
provenance: Stage 12 protocol v1.0; KDA01-KDA03 design lessons; prior research-efficiency method
known limitations: historical folds are programme-exposed and do not constitute independent validation
---

# Nested Hypothesis Search, Response Surfaces and Multiplicity

## Core design

The preferred workflow is:

```text
mechanism primitives
-> outcome-free measurement validation
-> registered response-surface development
-> sparse candidate beam
-> next-block evaluation
-> rolling forward replication
-> controls and prospective evidence
```

The purpose is to discover a good definition without allowing the discovery sample to masquerade as confirmation.

## Search lanes

A campaign may use one or more declared lanes.

### Lane A — descriptive response surface

For each causal feature and horizon, report:

- raw-unit and normalized bins;
- event/cluster count;
- gross and net response;
- monotonicity and sign stability;
- uncertainty by development fold;
- cost break-even.

Use continuous ranks or bins before hard thresholds.

### Lane B — component-incrementality ladder

Test nested components:

```text
price/path only
single causal variable
pairwise interaction
full proposed state
full state plus confirmation/context
```

A component belongs in the final rule only when it improves out-of-fold utility or clearly changes the mechanism state.

### Lane C — sparse model-assisted discovery

Permitted development-only models:

- regularized linear or logistic models;
- sparse generalized additive models;
- shallow trees or rule lists with fixed maximum depth;
- monotonic models where mechanism direction is known;
- stability selection across development folds.

Models discover response shape and interactions. They do not become execution rules until converted into a small, inspectable, frozen contract.

### Lane D — constrained rule/episode grammar

Search over predeclared mechanism primitives and state transitions with:

- maximum interaction depth;
- maximum number of conditions;
- fixed episode/reset grammar;
- complexity penalty;
- all generated rules retained.

### Lane E — horizon and payoff profile

Profile a fixed development set of horizons such as:

```text
15m, 30m, 1h, 2h, 4h, 6h, 12h, 24h
```

The campaign may narrow this set when market mechanics justify it. Selected horizons must be frozen before the next evaluation block.

## Fold design

Use forward-only rolling origin.

Default Kraken analytics schedule, subject to data availability:

```text
first usable date: 2023-04-01
first evaluation block: 2023-Q4
subsequent evaluation blocks: each calendar quarter through 2025-Q4
development window: expanding history ending before the evaluation block
inner selection: prior monthly or bimonthly subfolds
purge/embargo: at least maximum label horizon and episode overlap
```

Each campaign serializes its exact dates.

Data-exposure labels:

```text
campaign_sealed_outer_fold
program_exposed_historical
protected_prospective
```

A campaign-sealed fold improves internal evidence. It is not independent validation when the programme has previously inspected related outcomes.

## Search-space construction

Search spaces derive from measurement validation and mechanism primitives, not arbitrary indicator catalogues.

For every feature specify:

```text
raw unit
causal transformations
allowed lookbacks
allowed ranks or thresholds
expected direction if any
missingness contract
interaction partners
reason the feature belongs
```

Default threshold candidates may use economically meaningful raw levels and a small registered set of empirical quantiles. Do not search every decimal threshold.

## Candidate beam

Do not force one guessed rule or retain hundreds of correlated variants.

Default development beam:

```text
maximum candidates retained per family per fold: 5
maximum candidates sent to one outer evaluation block: 3
```

A campaign may set smaller values.

Use a Pareto frontier across:

- out-of-fold base net expectancy;
- worst and median inner-fold performance;
- cluster uncertainty;
- drawdown or left tail;
- opportunity frequency and capital occupancy;
- execution margin;
- complexity;
- correlation with retained candidates.

The deterministic tie-break is serialized before returns.

## Multiplicity

Every explored response cell, feature subset, model, rule, horizon, direction and context is registered.

Use three layers:

1. **Selection protection:** nested or rolling outer evaluation not used in candidate construction.
2. **Family-level comparison:** max-statistic/block-bootstrap, SPA or equivalent dependence-aware correction across the frozen candidate beam.
3. **Programme-level record:** effective trial count and lineage across prior translations and related families.

Discovery p-values are descriptive. Selection is based on economic utility and fold stability, not the smallest unadjusted p-value.

## Admission to an outer evaluation block

A candidate may advance when:

- integrity and measurement gates pass;
- raw magnitude is relevant relative to plausible costs;
- enough independent events/clusters exist for the intended sleeve;
- development evidence shows positive net utility or a predeclared convex profile;
- the response is not solely one unregistered event, symbol or technical artifact;
- the candidate is within the registered beam and complexity budget.

Year uniformity, positive median and severe-cost results are interpreted according to payoff archetype and intended claim; they are not universal admission gates.

## Redefining old families

Previously failed families may enter this protocol with all old attempts registered. Their historical outcomes can be used only as programme-exposed development evidence. A new translation must be tested forward through the campaign folds and later prospectively before validation.
