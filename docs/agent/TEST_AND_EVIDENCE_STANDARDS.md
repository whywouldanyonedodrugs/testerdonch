---
status: active testing manual after verified Stage 10 policy application
date: 2026-07-19
revision: 3.0
scope: multi-platform evidence, conditional alpha, gate routing, controls, reproducibility, validation, and deployment
authority: operating contract, machine contracts, finalized manifests, and explicit 2026-07-19 human policy
supersedes: received `12_MANUAL_Test_and_Evidence_Standards_2026-07-19_rev3.md`; prior manuals remain provenance
provenance: rev2; Stage 7C-9; aggressive conditional-alpha policy
known limitations: no strategy is validation-grade or live-ready
---

# Test and Evidence Standards

## Fixed boundaries

```text
rankable_interval: [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
protected_period_start: 2026-01-01T00:00:00Z
paid_historical_vendor_data: prohibited
orders_private_account_actions: not_authorized
```

A rankable runner fails closed before payload load when platform, purpose, interval, schema, content identity, or protected status cannot be proven.

## Separate status dimensions

```text
hypothesis_status
evidence_level
reproducibility_status
validation_status
deployment_status
research_route
```

`research_route` does not replace evidence level.

## Required economic contract

Every contract freezes:

```text
hypothesis_and_translation_id
mechanism_and_compelled_actor
platform_and_instrument
PIT universe and lifecycle claim
source availability and decision time
entry/exit and executable price fields
fees/funding/financing/slippage
payoff_archetype
intended_claim_scope
primary inference estimand
controls and ablations
multiplicity family
protected policy
hard_gates
routing_diagnostics
control_eligibility
independent_evidence_requirement
falsification
```

## Integrity gates

Always hard:

- authority and hashes;
- causal availability;
- protected firewall;
- deterministic identities and replay;
- exact execution fields;
- no outcome-derived contract mutation;
- independent review.

## Payoff archetypes

Every contract declares one before outcomes:

```text
symmetric_directional
mean_reversion
convex_right_tail
carry_or_yield
event_driven_sparse
execution_or_microstructure
```

Median, tail and concentration interpretation must follow the frozen archetype.

## Level-3 research routes

### Unconditional control candidate

Broadly positive base economics with adequate independent coverage and no major generality flag.

### Conditional context candidate

Positive economic relevance but temporal, regime, symbol, or breadth concentration limits an unconditional claim. Context controls are permitted; independent evidence remains required.

### Convex tail candidate

Positive base mean under a predeclared convex archetype despite a nonpositive median. Requires tail recurrence, left-tail and drawdown analysis, and prospective evidence.

### Execution-sensitive candidate

Gross mechanism evidence is positive but plausible execution/cost scenarios determine viability. Requires an externally justified execution contract.

### Sample-limited candidate

Coherent but too few independent clusters for broad inference. Preserve for prospective observation or a bounded event sleeve.

### Translation rejected

No meaningful positive economic component under the intended implementation, or mechanism controls later falsify the distinct component.

### Mechanically/data unavailable

Required data, opportunities, or authority are insufficient. This is not negative economics.

## Gate interpretation

- Positive base mean is the default economic-relevance gate.
- Positive median is required for symmetric and mean-reversion contracts; it is diagnostic for frozen convex contracts.
- Bootstrap lower bounds classify uncertainty. They may route a positive object to controls or prospective evidence rather than universally kill it.
- Market-day, symbol and year concentration define claim scope.
- Year concentration above 70% blocks an unconditional claim but routes a positive object to context research.
- Per-year minimum counts apply to an unconditional multi-year claim, not automatically to sparse event sleeves.
- Stress results define execution fragility; base costs must remain realistic.

Exact numeric rules remain serialized in each contract. There is no universal right to weaken a gate after outcomes.

## Controls

Controls test the claimed mechanism. They may run for unconditional, conditional-context, convex-tail, execution-sensitive, or sample-limited candidates when integrity and minimum economic-relevance gates pass.

Eligibility does not authorize execution. Every control or ablation run requires separate exact task authorization, and the frozen contract must distinguish `control_eligibility` from that run authority.

Controls do not validate a candidate. A control phase must pre-register:

- price/path-only control;
- component removals;
- matched non-events;
- common-factor and timing controls;
- episode and context clustering;
- no caliper widening after outcomes.

## Post-hoc context

Outcome-derived context hypotheses are allowed only as explicitly post-hoc research presence. The same sample may support explanation and control design, not final validation.

A valid context is observable at decision time and is not a calendar label.

## Reproducibility

Record:

- commit and dirty state;
- code/config/data/universe/cost hashes;
- exact commands and exit codes;
- input and protected audits;
- event/control/episode/cluster identities;
- all attempted definitions and killed branches;
- artifact manifest;
- independent review;
- old and superseding decisions.

## Validation and deployment

No route becomes validation-grade without independent or prospective evidence. No route becomes live-ready without calibrated execution, drawdown, risk of ruin, capacity, context-disable rules, and portfolio interaction review.

Aggressive risk tolerance affects sizing after evidence; it does not lower integrity or validation standards.
