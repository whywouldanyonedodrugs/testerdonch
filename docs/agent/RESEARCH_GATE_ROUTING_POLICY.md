---
status: approved policy; verified repository application active
date: 2026-07-19
revision: 1.0
scope: research objectives, evidence gates, conditional sleeves, control routing, and small-account opportunity capture
authority: explicit human decision of 2026-07-19, subordinate to machine contracts and finalized run evidence
supersedes: universal binary use of Level-3 stability/concentration gates for future research routing; does not alter historical run decisions
provenance: current operating contract; first-wave review; Stage 7C analytics acquisition; Stage 8A-8C1 KDA01; Stage 9 KDA02; prior research audit
known limitations: no current strategy is validation-grade or live-ready; post-hoc conditional hypotheses require independent evidence; this policy does not authorize an economic run
---

# Aggressive Conditional-Alpha and Gate-Routing Policy

## Decision

The programme is aggressive in **opportunity capture and research breadth**, not careless about evidence.

The objective is not to find one rule that works in every year, symbol, and market state. It is to build a portfolio of distinct, causally defensible sleeves that may each operate only in identifiable contexts.

A result is not rejected merely because:

- one year or regime supplies most positive PnL;
- the median is negative under a predeclared convex payoff design;
- the sample is narrow but economically coherent;
- severe-cost stress is negative while a realistic base implementation is positive;
- an edge is concentrated in a named instrument or event class that can be identified before entry.

These conditions limit the claim and determine the next research route. They are not automatically evidence of overfitting.

## What remains non-negotiable

The following are hard gates and always fail closed:

1. source, purpose, interval, schema, and hash authority;
2. zero protected-period selection leakage;
3. causal feature and decision availability;
4. deterministic candidate, event, control, episode, and run identity;
5. execution fields and costs tied to the actual platform contract;
6. reproducible arithmetic, manifests, tests, and independent review;
7. no outcome-conditioned rule, direction, threshold, symbol, horizon, or context change presented as independent evidence;
8. no same-sample rescue of a closed translation through renaming;
9. no deployment claim without independent evidence and risk review.

Aggressiveness never relaxes these gates.

## Gate classes

### A. Integrity gates: hard stop

Examples:

- unprovable or mixed source;
- lookahead or timestamp ambiguity;
- protected rows;
- unreconciled identities;
- missing executable prices;
- deterministic replay failure;
- secret or private-account exposure;
- material code or data defect.

Failure means `blocked` or `invalid_run`, not a strategy conclusion.

### B. Economic-relevance gates: determine whether further research is worth doing

Core questions:

- Is gross expectancy directionally meaningful?
- Is base-cost expectancy positive under the declared implementation?
- Is the result supported by enough independent events or clusters for the intended claim?
- Is the loss distribution compatible with the predeclared payoff archetype?
- Is the effect materially larger than known execution uncertainty?

A candidate with no positive economic relevance is closed at the exact translation. A candidate with positive base expectancy may continue to controls even when generality or concentration flags are active.

### C. Generality and concentration diagnostics: route, do not automatically kill

These include:

- positive-year contribution;
- per-year event counts;
- market-day concentration;
- symbol concentration;
- broad-versus-narrow universe;
- sensitivity to cluster estimand;
- regime or calendar dependence.

They decide the claim scope:

```text
broad_candidate
conditional_context_candidate
narrow_instrument_sleeve
sample_limited_candidate
prospective_only_candidate
```

A concentration failure remains a hard blocker for an **unconditional** claim. It does not by itself erase a potentially profitable **conditional** claim.

### D. Validation and deployment gates: remain strict

A research candidate cannot become validation-grade or live-ready without:

- mechanism-relevant controls;
- multiplicity and overlap review;
- independent or prospective evidence;
- calibrated execution and costs;
- drawdown, tail, capacity, and risk-of-ruin analysis;
- a frozen context/disable contract;
- no protected-data contamination.

## Payoff archetype must be frozen

Every economic contract declares one primary payoff archetype before outcomes:

```text
symmetric_directional
mean_reversion
convex_right_tail
carry_or_yield
event_driven_sparse
execution_or_microstructure
```

Gate interpretation follows the archetype.

### Symmetric or mean-reversion

Positive median remains a strong default requirement.

### Convex right-tail

A negative median is not automatically fatal. The contract must instead require:

- positive base mean;
- recurring winning tails across independent temporal blocks or contexts;
- bounded left-tail and drawdown;
- explicit top-event, top-day, top-symbol, and top-period contribution;
- realistic sizing that survives losing streaks;
- independent evidence before deployment.

The archetype cannot be changed after seeing the return distribution.

### Event-driven sparse

Per-year count gates may be inappropriate. The claim must instead be sample-limited, event-clustered, and prospective unless multiple independent event cycles exist.

## Prospective routing statuses

Future tasks use these bounded statuses in addition to the existing evidence dimensions:

```text
unconditional_control_candidate
conditional_context_candidate_unvalidated
convex_tail_candidate_unvalidated
execution_sensitive_candidate
narrow_sleeve_candidate
sample_limited_prospective_candidate
mechanically_unavailable
translation_rejected
blocked_by_data_or_authority
```

These are routing states, not evidence levels or deployment approvals.

## Revised control eligibility

Controls and ablations are research tools, not rewards granted only after universal stability.

A primary definition may proceed to mechanism controls when:

- integrity gates pass;
- the declared base-cost mean is positive;
- the payoff distribution satisfies its frozen archetype rule;
- independent event/cluster count is adequate for the intended claim;
- the cluster bootstrap is not clearly incompatible with the claimed effect;
- no single event or technical defect explains the result.

Temporal or regime concentration should normally **trigger** context controls rather than automatically prohibit all controls.

Controls still cannot promote a candidate by themselves. They test whether the claimed component adds information.

## Conditional-context route

A context-dependent candidate is acceptable only when the context is:

1. observable before the trade;
2. economically or mechanically related to the mechanism;
3. frozen without using protected outcomes;
4. not merely a calendar label such as `2024`;
5. tested against matched out-of-context observations;
6. carried forward to independent or prospective evidence.

When context dependence is first noticed from outcomes, register:

```text
post_hoc_context_hypothesis
```

The same sample may support mechanism explanation and controls, but not final validation. The historical run decision remains unchanged.

## Default interpretation of prior Level-3 gates

| Existing gate | Future default role |
|---|---|
| Trade and cluster counts | Hard for the exact intended claim; contract-specific for sparse sleeves. |
| Positive base mean | Core economic-relevance gate. |
| Positive median | Hard for symmetric/mean-reversion; diagnostic for frozen convex designs. |
| Bootstrap lower bound | Strength/uncertainty classification; may route to controls or prospective evidence rather than universal rejection. |
| Market-day contribution | Independence/sample-limit flag; hard for an unconditional broad claim. |
| Symbol contribution | Narrow-sleeve flag; hard only when the contract claims broad cross-sectional portability. |
| Positive-year contribution | Context/generalization flag, not a universal kill gate. |
| Stress mean | Execution sensitivity and risk flag; base implementation must still be realistic. |

Past contracts and decisions keep their original gates and outcomes. This policy is prospective and may create a new conditional translation without rewriting a failed unconditional one.

## KDA02 example

The Stage 9 negative completed-purge reversal at six hours had:

```text
base equal-market-day mean: +14.4404 bps
base equal-market-day median: +10.3686 bps
bootstrap lower bound: -2.6700 bps
stress mean: -3.5596 bps
market-day positive share: 2.08%
symbol positive share: 11.65%
positive-year share: 86.10%
```

Under the original frozen contract it correctly received:

```text
KDA02_level3_no_primary_pass_stop
```

because the unconditional 70% positive-year gate failed.

Under this new policy, that machine decision remains unchanged, while the mechanism-level route becomes:

```text
conditional_context_candidate_unvalidated
post_hoc_context_hypothesis_required
```

It is not a live candidate. It may receive a separately approved context/control study using pre-existing causal variables and must later obtain independent evidence.

## Portfolio and small-account objective

A small account can benefit from:

- low-capacity opportunities;
- sparse but high-payoff events;
- context-specific sleeves;
- strategies that would not scale institutionally;
- heterogeneous return streams rather than one universal rule.

Portfolio construction occurs only after sleeve-level evidence. Correlated weak strategies may not be combined to manufacture stability.

Required portfolio fields later include:

```text
opportunity frequency
capital occupancy
expected net return per unit of time and risk
worst losing streak
left-tail and drawdown
context overlap
cross-sleeve covariance
capacity at intended account size
execution mode
```

Leverage scales a demonstrated edge; it does not compensate for absent expectancy.

## Research posture

The programme should be decisive:

- close translations with no meaningful positive economic component;
- preserve and route conditional or convex evidence rather than forcing universal gates;
- run controls when they can resolve mechanism uncertainty;
- stop cosmetic threshold search;
- register post-hoc context hypotheses honestly;
- require independent evidence before live use.
