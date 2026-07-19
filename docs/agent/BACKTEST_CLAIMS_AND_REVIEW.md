---
status: active claims and review guide after verified Stage 12 method reset
date: 2026-07-19
revision: 4.0
scope: bounded claims, conditional sleeves, gate routing, Kraken/Capital.com separation, and review outcomes
authority: operating contract and Test and Evidence Standards rev4
supersedes: received `13_GUIDE_Backtest_Claims_and_Review_2026-07-19_rev3.md`; prior guides remain provenance
provenance: rev3; Stage 7C-11; aggressive conditional-alpha policy; Stage 12 method reset
known limitations: exact machine decisions and manifests remain higher authority
---

# Backtest Claims and Review Guide

## Required claim fields

Every claim names:

- platform and instrument contract;
- exact rankable interval and protected-row count;
- hypothesis, translation and payoff archetype;
- data and universe authority;
- decision and executable price clocks;
- costs and cashflows;
- primary inference estimand;
- controls and multiplicity;
- concentration identities;
- evidence, reproducibility, validation and deployment status;
- research route;
- prohibited next actions;
- development phase, fold role, explored-cell register, and whether the evaluated block was untouched;
- evidence-limitation tags, reported separately from route and evidence level.

## Allowed bounded wording

```text
unconditional_control_candidate
conditional_context_candidate_unvalidated
convex_tail_candidate_unvalidated
execution_sensitive_candidate
narrow_sleeve_candidate
sample_limited_prospective_candidate
translation_rejected
mechanically_unavailable
blocked_by_data_or_authority
```

These terms do not mean validated or live-ready.

## Conditional claims

Allowed:

> The unconditional translation failed its frozen year-concentration gate, while one branch retained positive base economics and is preserved as a post-hoc conditional-context hypothesis requiring independent evidence.

Not allowed:

> The strategy works in the good year, so trade it when that year returns.

A calendar period is evidence of concentration, not a causal context.

## Convex claims

Allowed only when the payoff archetype was frozen before outcomes. Report mean, median, win rate, payoff ratio, left tail, top-event contributions, losing streaks and temporal recurrence.

## Review order

1. protocol phase, actor/mechanism identification, and raw measurement validity;
2. authority and current machine root;
3. platform and protected boundary;
4. timestamp and execution semantics;
5. development-fold registration, translation freeze, and untouched-block status;
6. event/control/episode/cluster identity;
7. payoff archetype and intended claim scope;
8. base economic relevance;
9. uncertainty, limitation tags, and concentration;
10. mechanism controls;
11. multiplicity and overlap;
12. independent evidence;
13. risk, capacity and deployment limits.

## Gate-failure interpretation

Reviewers must state whether a failed gate means:

```text
invalid evidence
no economic relevance
uncertain evidence
context concentration
narrow sleeve
execution sensitivity
sample limitation
```

Do not collapse these into one generic `failed` label.

## Historical decisions

Past terminal decisions remain unchanged. New gate policy may change the permitted next route, but cannot retroactively convert a failed run into a pass.

## Review template

```text
artifact_or_run:
authoritative_root:
platform_and_instrument:
rankable_and_protected_boundary:
mechanism_and_payoff_archetype:
source_availability_and_execution_clock:
economic_relevance:
uncertainty:
market_day_symbol_year_concentration:
claim_scope:
controls_and_ablations:
overlap_and_multiplicity:
reproducibility:
evidence_level:
research_route:
claim_supported:
claim_not_supported:
binding_decision:
permitted_next_action:
prohibited_next_actions:
human_approval_required:
```

Reject confirmatory wording when the evidence came from Phase 2 development folds, when explored cells are missing from the attempt registry, or when a later fold changed an earlier translation. A limitation tag never changes the binding research route.
