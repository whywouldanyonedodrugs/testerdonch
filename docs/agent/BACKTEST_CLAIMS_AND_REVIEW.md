---
status: active claims and review guide after verified Stage 10 policy application
date: 2026-07-19
revision: 3.0
scope: bounded claims, conditional sleeves, gate routing, Kraken/Capital.com separation, and review outcomes
authority: operating contract and Test and Evidence Standards rev3
supersedes: received `13_GUIDE_Backtest_Claims_and_Review_2026-07-19_rev3.md`; prior guides remain provenance
provenance: rev2; Stage 7C-9; aggressive conditional-alpha policy
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
- prohibited next actions.

## Allowed bounded wording

```text
unconditional_control_candidate
conditional_context_candidate_unvalidated
convex_tail_candidate_unvalidated
execution_sensitive_candidate
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

1. authority and current machine root;
2. platform and protected boundary;
3. timestamp and execution semantics;
4. event/control/episode/cluster identity;
5. payoff archetype and intended claim scope;
6. base economic relevance;
7. uncertainty and concentration;
8. mechanism controls;
9. multiplicity and overlap;
10. independent evidence;
11. risk, capacity and deployment limits.

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
