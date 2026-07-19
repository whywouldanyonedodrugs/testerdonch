---
status: active continuity after verified Stage 10 policy application
date: 2026-07-19
revision: rev11
scope: multi-platform programme state, Kraken analytics, KDA01/KDA02 decisions, and aggressive conditional-alpha routing
authority: machine contracts and finalized manifests first; explicit 2026-07-19 human gate-policy decision
supersedes: received `02_STATE_Master_Continuity_Brief_2026-07-19_rev11.md`; prior continuity remains provenance
provenance: rev10; Stage 7C; Stage 8A-8C1; Stage 9; gate-policy review
known limitations: Capital.com intake remains separate; no strategy is validation-grade or live-ready
---

# Master Continuity Brief, rev11

## Fixed programme scope

```text
research_platforms:
  - Kraken derivatives
  - Capital.com instruments present in a verified acquisition manifest
rankable_interval: [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
protected_period_start: 2026-01-01T00:00:00Z
Kraken_2026_purpose: execution_calibration_only
Capitalcom_2026_default: data_engineering_only
paid_historical_vendor_data: prohibited
orders_private_account_actions: not_authorized
```

## Current repository state

Verified Stage 10 starting authority after Stage 9:

```text
3ea0d320d71716a5c0890f4c924ed924224beda2
```

Repository state must be reverified at every task start.

## Kraken data state

Verified acquired rankable data now include:

- five-minute PF trade and mark bars;
- official same-asset Kraken spot history for the bounded C02 panel;
- exact late-period funding plus explicit mixed/imputed partitions;
- Stage 7C historical future basis, OI, and liquidation analytics;
- broad five-minute analytics and BTC/ETH one-minute analytics.

The analytics semantics are `inferred_authoritative_v1`. The universe remains current-roster/lifecycle capped and is not survivorship-free.

## Terminal Kraken decisions

```text
C01: level3_no_primary_pass_stop
C02: level3_no_primary_pass_stop
C03: C03_PIT_authority_unavailable
C16: C16_flow_authority_unavailable
KDA01: KDA01_level3_repaired_no_primary_pass_stop
KDA02A: KDA02_level3_no_primary_pass_stop
```

These exact run decisions remain immutable.

## Preserved conditional and unavailable routes

```text
KDA02 negative completed-purge reversal 6h:
    conditional_context_candidate_unvalidated
    historical unconditional run remains failed

KDA02 active purge continuation:
    mechanically unavailable under strict 5m contract

KDA02B OI vacuum:
    candidate_library_only; no outcomes opened

KDA02C isolated-versus-systemic purge breadth:
    post_hoc_context_hypothesis; no evidence yet

KDA03 basis state/dislocation:
    next untested Kraken mechanism
```

## Gate-policy update

Future research separates:

```text
integrity
economic relevance
generality/context
controls
independent validation
deployment
```

Year, symbol, or regime concentration no longer automatically erase a positive context-specific object. They limit the claim and route it to a conditional or sample-limited lane.

Past run decisions keep their original gates. No historical result is retroactively promoted.

## Main learned mechanics

- completed trade-and-mark failure remains a reusable confirmation component;
- OI, basis, and liquidation do not supply direction alone;
- downside and upside purge states are asymmetric;
- post-purge recovery, where present, is slower than one hour;
- market-wide breadth may distinguish isolated mean reversion from systemic continuation;
- five-minute entry must be at-or-after completed-bar availability;
- every interior timestamp of a feature window must exist;
- controls may be appropriate for a positive conditional candidate before universal generality is established.

## Next eligible phase

Before KDA03 economics:

1. prepare an outcome-free KDA03 adjudication under the new routing policy;
2. freeze its payoff archetype, claim scope, gates, diagnostics, controls eligibility, and independent-evidence requirement before any separately authorized outcomes;
3. retain KDA02 conditional-context work as a separate possible route, not an automatic next run.

No economic task is authorized by this brief.
