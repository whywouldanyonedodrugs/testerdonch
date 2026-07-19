# Stage 11 — KDA03 Basis-Shock Research

```text
task_id: donch_bt_stage_11_kda03_basis_shock_20260719_v2
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
outcome_free_adjudication_authorized: yes
conditional_economic_run_authorized: yes — frozen KDA03 definitions only after independent pre-outcome approval
controls_authorized: no
protected_outcome_access: no
new_data_acquisition: no
Capitalcom_payload_access: no
commit_authorized: yes
push_authorized: yes — non-force
```

## Objective

Research a fresh Kraken basis-change family:

```text
KDA03A reference-led basis catch-up proxy
KDA03B early leverage-backed basis impulse continuation
KDA03C completed basis-impulse rejection
```

Build the causal episode/event contract first. If at least one primary branch is mechanically feasible and independently approved before outcomes, execute one frozen Level-3 run and assign policy-v1.0 research routes.

## Verify before change

```text
expected main: e841469984478f7436db824587eac46dcd454c6d
policy JSON: docs/agent/RESEARCH_GATE_ROUTING_POLICY.json
policy SHA-256: c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa
analytics manifest: f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d
semantic contract: 289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60
Stage 8A feature contract: 4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4
rankable interval: [2023-01-01, 2026-01-01)
```

Verify root, `AGENTS.md`, Git/remotes/worktree, source/data hashes, Stage 8A cache, current registries, archive convention and Drive target.

Preserve exactly:

```text
KDA01_level3_repaired_no_primary_pass_stop
KDA02_level3_no_primary_pass_stop
KDA02 negative reversal 6h: conditional_context_candidate_unvalidated
```

No KDA01/KDA02 rescue, controls, protected data or Capital.com data.

## Stage 8A adjudication

Reconcile and preserve the 494,270 KDA03 feasibility rows:

```text
basis change + OI: 186,265
basis change without price confirmation: 247,169
extreme basis + liquidation/OI reset: 60,836
```

Document that these were broad row masks, not economic-ready events. The liquidation/OI-reset state remains KDA02 overlap evidence and receives no KDA03 economics.

## Semantic boundary

Basis is `inferred_authoritative_v1`: signed decimal state, positive when futures are above spot/reference. KDA03A is a **reference-led proxy** because the reference leg is not directly observed as a complete executable panel. This is a directional PF-futures study, not an arbitrage or spread backtest.

## Causal 15-minute features

Add if absent:

```text
basis change, level and prior level
trade and mark return/displacement
OI change
liquidation/OI intensity
signed basis-shock breadth and eligible denominator
BTC/ETH basis context
```

For basis change, price displacement and OI/liquidation states create prior-day 60-calendar-day robust z-scores and percentiles. Require 30 valid days, 70% expected coverage and every interior five-minute timestamp. Later same-day/future rows cannot change past values.

## KDA03A — reference-led catch-up proxy

Let `s = sign(basis_change_15m)`.

Primary onset:

```text
abs(basis_change_15m_z) >= 2
trade and mark absolute-displacement percentiles <= 0.25
abs(OI_change_15m_z) <= 1
liquidation_intensity_15m_z < 1
```

Robustness uses signed 95th/5th basis-change percentiles with the same remaining rules.

Direction:

```text
positive basis shock -> short
negative basis shock -> long
```

Archetype: `mean_reversion`. Emit immediately at completed-parent availability.

## KDA03B/C — early leverage-backed basis impulse

Primary onset:

```text
abs(basis_change_15m_z) >= 2
trade and mark 15m signs equal s
trade and mark absolute-displacement percentiles >= 0.75
OI_change_15m_z >= 2
abs(prior_basis_level_z) < 1.5
liquidation_intensity_15m_z < 2
```

Robustness uses signed 95th/5th basis-change percentiles and OI-change percentile >= 0.95.

### Immediate continuation

```text
positive shock -> long
negative shock -> short
archetype: symmetric_directional
```

### Completed rejection

Freeze the basis level immediately before the shock window and the onset trade/mark opens. Within six hours confirm the first completed bar where:

```text
positive shock:
    basis <= pre-shock basis
    trade close < onset trade open
    mark close < onset mark open

negative shock:
    basis >= pre-shock basis
    trade close > onset trade open
    mark close > onset mark open
```

Then trade opposite `s`. Archetype: `mean_reversion`. Touches, wicks or one-price-field confirmation do not qualify.

## Episodes, execution and clusters

For each symbol/attempt/direction:

- require 60 minutes without the same parent before a new episode;
- end after 30 minutes outside state or six hours;
- emit at most one immediate and one rejection candidate per episode;
- preserve zero-candidate episodes;
- keep primary and robustness separate.

Use deterministic parent-episode, UTC market-day and fixed six-hour cluster IDs.

Correct clock:

```text
stored candle time = interval start
decision = completed-bar start + 5m
entry = first authorized PF 5m open at or after decision
entry/exit maximum delay = 10m
exit = first authorized open at or after entry + timeout
```

Use definition-local, symbol-local actual-exit non-overlap.

Separate positive/negative branches. Freeze 1h and 6h timeouts. Maximum: 12 primary plus 12 robustness definitions.

## Mechanical eligibility

A branch enters economics only when it has:

```text
>=100 candidates
>=50 market-day clusters
>=10 symbols
0 duplicate event/economic addresses
0 protected rows
```

Per-year counts and concentration are routing diagnostics, not mechanical kill gates. Omit infeasible branches without replacement. If none are feasible, return `KDA03_mechanically_unavailable`.

## Conditional Level-3 run

After independent pre-outcome approval, execute once using:

```text
official PF 5m opens
fixed notional
14 bps base and 32 bps stress round-trip costs
1h/6h timeout only
funding diagnostic and excluded from gates
10,000 market-day bootstrap resamples; seed 20260719
```

Primary inference is equal-weight trades within each market day, then equal-weight market days. Also report trade, six-hour-cluster and parent-episode estimands.

## Policy-v1.0 routing

For these symmetric/mean-reversion definitions:

```text
base market-day mean > 0
base market-day median > 0
```

Otherwise route `translation_rejected`.

Control eligibility additionally requires bootstrap lower >= -5 bps, adequate clusters and no single-event/defect explanation. Controls remain unexecuted.

Report market-day, symbol and year contribution; per-year counts; estimand sensitivity; stress mean; basis breadth; BTC/ETH context.

Assign one exact route in this priority:

```text
translation_rejected
sample_limited_prospective_candidate
execution_sensitive_candidate
narrow_sleeve_candidate
conditional_context_candidate_unvalidated
unconditional_control_candidate
```

Interpretation:

- positive mean/median but bootstrap lower < -5 -> sample-limited;
- stress mean < -10 -> execution-sensitive;
- symbol contribution >25% -> narrow sleeve;
- year >70%, day >10% or material estimand/context dependence -> conditional-context;
- otherwise, positive and control-eligible -> unconditional-control candidate.

Do not use diagnostics to modify a definition.

## Freeze but do not execute controls

1. same trade/mark state without basis shock;
2. KDA03A without price non-confirmation;
3. KDA03A without stable OI;
4. price+OI impulse without basis shock;
5. basis+price impulse without OI expansion;
6. price-only structural rejection;
7. basis-level extreme as KDA01 overlap/non-rescue;
8. basis + liquidation/OI reset as KDA02 overlap/non-rescue;
9. matched BTC/ETH basis context;
10. timestamp null.

## Required evidence

Before outcomes: adjudication, feature contract, attempt registry, episode/event tapes, counts, eligibility, breadth/context summary, cluster summary, frozen contract/definitions/controls and independent pre-run review.

If economics execute: accepted schedule/trade tapes, definition metrics, route matrix, market-day returns, bootstrap, concentration with contributor identities, context diagnostics, funding partitions, decision and claim boundary.

Always: `VALIDATION.md`, `REVIEW.md`, `ARTIFACT_MANIFEST.json`, `COMPLETION.md`, `NEXT_ACTION.md`.

Large Parquet/bootstrap files remain local and hash-manifested.

## Tests and stops

Test causal windows and future invariance; signed shock direction; KDA03A non-confirmation/stable OI; KDA03B early basis/OI impulse; KDA03C basis+trade+mark rejection; corrected at-or-after entry; episodes/clusters; official-open fills; costs; archetype gates; exact policy vocabulary; robustness non-rescue; deterministic replay; no protected/Capital.com/control/KDA01/KDA02 outcomes.

Require independent pre-outcome and, when applicable, post-run review.

Stop on authority/hash failure, semantic or timestamp ambiguity, outcome-dependent definition change, KDA01/KDA02 rescue, protected/Capital.com/control access, unsafe Git state or replay failure.

## Overall status

Return exactly one:

```text
KDA03_level3_routes_assigned
KDA03_mechanically_unavailable
blocked_with_exact_mechanical_remedy
```

Report every primary definition and exact route; `routes_assigned` is not a pass.

After review, update KDA03 registries, create task-scoped commits, non-force push and complete verified Drive handoff.

## Final response

```text
status:
actual_starting_commit:
policy_hash_verified:
Stage8A_adjudication:
feature_and_generator_hashes:
episode_and_branch_counts:
mechanical_eligibility:
contract_hash_and_definition_count:
economics_executed:
primary_metrics_and_routes:
robustness_diagnostics:
control_eligible_definitions:
controls_executed: no
protected_rows_opened: no
Capitalcom_payload_opened: no
tests_and_reviews:
artifact_manifest_hash:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path_and_verification:
remaining_limits:
human_approval_required:
```
