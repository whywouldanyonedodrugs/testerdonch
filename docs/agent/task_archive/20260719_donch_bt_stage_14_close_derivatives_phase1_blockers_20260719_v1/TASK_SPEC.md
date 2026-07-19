# Stage 14 — Close Derivatives Phase-1 Blockers and Regenerate the Campaign Packet

```text
task_id: donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
outcome_free_measurement_authorized: yes
forward_return_or_PnL_access: no
phase_2_or_later_economics_authorized: no
protected_outcome_access: no
new_data_acquisition: no
Capitalcom_payload_access: no
C17_work_authorized: no
commit_authorized: yes
push_authorized: yes — non-force
```

## Objective

Close the three outcome-free Phase-1 blockers for the first derivatives-state campaign:

```text
KDA02B_v2_oi_vacuum_redevelopment
KDA02C_v1_purge_breadth_context
KDX01_v1_downside_completed_derivatives_state_rejection
```

Materialize and validate the contemporaneous measurements, revise the placeholder search spaces where needed, benchmark the campaign, and regenerate one independently reviewed, hash-bound, non-authorizing approval packet for a later unattended Phase 2–5 campaign.

Do not calculate any forward return, trade return after decision, PnL, bootstrap economic result, candidate ranking, or protected outcome.

## Current authority to verify

```text
expected starting commit:
    e14bbd0d26c14e48a347481f170fcfe8851df625

Stage 13 campaign manifest:
    09f4a534dcd799bf252e6e47037af58ad2177256d976b37defd8c257d2949e27

Stage 13 non-authorizing approval request:
    ed27fc30d39155d6b08d63539a22d34b50d19ab3c909396a8719c443057fcd5c

hypothesis protocol v2:
    4effe3e9608377876486b1583854a7e1479c8e93d6de1661f13d35842bbc6a73

family-redefinition policy:
    d07b0d290e59d17f7d6a587e2a31e6e550468f9ad403250cd6183c549e685335

campaign protocol:
    7091156df9bf815a001423b63d673d2d4217f2616fd40e76a261f218a2d614df

route policy v1:
    c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa

analytics manifest:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

authorized cohort:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636
```

Verify repository root, `AGENTS.md`, Git/remotes/worktree, Stage 13 campaign engine and archive, source/data hashes, supported commands, data roots and Drive target.

Preserve all historical run decisions and registry fields.

## Outcome firewall

Allowed reads:

- source manifests and schemas;
- contemporaneous trade, mark, OI, basis and liquidation features through `decision_ts`;
- frozen Stage 8–11 parent/event tapes when their fields are available by `decision_ts`;
- cohort membership, coverage, timestamps and lifecycle masks;
- state incidence, duration, overlap, breadth and raw-magnitude distributions;
- resource benchmarks using no forward labels.

Forbidden reads or computations:

- any price, return or cashflow after `decision_ts`;
- entry/exit prices;
- forward horizon labels;
- PnL or strategy statistics;
- bootstrap economic results;
- candidate ranking by outcome;
- protected rows;
- KDA02/KDA03 result tables as feature-selection inputs beyond recorded lineage/contamination labels.

Add a read-spy or equivalent test proving that no forward-outcome reader is invoked.

## 1. KDA02B — OI-vacuum Phase-1 closure

Do not rely on the Stage 8A count matrix as if 1,176,354 rows were independent onsets.

Reconstruct the source state causally from the frozen feature authority and produce:

```text
false-to-true onset tape
state episode tape
episode duration
time between episodes
directional price state at decision
native OI log-change magnitude
OI-change percentile and robust z
contemporaneous trade/mark displacement in bps
contemporaneous realized-vol ratio
liquidation present/absent state
coverage by symbol/year/liquidity cohort
```

### Retention boundary

For every symbol record:

```text
first raw OI timestamp
first timestamp with complete causal lookback
first admissible onset timestamp
invalid boundary interval
gaps and truncation status
```

No onset may be created by the start of OI retention or an incomplete lookback.

### Semantic decision

OI is unsigned and cannot identify new longs, new shorts or liquidation direction. Decide outcome-free whether the family remains:

```text
mechanism_development_ready
```

or:

```text
mechanism_underidentified
```

Continuation and reversal may both remain development branches when the actor direction is not identified. Do not choose between them from prior returns.

## 2. KDA02C — purge-breadth materialization

Using the frozen KDA02 completed-purge parent/event identity and the authorized cohort, materialize a point-in-time breadth panel.

For each five-minute timestamp store:

```text
eligible denominator
known and excluded members
completed-purge onset count
active completed-purge episode count
negative and positive counts separately
cohort share
membership-change count
BTC/ETH state identifiers
source hashes
```

Measure outcome-free distributions and episode persistence for fixed diagnostic windows:

```text
5m
15m
30m
60m
```

Report denominator stability, isolated-event frequency, broad-event frequency, clustering, year/symbol coverage and sensitivity to primary versus robustness purge identity.

The breadth panel must use only members eligible at that timestamp. The current-roster/lifecycle cap remains explicit.

Retain:

```text
post_hoc_context_hypothesis
program_exposed_historical
```

No breadth threshold may be chosen from future outcomes.

## 3. KDX01 — joint-state materialization

Do not define KDX01 merely as a union of previously favourable branches.

Build an outcome-free primitive state table using causal contemporaneous components:

```text
downside trade displacement
downside mark displacement
OI contraction
liquidation intensity
basis level and basis change
structural reclaim/rejection completion
market-wide breadth
```

Produce:

- raw-unit distributions;
- component availability and missingness;
- all registered component-overlap counts;
- event and episode durations;
- joint-state incidence by year/symbol;
- overlap with KDA01, KDA02 and KDA03 source identities;
- duplicate/economic-address analysis;
- actor-identification and proxy limitations.

Create a constrained component grammar rather than selecting one prior winner. Limit interaction depth and preserve:

```text
cross_family_program_exposed_redevelopment
```

If the joint state cannot be distinguished from contemporaneous price-only movement without outcomes, stop only KDX01 as `mechanism_underidentified`.

## 4. Universe reconciliation

Using existing manifests only, produce the exact ledger:

```text
K0 PF inventory
crypto PF identities
rankable trade coverage
rankable mark coverage
OI coverage
basis coverage
liquidation coverage
causal-lookback eligibility
ever in authorized cohort
final campaign-eligible symbols
```

Give one exclusion reason per omitted identity.

Do not add symbols or acquire data. State whether any existing fully covered symbol can be admitted without changing the cohort contract. The future campaign remains on the current authorized cohort unless a separate cohort rebuild is approved.

## 5. Raw economic relevance without outcomes

For each family, report contemporaneous state magnitudes against the fixed cost contract:

```text
14 bps base round trip
32 bps stress round trip
```

This is not a forward-return test. It asks whether the state’s observed raw price displacement and plausible remaining movement scale are economically meaningful enough to justify Phase 2.

Do not infer expected profit from contemporaneous magnitude.

## 6. Funding/cashflow contract audit

The Stage 13 placeholder packet requires exact funding and fails closed when exact funding is missing. Audit this against actual historical funding coverage and prior authoritative contracts.

Do not silently discard most 2023–2025 folds.

Produce a final proposed campaign cashflow contract that:

- preserves 14/32 bps pre-funding costs;
- records exact, mixed, imputed and zero-boundary partitions;
- prevents favourable imputed funding from selecting candidates;
- states the primary development/evaluation metric;
- states exact-funding and conservative missing-funding sensitivities;
- fails closed only where the approved contract actually requires exact funding.

This proposal remains non-authorizing and will be approved or rejected with the final packet.

## 7. Search-space revision

The Stage 13 search spaces are placeholders. Revise them only from Phase-1 measurement and mechanism semantics, never from forward outcomes.

The final spaces should use:

- raw-unit and fold-local percentile/rank axes;
- continuous response surfaces before hard thresholds;
- fixed component-incrementality ladders;
- separate positive/negative states where applicable;
- a registered horizon profile;
- limited interaction depth;
- sparse model/rule discovery only where protocol v2 permits;
- complete cell registration and inherited multiplicity.

Do not preserve an arbitrary eight- or twelve-cell space merely because it already exists.

Also do not create an unbounded grid.

Determine exact cell budgets from a no-outcome runtime/memory benchmark. Bind:

```text
maximum cells per family
maximum total cells
candidate beam
complexity budget
deterministic tie-break
```

## 8. Selection and route contract

Regenerate the future packet with a fuller predeclared selection objective:

```text
base net expectancy after authorized costs
median and worst inner-fold utility
cross-fold sign/stability
effective independent cluster count
left-tail/drawdown proxy appropriate to archetype
opportunity frequency
capital occupancy
execution margin
complexity
candidate correlation
```

These fields define later selection; do not compute them now.

Multiplicity must include every response bin, feature subset, model, rule, direction, horizon and context cell.

The future packet must retain:

```text
programme_exposed_historical
not independent validation
```

## Phase-1 admission

Each derivatives lane receives one outcome-free status:

```text
phase_2_ready
mechanically_unavailable
mechanism_underidentified
blocked_with_exact_mechanical_remedy
```

`phase_2_ready` requires:

- complete causal measurement;
- retention and missingness controls;
- economically relevant raw scale;
- sufficient incidence and independent clusters for development;
- exact search-space manifest;
- fold schedule;
- cost/cashflow contract;
- resource benchmark;
- no unresolved authority defect.

A family-specific failure does not block unrelated ready lanes.

## Final campaign packet

If at least one lane is `phase_2_ready`, regenerate:

```text
CAMPAIGN_MANIFEST.json
SEARCH_SPACE_REGISTRY.json
FOLD_AND_EXPOSURE_MAP.json
RESOURCE_PROJECTION.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md
```

The packet must:

- name only ready lanes;
- request exact Phases 2–5;
- exclude C17;
- bind repository/data/feature/code/cost/search/fold hashes;
- require an external human-approval artifact naming the final packet hash;
- be non-authorizing by itself;
- support one later unattended run without routine approval between folds or hypotheses.

## Required outputs

```text
KDA02B_PHASE1_MEASUREMENT.md
KDA02B_ONSET_AND_EPISODE_SUMMARY.csv
KDA02B_RETENTION_BOUNDARY.csv
KDA02C_PIT_BREADTH_CONTRACT.md
KDA02C_BREADTH_SUMMARY.csv
KDX01_PRIMITIVE_STATE_CONTRACT.md
KDX01_COMPONENT_OVERLAP_MATRIX.csv
KDX01_INCIDENCE_SUMMARY.csv
KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.csv
CAMPAIGN_FUNDING_AND_COST_CONTRACT.md
PHASE1_ADMISSION_DECISIONS.json
SEARCH_SPACE_REGISTRY.json
RESOURCE_PROJECTION.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large outcome-free state tapes may remain local and hash-manifested.

## Tests and independent review

Test at minimum:

1. no forward-return or execution-price read;
2. zero protected rows;
3. KDA02B false-to-true onset and duration logic;
4. OI retention-boundary exclusion;
5. KDA02C PIT denominator and breadth invariance;
6. KDX01 component grammar and contamination labels;
7. exact timestamp availability;
8. universe exclusion ledger reconciliation;
9. funding-partition authority;
10. search spaces derived only from Phase-1 measurement;
11. all cells registered and bounded;
12. deterministic benchmark and packet replay;
13. family/global stop isolation;
14. packet cannot self-authorize;
15. historical terminal decisions unchanged.

Require independent adversarial review of measurements, semantics, universe, search spaces, funding contract, campaign packet and outcome firewall.

## Integration and handoff

After review:

- update Phase-1 and campaign-readiness registries only;
- preserve all historical run decisions;
- create task-scoped commits;
- non-force push;
- complete `drive_handoff: approved_default`;
- round-trip verify uploaded size and SHA-256;
- retain the local archive.

## Stop conditions

Stop globally for authority mismatch, protected exposure, shared timestamp defect, unsafe Git/storage or deterministic replay failure.

Stop only the affected family for missing measurement, underidentified mechanism, insufficient incidence or family-specific defect.

Do not proceed to Phase 2.

## Final response

```text
status:
actual_starting_commit:
authority_hashes_verified:
outcome_firewall_verified:
KDA02B_phase1_result:
KDA02C_phase1_result:
KDX01_phase1_result:
universe_reconciliation:
funding_cost_contract:
revised_search_space_hashes:
resource_benchmark:
ready_campaign_lanes:
final_campaign_manifest_hash:
final_approval_packet_hash:
economic_outputs_computed: no
protected_rows_opened: no
Capitalcom_payload_opened: no
tests_and_review:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
human_approval_required:
```
