# Stage 12 — KDA03 Closure and Hypothesis-Development Method Reset

```text
task_id: donch_bt_stage_12_kda03_closure_method_reset_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
economic_run_authorized: no
new_candidate_returns_authorized: no
existing_Stage11_outputs_read_authorized: yes — documentation and method diagnosis only
protected_outcome_access: no
new_data_acquisition: no
Capitalcom_payload_access: no
strategy_code_change_authorized: no
commit_authorized: yes — documentation, registries, policy-support objects, templates and non-economic tests only
push_authorized: yes — non-force under the standing workflow
```

## Objective

Close and document Stage 11 accurately, preserve its one prospective KDA03 object without promotion, and replace the programme’s implicit one-shot hypothesis-translation workflow with a formal staged development protocol.

This task must not compute new returns or open new market outcomes.

## Current state to verify

```text
expected starting commit:
    0fb08802a1eaa44d379618d10882198a3c9d0e9a

Stage 10 policy JSON:
    docs/agent/RESEARCH_GATE_ROUTING_POLICY.json

policy version / SHA-256:
    1.0
    c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa

Stage 11 contract hash:
    5f1abdc7e21ab9e3a6851b21930a7af745cbb865be334f8be45c9a561d2411e7

Stage 11 artifact-manifest file SHA-256:
    07a9a18f75320d44703b42b3ed0d0a03fe143bba89c24a875ec5e4ac6a9b2856

Stage 11 artifact-manifest content hash:
    0bc85e5056db8ddb38e7977761e2fe657647cf2cb632e0447153bf64e1cd3af7

Stage 11 terminal status:
    KDA03_level3_routes_assigned

Stage 11 published main:
    0fb08802a1eaa44d379618d10882198a3c9d0e9a
```

Verify repository root, `AGENTS.md`, branch/commit/remotes, working tree, current registries/manuals, Stage 11 task archive and manifests, archive convention and Drive target before change.

## Binding decisions to preserve

```text
KDA01:
    KDA01_level3_repaired_no_primary_pass_stop

KDA02A:
    KDA02_level3_no_primary_pass_stop

KDA02 negative completed-purge reversal 6h:
    conditional_context_candidate_unvalidated
    post_hoc_context_hypothesis

KDA03A reference-led basis catch-up:
    exact translation rejected

KDA03B immediate leverage-backed continuation:
    exact translation rejected

KDA03C positive completed rejection:
    exact translation rejected

KDA03C negative completed rejection 6h:
    sample_limited_prospective_candidate
    not control-eligible
    not validated
    not live-ready
```

Do not reinterpret any historical run as a pass.

## Required Stage 11 documentation

Record all 12 primary definitions and routes.

For the sole non-rejected primary object, preserve:

```text
definition:
    negative completed-basis rejection 6h

equal-market-day base mean:
    +9.1570 bps

equal-market-day base median:
    +2.9323 bps

bootstrap lower:
    -8.2953 bps

stress mean:
    -8.8430 bps

route:
    sample_limited_prospective_candidate

control eligibility:
    no
```

Also record that the limitation was not literally a tiny raw trade count. The object had substantial events/clusters but weak precision and threshold robustness. Add evidence-limitation tags without changing the active top-level route vocabulary:

```text
high_variance
wide_cluster_uncertainty
threshold_sensitive
not_control_eligible
```

These tags supplement routes. They do not promote evidence.

## Durable lessons to register

### Technical validity

- Stage 11 was independently valid and reproducible.
- No sign, timestamp, PF-open, cost, bootstrap, protected-data or policy-routing defect was found.
- The first pre-outcome freeze was correctly blocked and repaired before outcomes.

### Hypothesis-design lessons

1. Most KDA03 definitions were near-flat gross rather than strongly wrong after removing the frozen 14-bps cost.
2. Standardized extremes were not tied to minimum raw economic magnitudes.
3. Feature normalizations against prior daily summaries may make “extreme” intraday states broader than their labels imply.
4. A mechanism described as an extreme shock generated very frequent parent episodes.
5. KDA03A was underidentified because the reference leg was not directly observed as a complete executable panel.
6. OI expansion does not identify new-long, new-short, hedge or arbitrage actor direction.
7. Positive and negative derivatives shocks are not structurally symmetric.
8. Negative completed derivatives-state rejection over a slower horizon has now appeared in both KDA02 and KDA03, but neither object is validated.
9. A first plausible hand-written translation should not automatically be treated as the definitive confirmatory translation.
10. No same-sample threshold rescue is authorized by these lessons.

## New formal development protocol

Create a versioned machine-readable object, for example:

```text
docs/agent/HYPOTHESIS_DEVELOPMENT_PROTOCOL.json
```

and an explanatory manual.

The protocol must define these phases:

### Phase 0 — mechanism and actor specification

Require:

```text
compelled_actor_or_balance_sheet
why_pressure_should_persist_or_reverse
observable_Kraken_state
information_availability
main_null
falsification
data_semantic_limit
```

A generic multivariate state without a compelled actor receives `mechanism_underidentified`.

### Phase 1 — measurement validation, outcome-free

Before return analysis require:

```text
raw economic magnitude in native units and bps
standardized score
event frequency and duration
coverage by year/symbol/liquidity cohort
reference-leg authority where claimed
cost break-even magnitude
feature-window exactness
timestamp availability
```

A statistically unusual feature that is economically too small relative to plausible costs must be flagged before economics.

### Phase 2 — development-fold response characterization

This phase is outcome-bearing and always requires separate exact human approval.

When authorized, it must use predefined development folds only and register all explored cells.

Allowed purposes:

```text
continuous response surfaces
monotonicity
directional sign
horizon profile
component incrementality
context interaction
```

It is not confirmatory evidence.

### Phase 3 — translation freeze

Freeze from development evidence:

```text
direction
feature form
threshold or continuous rank
horizon
payoff archetype
context
controls
costs
multiplicity family
```

Do not transfer information backward from a later fold.

### Phase 4 — untouched fold evaluation

Evaluate the frozen translation on the next untouched rankable block using purging/embargo where needed.

### Phase 5 — rolling or nested replication

Repeat development/freeze/evaluation forward in time without exposing future blocks to earlier design.

### Phase 6 — controls, independent evidence and deployment review

Maintain existing control, validation, protected and deployment requirements.

## Economic-run admission standard

Update manuals/templates so a future full Level-3 run is not admitted directly from a broad qualitative idea unless all are documented:

```text
measurement_semantics_valid
event_frequency_consistent_with_mechanism
raw_magnitude_economically_relevant
actor_or_structural_mechanism_identified
development_route_or_explicit_one_shot_reason
horizon_justified
component_controls_defined
payoff_archetype_frozen
```

A genuinely simple one-shot hypothesis remains allowed when its rule follows directly from a strong external/mechanical theory. The reason must be explicit.

## Evidence-limitation tags

Create a machine-readable tag registry separate from route status. Include at minimum:

```text
small_sample
few_independent_clusters
high_variance
wide_cluster_uncertainty
threshold_sensitive
cost_sensitive
symbol_concentrated
time_concentrated
context_dependent
measurement_underidentified
proxy_side_or_reference
mechanically_sparse
lifecycle_capped
funding_incomplete
```

Tags describe why evidence is limited. They do not change a route or evidence level.

## Required repository updates

Update surgically:

```text
docs/QLMG_PERP_PROJECT_STATE.md
current continuity brief
current research decisions
Kraken derivatives learning state
family/hypothesis registry
attempt/multiplicity registry
run/evidence registry
gate-routing/route-support registry where appropriate
agent operating instructions
test/evidence manual
claims/review guide
research-efficiency method
task template
```

Add:

```text
KDA03_STAGE11_LESSONS.md
HYPOTHESIS_DEVELOPMENT_PROTOCOL.md
docs/agent/HYPOTHESIS_DEVELOPMENT_PROTOCOL.json
docs/agent/EVIDENCE_LIMITATION_TAGS.json
```

Use current repository paths and schemas where they already exist.

Do not change the nine active route statuses in policy v1.0 during this task. Add limitation tags and the staged protocol around them.

## Next-research planning output

Create one non-authorizing planning document:

```text
NEXT_HYPOTHESIS_DEVELOPMENT_CANDIDATES.md
```

It should compare, without new outcomes:

- KDA02C isolated-versus-systemic purge breadth;
- KDA02B OI vacuum;
- a distinct downside completed derivatives-state rejection mechanism combining only pre-existing causal components;
- a fresh non-derivatives family from the current candidate library.

For each state:

```text
mechanism
compelled actor
available data
semantic weaknesses
measurement-validation needs
development-fold design
main controls
same-sample restrictions
priority rationale
```

Do not rank using protected data or new return analysis.

## Prohibited actions

- no new return, PnL, bootstrap or price-path computation;
- no reading new market outcome files;
- no KDA03 threshold, direction, symbol, year, cost or horizon variants;
- no KDA02 context/control outcomes;
- no route promotion;
- no strategy, loader or simulator change;
- no protected or Capital.com payload;
- no acquisition;
- no deployment or live action.

## Validation and review

Verify:

- Stage 11 roots, hashes, metrics and routes;
- all prior terminal decisions remain unchanged;
- route vocabulary remains policy v1.0;
- limitation tags are separate from routes;
- development-fold analysis is explicitly approval-gated;
- no market reader or economic module is invoked;
- JSON/CSV schemas and links pass;
- secret scan and `git diff --check` pass;
- independent review confirms the protocol permits disciplined exploration without same-sample rescue.

The absence of optional test dependencies may be recorded only if repository-supported dependency-free validation covers the documentation/policy objects. Do not install dependencies unless authorized.

## Archive and handoff

Archive:

```text
TASK_SPEC.md
DONCH_CONTEXT.md
PLAN.md
DECISIONS_AND_PROGRESS.md
CHANGED_FILES.md
COMMANDS_AND_RESULTS.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

After approval:

- create task-scoped documentation/registry commits;
- non-force push;
- complete `drive_handoff: approved_default`;
- round-trip verify size and SHA-256;
- retain the local archive.

## Stop conditions

Stop if:

- repository or Stage 11 authority diverges;
- a terminal decision would be overwritten;
- market outcomes would be reopened;
- a route is retroactively promoted;
- strategy code or data would change;
- protected/Capital.com data would be opened;
- the method would permit unregistered same-sample exploration;
- verification fails.

## Final response

```text
status:
actual_starting_commit:
Stage11_authority_verified:
terminal_decisions_preserved:
KDA03_results_registered:
development_protocol_version_and_hash:
evidence_limitation_tags_registered:
route_vocabulary_changed: no
manuals_and_registries_updated:
strategy_code_changed: no
economic_outputs_computed: no
protected_rows_opened: no
tests_and_review:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
next_candidate_comparison:
human_approval_required:
```
