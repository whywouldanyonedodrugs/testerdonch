# Stage 17 — Launch and Complete the Approved Unattended Derivatives Campaign

```text
task_id: donch_bt_stage_17_run_unattended_derivatives_campaign_20260720_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default

economic_campaign_authorized: yes — exact Stage-16 packet and attached human approval only
approved_phases: 2, 3, 4, 5
approved_lanes:
  - KDA02B_v2_oi_vacuum_redevelopment
  - KDA02C_v1_purge_breadth_context
  - KDX01_v1_downside_completed_derivatives_state_rejection
approved_executable_cells: 186

phase_6_controls_authorized: no
phase_7_validation_or_deployment_authorized: no
protected_outcome_access: no
Capitalcom_payload_access: no
new_data_acquisition: no
cohort_rebuild_or_symbol_expansion: no
orders_or_live_trading: no
commit_authorized: yes
push_authorized: yes — non-force
```

## Exact external human approval

The attached file is the sole external approval artifact:

```text
HUMAN_APPROVAL_Kraken_Derivatives_Campaign_001_Stage16_Phases_2_5_2026-07-20_v1.json
SHA-256: fe57d5c1efca3af3cb83c3e07b399e03c51f5dbe635b03bd48201944506c6853
```

It approves only:

```text
campaign manifest file SHA-256:
    cc07499c671cf39b8ceaee91156f141dcc2c5532142af29a38a4f6830b73f23d

approval packet file SHA-256:
    c01281e50f40f95b922a04ed01c5b3d28ed325577891eed8e2ca5d32286965ca

phases:
    2, 3, 4, 5

lanes:
    KDA02B_v2_oi_vacuum_redevelopment
    KDA02C_v1_purge_breadth_context
    KDX01_v1_downside_completed_derivatives_state_rejection
```

Do not reuse the Stage-15 approval. Do not broaden this approval.

## Current authority to verify

```text
expected starting main/origin/main:
    a3981b505e908b5fb617a0921f45869535e2b542

Stage-16 semantic implementation:
    bd2eee2b7c8c90b5c392609a9f6fc70294326ec6

Stage-16 deterministic replay cleanup:
    5853679f9cec19937fdc6818b8f946f67c1c430a

Stage-16 packet binding:
    0addc85647cf6b77e28555d9320c5d157583991b

Stage-16 final publication:
    a3981b505e908b5fb617a0921f45869535e2b542

replacement campaign manifest:
    cc07499c671cf39b8ceaee91156f141dcc2c5532142af29a38a4f6830b73f23d

replacement approval packet:
    c01281e50f40f95b922a04ed01c5b3d28ed325577891eed8e2ca5d32286965ca

economic translation registry:
    9c07f8695b117afe61f03354e0b6ab39a4c11bd4f0335f9f066cca33560ff1f8

search/bin specification:
    1cc994543371a8db86f428957a27d3a62d93f390e30f99460aa3e1d973b8c67b

estimator/rule inventory:
    b30b7c115d6d1ed765542c44d791c44117a0387f71ae4b33ec1219d4243a3b

inner-fold map:
    6ee8cbc53eac9326f904eec760ae82861a533b8165be6e1ccfc440dd0fa32ba0

utility/Pareto contract:
    8b9d6afa6c1ea6676cdac9b1703223ef1d47d929988c8e419fc1c112e4a3f093

boundary contract:
    460eb30f45c232c79729765b8724ddd206ed09671bb98f485b2175bf83cc5e75

funding contract:
    d95c3b21c495a712efb6f9834200c6d3aa95f4bc7a9fda5492e2230a739fa04a

Telegram/supervision contract:
    f23736649f4523add8618720744673a5554d51246b9bcb01bcccf0a64591e70e

resource projection:
    343735cba47cb39d15d56ded7649338a95980b09c7b236894439026ea7a2664c

analytics manifest:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

authorized cohort:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636
```

Verify repository root, `AGENTS.md`, branch/remotes/worktrees, clean status, task archives, packet bytes, dependency bytes, data roots, available RAM/disk/CPU, campaign state and Drive target.

## Non-negotiable semantic freeze

Stage 16 concluded that no economic or selection semantics remain to be invented. Therefore:

- use the exact 186 executable translations in the Stage-16 registry;
- retain the 42 inherited KDX continuation-null attempts as multiplicity only;
- do not add, remove, merge or reinterpret cells;
- do not change bins, rules, sides, instruments, horizons, entries, exits, costs, funding, folds, metrics, Pareto rules or beam rules;
- do not manually select candidates;
- do not inspect partial winners or intervene based on outcomes.

Mechanical runtime wiring is allowed only when it directly implements the frozen packet without changing its meaning. Any required semantic decision causes a pre-outcome stop and requires a new packet.

## Pre-outcome sequence

Complete in this order:

1. verify exact approval-artifact bytes and all raw/canonical packet/dependency hashes;
2. validate the 186-cell translation registry and 27-fold inner/outer map;
3. compile and run the focused Stage-16/campaign suites;
4. run the launch-readiness validator and synthetic end-to-end canary;
5. materialize the frozen 2023 funding extension using only PIT inputs;
6. verify funding coverage gates;
7. verify resource limits and persistent-supervisor availability;
8. perform independent read-only pre-outcome review;
9. validate Telegram notifier and alerts;
10. launch.

No outcome reader may open before all ten gates pass.

## Funding gate

Use exact Stage-16 funding contract:

```text
base pre-funding round trip: 14 bps
stress pre-funding round trip: 32 bps
minimum coverage per hypothesis-fold: 90%
minimum campaign-weighted coverage: 95%
missing required boundary: reject before outcome; never call zero-boundary
favourable imputation: cannot activate, rescue, rank or select
```

Recompute the frozen 2023 model point-in-time with unchanged parameters. Record exact, mixed, imputed and zero-boundary partitions. Independent review must verify the boundary manifest before outcomes.

## Telegram launch gate

Use only the existing secure notifier configuration. Never print, archive, commit or return bot tokens, chat IDs or secret-file contents.

Before outcomes:

- send a dry-run notification with campaign ID, starting commit, approval packet hash and `preflight`;
- verify delivery from the notifier response/log;
- send one synthetic heartbeat;
- send one synthetic stop alert;
- verify both without exposing credentials.

Stop before outcomes as:

```text
blocked_telegram_notifier_unavailable
```

when secure configuration or delivery fails.

During the campaign, send:

- campaign start;
- every phase transition;
- every completed outer fold;
- family stop;
- global warning or stop;
- terminal completion;
- heartbeat every 30 minutes.

Heartbeat fields:

```text
campaign ID
phase / fold / family
completed cells / total cells
elapsed / ETA
workers
aggregate RSS
free disk
errors / retries
heartbeat age
```

Do not include partial economic rankings or candidate results in routine messages.

## Persistent unattended execution

Run under a persistent supervisor, preferably repository-supported `systemd --user`; otherwise use a reviewed `tmux`-based supervisor.

Persist:

```text
service/session identity
PID tree
run root
campaign state generation
heartbeat
stdout/stderr logs
resource telemetry
exact stop reason
```

Limits:

```text
workers: maximum 4
wall time after economic launch: 14,400 seconds
aggregate process RSS: maximum 5 GiB
campaign output: maximum 5 GiB
candidate beam: maximum 5 per family/fold
```

Use idempotent restart and atomic state/artifacts. A resumed process must match the approval, packet, manifest, translation registry and last committed state generation exactly.

After launch, observe only until:

- the supervisor remains healthy;
- at least one real registered cell completes and reconciles;
- the first scheduled 30-minute Telegram heartbeat is delivered;
- campaign state and artifact hashes verify.

Then leave the campaign unattended. Continue automated monitoring, but do not inspect or tune partial economic results. Return only after terminal completion or a bound stop condition.

## Phase 2 — registered development

Execute every registered cell, or retain an exact mechanical skip/stop reason.

Requirements:

- all explored cells registered;
- development folds only;
- exact Stage-16 response bins and constrained rule inventory;
- no hidden variants;
- exact development metrics and equal-market-day weighting;
- preserve complete response surfaces, including negative cells;
- family-specific failure does not stop unrelated lanes.

## Phase 3 — deterministic freeze

For each family/fold:

- determine eligibility under the frozen contract;
- apply exact finite-value, Pareto-dominance and lexicographic rules;
- retain at most five candidates;
- enforce the correlation preference and high-correlation tag;
- freeze full economic addresses before the relevant outer block opens;
- obtain deterministic independent freeze verification.

No manual candidate selection is allowed.

## Phase 4 — outer evaluation

Execute the frozen translations using:

```text
native PF symbol and side from the registry
first authorized PF 5m open at or after decision
maximum entry delay 10 minutes
fixed/structural exit exactly as registered
maximum exit delay 10 minutes
definition-local symbol-local actual-exit non-overlap
same-fold decision, entry and exit
14/32-bps costs plus frozen funding contract
```

Evaluation information cannot change the candidate, rule, horizon or later interpretation of the same fold.

## Phase 5 — forward rolling replication

Continue through every authorized outer fold in order. Preserve each frozen candidate identity and fold result. Do not pool away negative or unavailable folds.

All outputs remain:

```text
program_exposed_historical
not independent validation
not live-ready
```

## Multiplicity and routes

Reconcile:

```text
186 executable cells
42 inherited non-executable KDX attempts
228 total programme attempts in this campaign lineage
```

Apply the frozen family-level dependence-aware comparison, routes and limitation tags. Do not treat concentration, convexity, uncertainty or one weak year as automatic family death when the route policy specifies a bounded candidate state.

Controls remain unexecuted.

## Stop rules

Global stops:

- approval, packet, dependency or data hash drift;
- protected or Capital.com access;
- common timestamp/execution/funding defect;
- deterministic replay failure;
- unsafe Git, storage or resource state.

Family stops:

- mechanically invalid;
- insufficient integrity or cluster count;
- no positive development candidate;
- family-specific defect.

A family stop must not terminate independent lanes.

At wall/resource stop, shut down gracefully, persist resumable state and report incomplete scope. Do not alter the packet to finish.

## Required artifacts

At minimum:

```text
HUMAN_APPROVAL.json
PREOUTCOME_AUTHORITY_AND_HASH_AUDIT.json
FUNDING_BOUNDARY_EXTENSION_MANIFEST.json
TELEGRAM_NOTIFICATION_VALIDATION.md
CAMPAIGN_LAUNCH_MANIFEST.json
CAMPAIGN_STATE.json
CAMPAIGN_HEARTBEAT.json
EXPLORED_CELL_REGISTRY
RESPONSE_SURFACE_RESULTS
CANDIDATE_BEAM_REGISTRY
FROZEN_TRANSLATION_REGISTRY
OUTER_FOLD_RESULTS
ROLLING_REPLICATION_RESULTS
MULTIPLICITY_RESULTS
ROUTE_AND_LIMITATION_MATRIX
FAMILY_DECISIONS
CAMPAIGN_DECISION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large economic tapes remain local and hash-manifested. Compact evidence goes to Drive.

## Tests and reviews

Require:

- approval and raw/canonical hash checks;
- complete 186-cell registry validation;
- no protected/Capital.com rows;
- funding extension and coverage verification;
- Telegram preflight without secret leakage;
- deterministic development, beam and freeze replay;
- exact execution, costs and funding arithmetic;
- inner/outer fold embargo and no backward leakage;
- complete cell/attempt reconciliation;
- post-run independent recomputation of selected candidates, outer-fold results, multiplicity and routes;
- secret scan and `git diff --check`.

## Git, publication and Drive

The repository starts clean. Keep unrelated changes out of scope.

Create separate reviewed commits for:

1. exact mechanical runtime implementation, if required;
2. pre-outcome authority/funding/Telegram closure;
3. terminal economic evidence and durable registry updates;
4. Drive verification record.

Non-force push only. Finish with clean main and task worktree.

Complete `drive_handoff: approved_default`, round-trip verify every compact file by size and SHA-256, and retain the complete local archive.

## Final response

```text
status:
actual_starting_commit:
external_approval_hash_verified:
packet_manifest_dependency_hashes_verified:
preoutcome_review:
funding_extension_coverage:
Telegram_notifications:
supervisor_and_run_root:
campaign_runtime_and_resources:
cells_planned_executed_skipped:
phase_2_response_surface_summary:
phase_3_frozen_candidates:
phase_4_outer_fold_results:
phase_5_rolling_results:
multiplicity_results:
family_routes_and_decisions:
controls_phase_6_executed: no
protected_rows_opened: no
Capitalcom_payload_opened: no
tests_and_reviews:
artifact_manifest_hash:
files_and_commits:
origin_main_updated:
original_checkout_final_status:
task_worktree_final_status:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
human_approval_required:
```
