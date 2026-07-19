# Stage 15 — Repair Launch Binding and Run the Approved Unattended Derivatives Campaign

```text
task_id: donch_bt_stage_15_unattended_derivatives_campaign_20260720_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
economic_campaign_authorized: yes — exact Stage-14 packet plus supplied human approval only
approved_phases: 2, 3, 4, 5
approved_hypotheses:
  - KDA02B_v2_oi_vacuum_redevelopment
  - KDA02C_v1_purge_breadth_context
  - KDX01_v1_downside_completed_derivatives_state_rejection
controls_phase_6_authorized: no
protected_outcome_access: no
Capitalcom_payload_access: no
new_data_acquisition: no
orders_or_live_trading: no
commit_authorized: yes
push_authorized: yes — non-force
```

## Exact human approval

The supplied file is the external human-approval artifact:

```text
HUMAN_APPROVAL_Kraken_Derivatives_Campaign_001_Phases_2_5_2026-07-20_v1.json
SHA-256: c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b
```

It explicitly approves Stage-14 approval packet:

```text
approval packet file SHA-256:
    c029830526ab7a8cd443acb0f9ec5441e1e50967200c8a95184cf8b46d71c313

campaign manifest file SHA-256:
    61119ce30ff142efdf0b40e8796e5ffb2efae0764f835b64e179b624db05f632
```

Do not broaden this approval.

## Current authority to verify

```text
expected starting commit:
    50dffb791c146b359cb210532e5f7291774e26f0

Stage-14 implementation commit:
    b4785ed2a06fbed50d20b7dcdf0bc27e93cd7bea

Stage-14 approval packet:
    c029830526ab7a8cd443acb0f9ec5441e1e50967200c8a95184cf8b46d71c313

Stage-14 campaign manifest:
    61119ce30ff142efdf0b40e8796e5ffb2efae0764f835b64e179b624db05f632

search registry:
    a533c8e507f78989963a2631115bd1a3f70c3dc34cb59d749bec57b53577294b

resource projection:
    6eb572e8c09be94d462702db54d98d3fa88b4aa0ee402b2873c8a7fe66235d97

local state-tape manifest:
    838be528ea0bcaeed132c4b95401594e2bb9a726178ece3f51de44dbbcd96380

analytics manifest:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

authorized cohort:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636
```

Verify root, `AGENTS.md`, Git/remotes/worktree, current task archives, data roots, available RAM/disk/CPU, campaign state, packet bytes and Drive target.

## Known pre-launch compatibility defects

Donch independently found that the Stage-14 packet is semantically accepted but cannot safely launch through the Stage-13 engine without a bounded repair:

1. the engine hashes canonical JSON while Stage 14 binds exact file-byte SHA-256;
2. the engine expects `phase_permissions_requested` / `candidate_list`, while Stage 14 uses `phases_requested` / `ready_lanes`;
3. the readiness manifest has Phases 2–5 false, and the current engine rejects before evaluating external approval;
4. the funding contract references a predeclared coverage floor without serializing a numeric value;
5. Telegram notification is absent from the packet.

These are launch-control defects, not economic-definition permission.

## Pre-outcome compatibility repair

Before opening any outcome:

- update the campaign approval validator to support the exact Stage-14 schema;
- verify both exact file-byte hashes and canonical parsed hashes;
- permit Phases 2–5 only when the exact external approval validates;
- keep the readiness manifest non-self-authorizing;
- enforce the supplied approval's funding coverage and Telegram constraints;
- add focused adversarial tests for altered packet bytes, altered canonical content, alias-field substitution, phase widening, hypothesis widening, missing state, missing Telegram, inadequate funding coverage and direct permission bypass.

Do not change hypotheses, cells, folds, costs, selection objectives, candidate beams, execution rules or multiplicity.

Obtain an independent pre-outcome review. If the repair requires any economic-contract or packet-content change, stop and regenerate a new packet; do not run.

## Funding extension before outcomes

Apply frozen funding model:

```text
0054af0ee40740e39739bfade92f342867bb208a4fe7ed15b151a8a0a838d072
```

to registered 2023 boundaries with unchanged parameters and PIT inputs.

Enforce:

```text
minimum coverage per hypothesis-fold: 90%
minimum campaign-weighted coverage: 95%
```

Missing boundaries fail the event closed and are reported. Never relabel them as zero funding. Funding coverage or missingness may not select candidates.

Complete and independently verify this extension before any forward outcome reader opens.

## Telegram launch gate

Use only an existing securely configured Telegram notifier. Do not print, archive, commit or return bot tokens, chat IDs or secret file contents.

Before outcomes:

1. identify the existing notifier integration and configuration presence without exposing values;
2. send a dry-run message containing campaign ID, starting commit, approval-packet hash and `preflight`;
3. verify successful delivery from the notifier response/log;
4. test one synthetic heartbeat and one synthetic stop alert.

If secure Telegram configuration is unavailable or the test fails, stop before outcomes with:

```text
blocked_telegram_notifier_unavailable
```

Do not request or handle credentials in chat.

Required notifications:

- preflight passed;
- campaign started;
- every phase transition;
- each fold completed;
- family stop;
- global warning/stop;
- terminal completion;
- heartbeat every 30 minutes.

Heartbeat content:

```text
campaign ID
phase / fold / family
completed cells / total cells
elapsed and ETA
worker count
RSS and free disk
error/retry counts
heartbeat age
```

Do not include economic results in routine Telegram messages before terminal completion.

## Preflight and unattended launch

Run, in order:

1. exact authority/hash verification;
2. compatibility tests;
3. funding extension and coverage verification;
4. campaign-state initialization/resume test;
5. resource check against four workers, four-hour wall cap and 5 GiB campaign cap;
6. synthetic end-to-end campaign canary;
7. independent pre-outcome review;
8. Telegram dry-run;
9. launch.

Use a persistent supervised process, preferably repository-supported `systemd --user`; use `tmux` only if systemd-user supervision is unavailable. Persist PID/service identity, run root, logs, state, heartbeat and exact stop reason.

After launch, observe until:

- the persistent process remains healthy;
- at least one real registered cell completes;
- one scheduled Telegram heartbeat is delivered;
- campaign state and artifact hashes reconcile.

Then leave the campaign unattended. Do not tune, inspect partial winners or intervene based on outcomes.

The agent remains responsible for automatic monitoring, terminal collection, independent post-run review, archive and final response.

## Bound campaign

```text
lanes:
  KDA02B: 96 cells
  KDA02C: 48 cells
  KDX01: 84 cells

total cells:
  228

folds:
  27 programme-exposed quarterly records

candidate beam:
  maximum 5 per family per fold

workers:
  maximum 4

wall cap:
  14,400 seconds

campaign disk cap:
  5 GiB
```

Use exact Stage-14 search spaces, folds, deterministic tie-break, selection objectives, costs and stop rules.

Routine family failure stops that family only. Shared authority, protected exposure, timestamp defect, unsafe storage/Git or deterministic replay failure stops the campaign globally.

## Phase requirements

### Phase 2 — registered development

- execute every registered cell or record its exact stop/skip reason;
- use development data only;
- record all response bins, models, directions, horizons and contexts;
- no hidden variants;
- no outer-fold information.

### Phase 3 — deterministic freeze

- apply the exact Pareto and tie-break contract;
- retain at most the bound candidate beam;
- freeze each candidate before opening its evaluation block;
- independent deterministic freeze verification.

### Phase 4 — next-block evaluation

- first authorized PF five-minute opens at or after decision/target;
- maximum ten-minute delay;
- actual-exit non-overlap;
- 14/32-bps pre-funding costs;
- exact or conservative-adverse funding partition contract;
- no evaluation information feeds backward.

### Phase 5 — forward rolling replication

Continue through all authorized folds without routine human approval. Preserve fold-by-fold candidate identity and results. Do not pool away failed folds.

## Outputs

At minimum:

```text
HUMAN_APPROVAL.json
PREOUTCOME_COMPATIBILITY_REPAIR.md
FUNDING_BOUNDARY_EXTENSION_MANIFEST.json
TELEGRAM_NOTIFICATION_VALIDATION.md
CAMPAIGN_LAUNCH_MANIFEST.json
CAMPAIGN_STATE.json
CAMPAIGN_HEARTBEAT.json
EXPLORED_CELL_REGISTRY
CANDIDATE_BEAM_REGISTRY
FROZEN_TRANSLATION_REGISTRY
FOLD_RESULTS
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

Large outcomes remain local and hash-manifested; compact evidence goes to Drive.

## Claim boundaries

All results remain:

```text
programme_exposed_historical
not independent validation
not live-ready
```

A positive candidate may be routed as unconditional, conditional, narrow, convex, execution-sensitive or sample-limited under policy. Do not validate or deploy it.

## Tests and review

Require:

- focused compatibility and approval-bypass tests;
- exact packet/manifest/raw/canonical hash checks;
- no protected or Capital.com rows;
- funding extension and coverage checks;
- Telegram dry-run without secret leakage;
- deterministic cell, beam and freeze replay;
- exact execution/cost/funding arithmetic;
- fold embargo and no backward leakage;
- complete 228-cell reconciliation;
- post-run independent recomputation of selected candidates, fold results, multiplicity and routes;
- secret scan and `git diff --check`.

## Commit, push and handoff

Pre-outcome compatibility changes may be committed and pushed only after independent review. Economic outputs receive task-scoped reviewed commits after terminal completion. Non-force push only.

Complete `drive_handoff: approved_default`, round-trip verify all compact artifacts and retain the complete local archive.

## Stop conditions

Stop before outcomes for any approval/hash/schema/funding/notifier/preflight failure.

Stop during campaign under the bound family/global rules. Preserve state and evidence. Never restart with altered definitions or packet contents.

## Final response

```text
status:
actual_starting_commit:
approval_artifact_hash_verified:
compatibility_repair_commit_and_review:
funding_extension_coverage:
telegram_notifications:
supervisor_and_run_root:
campaign_runtime_and_resources:
cells_planned_completed_stopped:
phase_2_results:
phase_3_frozen_candidates:
phase_4_5_fold_results:
multiplicity_results:
family_routes_and_decisions:
protected_rows_opened: no
Capitalcom_payload_opened: no
controls_phase_6_executed: no
tests_and_reviews:
artifact_manifest_hash:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
human_approval_required:
```
