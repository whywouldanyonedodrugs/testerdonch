# Stage 8B1 — KDA01 Independent-Episode and Execution-Contract Closure

```text
task_id: donch_bt_stage_8b1_kda01_contract_closure_20260719_v1
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
new_data_acquisition: no
commit_authorized: yes — task-scoped reviewed commits
push_authorized: yes — non-force under the standing workflow
```

## Objective

Close three pre-economic review gaps in the otherwise accepted Stage 8B KDA01 v2 package:

1. provide the omitted artifact manifest in the verified handoff;
2. replace symbol-specific parent episodes as the primary inference cluster with a cross-symbol market-day cluster;
3. freeze maximum entry and exit delays for missing five-minute bars.

Produce a new fully serialized and hashed Level-3 contract. Do not calculate returns, PnL, MAE/MFE, funding outcomes, control outcomes, or any other economic output.

## Authority to verify

```text
Stage 8B starting/published commit:
    2a3d38545600eb39f70f91180fb237bc436a1ece

Stage 8B task archive:
    docs/agent/task_archive/20260719_donch_bt_stage_8b_kda01_prerun_freeze_20260719_v1

Stage 8B Level-3 contract hash:
    2eef5efb631e49014ea239eef5b90d4f2d5932fdcd33e97ec26067b5288ef938

Stage 8B feature contract hash:
    6934f48dceae6a12c92198a689d773206710b91b543e263330164b856364b157

Stage 8B generator hash:
    aa83c46a73068872986277741407d05ab0156a5660cb97bfd5d453401123d017

Stage 8B reported artifact-manifest hash:
    569bb1c53ba58767091c42dbf96ead63b3df5a45ae7cad1b4f6d4425e5ef61d5

parent tape SHA-256:
    ae470d06c7da049d145ac8e083d0c4711c9c2dd6e4ec32fedf79224e8885ddbd

event tape SHA-256:
    7c9e682380681aa6fc83161ae29dafab9a047bdb19fe1fa8ade3ba5a4eb9c2e5
```

Verify repository identity, `AGENTS.md`, clean synchronized Git state, data roots, task archives, and Drive target.

Preserve Stage 8A and Stage 8B outputs unchanged as provenance. Preserve all C01/C02/C03/C16 decisions and same-sample prohibitions.

## Finding 1 — handoff completeness

The verified Stage 8B ZIP did not contain `ARTIFACT_MANIFEST.json`, although `READ_FIRST.md` said the large local tapes and shards were hash-manifested there.

Locate the authoritative local manifest and verify:

```text
SHA-256:
    569bb1c53ba58767091c42dbf96ead63b3df5a45ae7cad1b4f6d4425e5ef61d5
```

It must bind the exact Stage 8B parent tape, event tape, all 187 completed shards, contracts, code/config identity, sizes, and SHA-256 values.

If it is missing or its hash/content does not reconcile, stop with the exact blocker. Do not reconstruct it from prose without verifying every local object.

The replacement handoff must include the verified artifact manifest.

## Finding 2 — independent market episodes

Stage 8B's parent episode is symbol-specific. It prevents repeated same-symbol decisions but does not cluster simultaneous events across symbols during a common crypto-market shock.

Do not change event generation. Add these outcome-free identities to every Stage 8B event:

```text
parent_onset_utc_date:
    UTC calendar date containing parent_onset

market_day_cluster_id:
    SHA-256 of:
      translation_id
      attempt
      parent_onset_utc_date

market_6h_cluster_id:
    SHA-256 of:
      translation_id
      attempt
      floor(parent_onset to a fixed UTC six-hour block)
```

The primary inference cluster for the Level-3 contract becomes:

```text
market_day_cluster_id
```

This intentionally groups every symbol and both parent directions occurring on the same UTC parent-onset date within the same primary/robustness attempt.

`market_6h_cluster_id` is a descriptive sensitivity identity only. It cannot rescue a failure under the daily cluster.

Keep symbol-specific `parent_episode_id` for overlap control and actual-exit scheduling.

Report, without outcomes:

```text
number of market-day clusters by attempt and year
events per market-day cluster: min/median/p90/p99/max
symbols per market-day cluster: min/median/p90/p99/max
largest cluster shares
```

## Finding 3 — execution availability

Retain the frozen economic entry and timeout intent, but fail closed on long data gaps.

For each definition, using timestamps and data availability only:

```text
entry:
    first PF five-minute trade-bar open strictly after decision

maximum entry delay:
    10 minutes after the expected next five-minute open

exit target:
    entry timestamp + frozen timeout

exit:
    first PF five-minute trade-bar open at or after exit target

maximum exit delay:
    10 minutes after exit target
```

Clarification:

- The expected next open is the first five-minute grid timestamp strictly after decision.
- Reject the candidate when the actual entry open occurs more than 10 minutes after that expected timestamp.
- Reject the candidate when the exit open occurs more than 10 minutes after the timeout target.
- Do not read entry or exit prices in this task.
- Do not search alternative delay caps.

Create timestamp-only execution-eligibility counts by definition, year, symbol, and rejection reason.

All four primary branches must still satisfy the original mechanical gates after availability filtering:

```text
events >= 100
each year >= 20
symbols >= 20
maximum symbol share <= 25%
duplicates = 0
protected rows = 0
```

Do not weaken the generator or delay rules if a branch fails.

## Amended frozen Level-3 inference rules

Retain unchanged:

```text
16 frozen definitions
14 bps base round-trip cost
32 bps stress round-trip cost
1h and 6h timeout definitions
fixed notional
definition-local actual-exit non-overlap
robustness cannot rescue primary
funding partitions separate and excluded from Level-3 gates
10,000 bootstrap resamples
seed 20260719
```

Amend:

```text
primary bootstrap unit:
    market_day_cluster_id

primary concentration gate:
    maximum positive market-day contribution /
    total positive net contribution <= 10%
```

Retain the symbol and year concentration gates:

```text
maximum positive-symbol contribution <= 25%
maximum positive-year contribution <= 70%
```

Report parent-episode and six-hour-cluster results only as sensitivities. They cannot override the daily-cluster gate.

The remaining primary Level-3 gates remain:

```text
executed trades >= 100
trades per year >= 20
base-net mean > 0 bps
base-net median > 0 bps
market-day bootstrap 95% lower bound >= -5 bps
stress-net mean >= -10 bps
```

No result may be pooled across definitions, branches, horizons, directions, or attempts.

## Frozen controls

Preserve the existing seven controls unchanged and unexecuted:

```text
price_progress_path_without_oi_or_basis
material_oi_without_basis
directional_basis_without_oi
structural_failure_after_price_only_parent
ordinary_oi_basis_matched_parent_episodes
btc_eth_parent_state
kda01_v1_overlap_ablation_non_rescue
```

No caliper, threshold, horizon, or control change is authorized.

## Required outputs

```text
STAGE8B_HANDOFF_GAP_REPORT.md
ARTIFACT_MANIFEST.json
KDA01_V2_EVENT_CLUSTER_IDENTITY.parquet
KDA01_MARKET_CLUSTER_SUMMARY.csv
KDA01_LEVEL3_EXECUTION_AVAILABILITY_COUNTS.csv
KDA01_LEVEL3_EXECUTION_REJECTIONS.csv
KDA01_FINAL_LEVEL3_ECONOMIC_CONTRACT_V2.md
KDA01_LEVEL3_DECISION_RULES_V2.json
KDA01_LEVEL3_DEFINITION_REGISTER_V2.csv
KDA01_PRERUN_APPROVAL_PACKET_V2.md
VALIDATION.md
REVIEW.md
COMPLETION.md
NEXT_ACTION.md
TRANSFER_MANIFEST.json
```

Large Parquet files remain local and hash-manifested. The compact Drive ZIP must include the actual `ARTIFACT_MANIFEST.json` and the complete human-readable and machine-readable contract.

## Validation and review

Test at minimum:

- artifact-manifest path, size, and hash reconciliation;
- deterministic UTC date and six-hour cluster IDs;
- same-day cross-symbol clustering;
- primary/robustness separation;
- expected next-open calculation;
- exact 10-minute entry and exit caps;
- timestamp-only eligibility does not read prices;
- actual-exit non-overlap remains definition-local;
- deterministic contract serialization and hash;
- daily-cluster bootstrap identity is frozen;
- no outcome reader/column;
- no pre-2023 or 2026+ row;
- no protected data.

Run focused tests and relevant Stage 8A/8B, source-boundary, lifecycle, analytics, archive, and protected-boundary regressions.

Require independent review of the actual artifact manifest, cluster identity, delay rules, updated counts, complete contract, hashes, and replacement handoff.

## Decision

Return exactly one:

```text
ready_for_human_KDA01_Level3_run_approval
KDA01_mechanically_unavailable_after_contract_closure
blocked_with_exact_non_economic_remedy
```

No economic execution is authorized.

## Final response

```text
status:
actual_starting_commit:
Stage8B_artifact_manifest_verification:
handoff_gap_repaired:
market_day_cluster_counts:
cluster_size_summary:
execution_availability_counts:
execution_rejections:
primary_branch_feasibility_after_filter:
amended_Level3_contract_hash:
definition_count:
controls_frozen:
protected_rows_opened: no
economic_outputs_computed: no
tests_and_review:
files_and_commits:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
human_approval_required:
```
