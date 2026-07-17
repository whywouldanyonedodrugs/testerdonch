# Stage 7C — Resume Kraken Analytics Acquisition Under Revised Storage and Semantic Policy

```text
task_id: donch_bt_stage_7c_resume_analytics_acquisition_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
acquisition_authorized: yes — exact scope below
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
private_endpoint_access: no
orders_or live-state changes: prohibited
paid_vendor_data: prohibited
persistent_unattended_execution: approved for this task
```

## Exact human decisions

The human approves these changes for this acquisition only:

1. Unresolved economic units/sign details are **not acquisition blockers** when exact raw payloads, positional schemas, timestamps, endpoint identity, and source hashes are preserved. They remain blockers for economic interpretation until separately adjudicated.
2. Replace the Stage 7B fixed `50 GiB` post-completion reserve floor with the revised storage policy below.
3. Start the resumable acquisition as soon as the focused implementation/review checks pass.
4. Do not delete existing project data to create space. Only verified task-local staging parts may be removed after successful deterministic packing/compaction.

## Supersession and continuation

Continue from the reviewed Stage 7B implementation and corrected Phase A evidence:

```text
Stage 7B final commit:
    35882880317f691219041d22daaf28c9e582a2bb

Stage 7B implementation commit:
    9dbf9e8

Stage 7B task archive:
    docs/agent/task_archive/20260717_donch_bt_stage_7b_resumable_analytics_acquisition_20260717_v1

corrected Phase A run root:
    results/rebaseline/phase_kraken_futures_analytics_acquisition_20260717_v1_20260717_195535

corrected data root:
    /opt/parquet/kraken_derivatives/analytics/stage7b_v1_attempt2

frozen symbol inventory SHA-256:
    0e1c35b3348b3187547f188dac47e5e0f5bc3d259207fa3884bc8efccdd36f09

included identities:
    460

excluded identities:
    19

Stage 7B data manifest:
    4f6500d1e85ecedd618c5062b191de80c5fc76d3fe413d738d55527ad27d62c9

Stage 7B data-content hash:
    8157e882bf7680eb8f2e6ec7082620ee871b52a205b34184667e7964a550b5cd
```

The quarantined first attempt remains immutable diagnostic provenance. Do not reuse its payloads.

## archive_context

```text
decision_ids:
    C01 level3_no_primary_pass_stop
    C02 level3_no_primary_pass_stop
    C03 C03_PIT_authority_unavailable
    C16 C16_flow_authority_unavailable
    Stage 7A partial_historical_analytics_requires_review
    Stage 7B blocked_by_units_pagination_or_storage

evaluation_conclusions:
    corrected Phase A verified pagination, inclusive bounds,
    deterministic replay, raw/Parquet hashing, and safe resume;
    three analytics types have historical retention;
    funding is empty and excluded

source_names/hashes:
    frozen symbol inventory and Stage 7B hashes above;
    official Kraken Futures analytics documentation;
    official Kraken Pro analytics-widget documentation;
    official Kraken Futures glossary

rejected_alternatives:
    all-symbol one-minute acquisition
    funding acquisition
    deleting existing project evidence
    requests without explicit pre-2026 upper bounds
    inferring economic edge while downloading
    relabeling basis as funding

gaps retained:
    OI four-value tuple component meanings unresolved;
    liquidation exact currency/unit and side split unresolved;
    basis numeric unit unresolved

approval_scope:
    broad five-minute acquisition for 460 frozen identities;
    one-minute acquisition for BTC and ETH;
    open-interest, liquidation-volume, and future-basis only
```

## Objective

Resume and complete a crash-safe local acquisition for:

```text
period:
    [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)

metrics:
    open-interest
    liquidation-volume
    future-basis

five-minute scope:
    all 460 frozen included identities supported by the endpoint

one-minute scope:
    PF_XBTUSD
    PF_ETHUSD
```

No other metric, symbol, resolution, or date range is authorized.

No signal construction, price/return join, threshold selection, ranking, PnL, MAE/MFE, or economic test is permitted.

## Semantic policy

Separate:

```text
source/acquisition authority
from
economic interpretation authority
```

Acquisition may proceed with these statuses:

### Future basis

```text
field identity:
    verified

sign convention:
    verified from official Kraken documentation:
    positive means futures above spot;
    negative means futures below spot

numeric unit:
    unresolved pending separate adjudication
```

Preserve every returned field, including `basis`, `usdValue`, or any schema variant. Do not silently discard fields.

### Liquidation volume

```text
concept:
    official total value of positions force-closed in the interval

side/direction:
    aggregate endpoint has no verified long/short split;
    absence of a side split is not an acquisition blocker

exact numeric unit/currency:
    unresolved pending separate adjudication
```

Store the exact scalar/string and schema metadata. Do not label it USD, contracts, base units, long liquidations, or short liquidations without later authority.

### Open interest

```text
concept:
    official open-interest measure of outstanding/open positions

returned schema:
    four-string tuple per timestamp

component meanings and exact unit:
    unresolved pending separate adjudication
```

Store all four values losslessly as positional raw fields:

```text
value_0_raw
value_1_raw
value_2_raw
value_3_raw
```

Do not label them open/high/low/close, contracts, base units, or USD until proven.

### General

For all metrics retain:

```text
semantic_status
schema_hash
source_request_id
original timestamp
original exact decimal strings
```

Do not down-cast exact strings. Parsed decimal helper columns are allowed only with round-trip equality and may not replace raw fields.

Economic use remains blocked until a separate semantics contract approves the required interpretation.

## Revised storage authorization

Stage 7B measured:

```text
filesystem total:
    160,970,244,096 bytes

free before full acquisition:
    56,093,024,256 bytes

projected raw plus Parquet:
    11,470,631,998 bytes

projected with 25% contingency:
    14,338,289,997 bytes

projected free after contingency:
    41,754,734,259 bytes
```

The fixed 50-GiB reserve floor is withdrawn for this exact task.

Use:

```text
pre-start projection gate:
    projected free after 25% contingency must remain
    at least 25% of the target filesystem total

25% threshold at the measured filesystem size:
    40,242,561,024 bytes

runtime warning threshold:
    40,242,561,024 bytes free

runtime hard-stop threshold:
    32,194,048,819 bytes free
    (20% of measured filesystem total)
```

Before launch, remeasure the filesystem. Recalculate the percentages from the current total size. Start only if the updated projection passes the 25% pre-start gate.

During acquisition:

- check free bytes and inodes at least every five minutes and before compaction;
- warn and increase compaction priority below 25%;
- stop cleanly and preserve resume state below 20%;
- do not delete any pre-existing project file;
- do not delete immutable Phase A or prior-task evidence;
- do not change the acquisition scope silently.

A hard storage stop returns `acquisition_partial_resume_ready`, not failure.

## File-count and compaction amendment

Do not leave the projected 506,520 page-level raw/Parquet files as the final layout.

### Deterministic shards

Assign the frozen 460-symbol inventory to immutable shards of at most 32 symbols using frozen inventory order. Record the shard map and hash it before launch.

### Acquisition order

Run in this order:

```text
1. one-minute PF_XBTUSD/PF_ETHUSD for all three metrics;
2. five-minute PF_XBTUSD/PF_ETHUSD;
3. remaining five-minute symbols in frozen shard order.
```

This order is based on predefined data utility, not analytics values.

### Final normalized layout

Produce one final Parquet file per:

```text
analytics_type
interval
UTC year
UTC month
symbol_shard
```

Example:

```text
normalized/analytics_type=open-interest/interval=300/year=2024/month=06/shard=003/data.parquet
```

Sort deterministically by:

```text
timestamp
symbol
source_request_id
```

Use Zstandard compression and bounded-memory DuckDB/Arrow/Parquet processing. Final files must retain symbol as a column.

### Raw source bundles

Page-level exact responses may be staged temporarily. After every shard-month is complete:

1. create one deterministic `.tar.zst` source bundle;
2. include a JSONL index containing member path, request ID, response bytes, uncompressed SHA-256, compressed member SHA-256, row count, and timestamp bounds;
3. verify every member against the SQLite ledger;
4. verify the bundle by extracting into a temporary verification directory;
5. atomically publish the bundle and bundle manifest;
6. only then delete that shard-month's task-local staging response files.

This deletion is approved only for verified Stage 7C staging files whose contents are present in a verified immutable source bundle. It does not authorize deletion of prior evidence or unrelated data.

### Staging bound

Schedule jobs shard-month-first so no more than two shard-months remain uncompacted at once. On restart, finish or safely rebuild incomplete bundles.

### Final file-count target

Final raw bundles plus final Parquet parts should remain below 5,000 files, excluding ledgers, manifests, logs, and preserved Phase A evidence. Report the exact count.

## Resumability and unattended execution

Reuse the reviewed transactional job-ledger implementation. Preserve one row per request/page and exact request identity.

Requirements:

- stale `running` jobs reset safely;
- completed jobs skipped only after raw-bundle and final-Parquet verification;
- exact request retries only;
- at most two concurrent response bodies;
- peak downloader/compactor RSS below 2 GiB;
- atomic writes and fsync checkpoints;
- SIGINT/SIGTERM-safe checkpointing;
- heartbeat at least every five minutes with job counts, metric, interval, shard/month, requests, rows, bytes, free space, inode use, RSS, and last error;
- no interactive prompt after launch.

Prefer remaining attached until completion. If the agent execution environment cannot remain attached, launch through an available reviewed persistent local mechanism, record the exact PID/service/session, command, log, heartbeat, and restart command, and do not mark the task complete while it is still running.

A running handoff must be sufficient for another agent to verify and resume without human diagnosis.

## Endpoint and data rules

Preserve the Stage 7B verified mechanics:

```text
maximum rows per response:
    2,000

bounds:
    inclusive lower and upper

research end-exclusive request rule:
    to = end_exclusive - interval

continuation:
    start at last returned timestamp;
    remove only exact timestamp/value boundary duplicates
```

Conflicting duplicates fail closed. Missing intervals remain missing. No fill or interpolation.

Reject any response carrying a timestamp at or after `2026-01-01T00:00:00Z` before traversing its analytics values. Preserve a fail-fast protocol incident if encountered.

## Pre-launch work

Before launching acquisition:

1. implement the shard-month packing/compaction amendment;
2. update the storage policy exactly as approved;
3. add semantic-status/raw-field schemas;
4. test crash recovery during staging, bundling, compaction, and cleanup;
5. run focused and applicable repository guards;
6. obtain independent review of the actual diff, commands, projection, hard stops, and no-economic scope;
7. commit the reviewed acquisition changes;
8. launch immediately after review approval.

Do not repeat the completed Phase A request matrix except for minimal code-level verification fixtures. Reuse its authoritative results.

## Required outputs

```text
KRAKEN_ANALYTICS_STAGE7C_SYMBOL_SHARDS.csv
KRAKEN_ANALYTICS_SEMANTIC_STATUS.md
KRAKEN_ANALYTICS_REVISED_STORAGE_AUTHORIZATION.md
KRAKEN_ANALYTICS_ACQUISITION_PLAN.json
KRAKEN_ANALYTICS_JOB_LEDGER.sqlite
KRAKEN_ANALYTICS_HEARTBEAT.json
KRAKEN_ANALYTICS_REQUEST_LEDGER.parquet
KRAKEN_ANALYTICS_COVERAGE_MATRIX.csv
KRAKEN_ANALYTICS_GAP_REGISTER.parquet
KRAKEN_ANALYTICS_RAW_BUNDLE_INDEX.parquet
KRAKEN_ANALYTICS_DATA_MANIFEST.json
KRAKEN_ANALYTICS_ACQUISITION_SUMMARY.md
HYPOTHESIS_DATA_READINESS_UPDATE.csv
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large raw bundles, Parquet data, SQLite ledger, logs, and gap tapes remain local and hash-manifested. Upload compact evidence only.

## Validation

Test at minimum:

- raw OI tuple preservation;
- all basis fields retained;
- liquidation raw scalar preservation;
- semantics remain non-economic;
- deterministic 32-symbol sharding;
- shard-month scheduling;
- raw bundle creation, extraction verification, and approved staging cleanup;
- bounded-memory Parquet compaction;
- crash/restart before and after bundle publication;
- no verified-part overwrite;
- free-space warning and hard stop;
- inode monitoring;
- exact pagination and duplicate handling;
- protected timestamp fail-fast before value traversal;
- final file-count target;
- deterministic manifests;
- zero signals, returns, or economic fields.

Require independent pre-launch and final reviews.

## Data-readiness implications

This task creates acquisition authority only.

After completion, report possible future data-contract readiness for:

```text
C04 crowding versus price progress
C08 basis/dislocation state
C09 liquidation/forced-flow state
```

Do not execute these hypotheses and do not reopen C01/C02.

C14 funding remains unchanged and blocked outside the existing exact slice.

## Status vocabulary

Return one of:

```text
historical_analytics_acquisition_complete
acquisition_running_resume_safe
acquisition_partial_resume_ready
blocked_by_runtime_storage_or_endpoint
```

## Drive handoff

At completion or a bounded resume-safe stop:

- update factual acquisition, source, data-capability, and continuity records;
- create reviewed task-scoped commits;
- non-force push under the standing workflow;
- use `drive_handoff: approved_default`;
- round-trip verify all uploaded compact files;
- retain local data and resume state.

## Final response

```text
status:
actual_starting_commit:
acquisition_implementation_commit:
terminal_record_status:
semantic_policy_applied:
revised_storage_gate:
prelaunch_free_space_and_projection:
persistent_worker_identity:
acquisition_scope_completed:
request_and_row_counts:
raw_bundle_and_parquet_roots:
final_file_counts:
coverage_and_gaps:
protected_rows_opened:
economic_outputs_computed: no
peak_RSS_and_runtime:
resume_state:
data_manifest_and_hashes:
hypothesis_readiness_update:
tests_and_reviews:
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
rollback:
human_approval_required:
```
