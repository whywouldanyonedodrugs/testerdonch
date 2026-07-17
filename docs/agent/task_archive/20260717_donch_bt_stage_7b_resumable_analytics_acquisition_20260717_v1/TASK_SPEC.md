# Stage 7B — Resumable Kraken Futures Analytics Authority and Acquisition

```text
task_id: donch_bt_stage_7b_resumable_analytics_acquisition_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
private_endpoint_access: no
orders_or live-state changes: prohibited
paid_vendor_data: prohibited
public_data_acquisition: approved only for the exact analytics scope below
```

## Objective

Establish full rankable-period authority and, only after the predeclared gates pass, acquire a resumable local Kraken Futures analytics panel for:

```text
open-interest
liquidation-volume
future-basis
```

Target period:

```text
[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
```

Target resolutions:

```text
5 minutes:
    every unambiguous observed Kraken PF identity supported by the endpoint

1 minute:
    PF_XBTUSD
    PF_ETHUSD
```

Funding is excluded because Stage 7A returned valid empty responses. No other analytics type is acquired in this task.

This task may download and normalize the exact data above. It may not calculate signals, returns, PnL, MAE/MFE, predictive labels, thresholds, candidate rankings, or economic results.

## Accepted authority and archive context

```text
Stage 7A decision:
    partial_historical_analytics_requires_review

Stage 7A final commit:
    7cd2444c92c5aecedeb1a897c919c25cbfd8749d

Stage 7A retention-matrix SHA-256:
    c125265fa92a4befe59209db3c0eb41bca58ca408e8fb9d1c09de46d275d8e6a

Stage 7A request-ledger SHA-256:
    cc3a4faa984b5278c0b53c7282a1bace0627a92dfeec96bf638a92feb6bf6943

Stage 7A official-schema snapshot SHA-256:
    e4a6ebd27aef347c87c18b852b5069e5feb1b88e2660b73d6437dd85de710f45

C01:
    level3_no_primary_pass_stop

C02:
    level3_no_primary_pass_stop

C03:
    C03_PIT_authority_unavailable

C16:
    C16_flow_authority_unavailable
```

Preserve all terminal decisions. New analytics data do not reopen or rescue any earlier strategy.

## archive_context

```text
decision_ids:
    C01 Level-3 stop
    C02 Level-3 stop
    C03 PIT authority unavailable
    C16 official-flow authority unavailable
    Stage 7A partial historical analytics

evaluation_conclusion:
    Stage 7A verified reproducible bounded 2023/2024/2025 hourly rows
    for BTC/ETH OI, liquidation volume, and futures basis;
    funding remained empty

rejected_alternatives:
    downloading all analytics types without authority
    all-universe one-minute acquisition before storage/pagination audit
    funding acquisition from the empty analytics route
    requests without explicit pre-2026 upper bounds
    private endpoints or paid vendors

approval_scope:
    Phase A authority/pagination/storage audit;
    conditional Phase B exact 5m broad-universe acquisition;
    conditional Phase C exact 1m BTC/ETH acquisition
```

## Start checks

Verify:

- repository root and applicable `AGENTS.md` chain;
- clean synchronized `main` and actual starting commit;
- current machine contracts;
- Stage 7A artifacts and hashes;
- the authoritative C03 PF identity inventory;
- repository data-root policy;
- filesystem free space, inode availability, and mount type;
- official current Kraken analytics documentation.

Stop for divergent evidence, dirty Git state, missing source roots, or inability to enforce the protected boundary before response parsing.

## Phase A — Authority, pagination, and storage audit

Do not begin full acquisition until Phase A passes independent review.

### A1. Freeze the symbol inventory

From the C03 observed PF identity inventory, create:

```text
KRAKEN_ANALYTICS_FROZEN_SYMBOL_INVENTORY.csv
```

Include every unique `PF_*` perpetual identity with an unambiguous exact Kraken symbol. Exclude:

- futures maturities and non-perpetual products;
- unresolved identity collisions;
- non-Kraken identities;
- symbols whose exact endpoint identity cannot be established.

Retain excluded identities and reasons. Freeze and hash this inventory before analytics-value access beyond the audit cells.

### A2. Audit cells

Use:

```text
symbols:
    PF_XBTUSD
    PF_ETHUSD
    four additional symbols selected deterministically from the frozen inventory:
        earliest canonical asset ID among identities with existing PF trade/mark
        coverage in 2023, 2024, and 2025

analytics:
    open-interest
    liquidation-volume
    future-basis

resolutions:
    60 seconds
    300 seconds

windows:
    2023-01-01 to 2023-01-08
    2023-12-25 to 2024-01-01
    2024-06-01 to 2024-06-08
    2025-12-24 to 2026-01-01 exclusive
```

Every request must carry explicit `since` and `to` values below 2026.

### A3. Determine endpoint mechanics

Verify and document:

- maximum rows returned per response;
- meaning of `more`;
- correct continuation rule using the last returned timestamp;
- inclusive/exclusive lower and upper boundaries;
- duplicate boundary behavior;
- empty-window behavior;
- rate-limit and retry behavior;
- schema stability across symbols, years, types, and resolutions;
- timestamp unit and interval alignment;
- earliest and latest accessible pre-2026 interval;
- deterministic replay behavior.

Never assume that a single response covers the requested range.

### A4. Units and semantics

Establish each field as `verified`, `inferred`, `unavailable`, or `blocked`.

Minimum gate for acquisition:

```text
open-interest:
    exact value field and unit/contract interpretation verified or
    bounded by official schema plus repository-authoritative invariants

liquidation-volume:
    exact value field, direction structure if any, and unit verified or bounded

future-basis:
    exact basis field, sign convention, and unit verified
```

Do not infer units from future returns or profitability. If a metric's semantics remain materially ambiguous, exclude that metric from acquisition and return `blocked` or `partial` rather than guessing.

### A5. Storage benchmark

For every metric/resolution schema:

- measure raw JSON bytes per million rows;
- measure `.json.zst` bytes per million rows;
- measure normalized Parquet bytes per million rows;
- benchmark parse/write peak RSS and wall time;
- project request count, raw size, Parquet size, inode count, and duration for the exact target.

The full acquisition gate passes only if the projected combined raw-plus-Parquet footprint plus 25% contingency leaves at least:

```text
max(25% of the target filesystem, 50 GiB)
```

free after completion.

If the gate fails, do not delete existing data or silently reduce scope. Return an exact storage remedy and preserve the resumable downloader.

### A6. Independent pre-acquisition review

Review:

- frozen symbol inventory;
- endpoint paging and boundary logic;
- schemas and units;
- protected filters;
- disk projection;
- downloader design;
- synthetic tests.

Proceed only on review `approve`.

## Production downloader requirements

Create a production tool and tests, for example:

```text
tools/acquire_kraken_futures_analytics.py
unit_tests/test_acquire_kraken_futures_analytics.py
```

The downloader must be resumable and idempotent.

### Job ledger

Use SQLite or an equivalently transactional local ledger with one row per request chunk:

```text
job_id
analytics_type
symbol
interval_seconds
since
to
request_url_without_secrets
status
attempt_count
started_utc
completed_utc
HTTP status
response bytes
raw SHA-256
row count
first/last timestamp
schema hash
parquet path
parquet bytes
parquet SHA-256
validation status
error class/message
```

Statuses:

```text
planned
running
raw_verified
normalized
validated
complete
retryable_error
blocked_error
```

On restart, reset stale `running` jobs safely and skip only jobs whose raw and Parquet hashes still verify.

### Memory and process safety

- never concatenate the full panel in memory;
- process one bounded response chunk at a time;
- use at most two concurrent response bodies unless a measured lower-risk setting is approved by the pre-acquisition review;
- write Arrow/Parquet incrementally;
- maintain peak downloader RSS below 2 GiB;
- use temporary files on the target filesystem and atomic rename after validation;
- fsync ledgers/manifests at safe checkpoints;
- handle SIGINT/SIGTERM by finishing or safely rolling back the current chunk;
- never overwrite a verified part.

### Raw evidence

Retain every successful response as compressed immutable source evidence:

```text
raw/<analytics_type>/<symbol>/<interval>/<year>/<month>/<job_id>.json.zst
```

Record SHA-256 of both the exact uncompressed response bytes and the compressed file.

### Parquet layout

Use Zstandard-compressed Parquet with explicit stable schemas:

```text
normalized/<analytics_type>/interval=<60_or_300>/symbol=<PF_SYMBOL>/year=<YYYY>/month=<MM>/part-*.parquet
```

Requirements:

- timestamp stored as UTC nanosecond or millisecond logical type plus original epoch value;
- source/request/job identity retained;
- no float down-casting unless exact round-trip equivalence is proven;
- row groups sized from the benchmark, target approximately 64-256 MiB uncompressed;
- deterministic column order and metadata;
- no tiny-file explosion: compact only within the same metric/symbol/interval/month and only through atomic validated replacement;
- original raw evidence remains immutable.

### Boundary and coverage rules

- requests and reader filters must reject pre-2023 and 2026+ timestamps before normalization;
- inclusive endpoint duplicates must be deduplicated by exact source timestamp within metric/symbol/interval;
- conflicting duplicate values fail closed;
- preserve missing intervals as missing;
- never forward-fill or interpolate;
- create exact coverage and gap masks;
- verify no mixed protected payload.

### Rate limiting and retries

Use conservative deterministic throttling. Retry only exact requests with exponential backoff and jitter bounded by the task policy. Do not alter dates or chunk size to obtain more favorable values. Stop on repeated 429/5xx rates above the predeclared threshold and preserve resume state.

## Phase B — Broad five-minute acquisition

After Phase A approval, acquire exactly:

```text
analytics:
    every Phase-A-approved metric among:
        open-interest
        liquidation-volume
        future-basis

symbols:
    all endpoint-supported identities in the frozen symbol inventory

interval:
    300 seconds

period:
    [2023-01-01, 2026-01-01)
```

Unsupported symbols remain explicit in a coverage ledger. Do not substitute current roster or infer lifecycle continuity.

## Phase C — BTC/ETH one-minute acquisition

After Phase B completes and validates, acquire exactly:

```text
analytics:
    same Phase-A-approved metrics

symbols:
    PF_XBTUSD
    PF_ETHUSD

interval:
    60 seconds

period:
    [2023-01-01, 2026-01-01)
```

Do not expand one-minute acquisition to additional symbols in this task.

## Required outputs

```text
KRAKEN_ANALYTICS_FROZEN_SYMBOL_INVENTORY.csv
KRAKEN_ANALYTICS_ENDPOINT_MECHANICS.md
KRAKEN_ANALYTICS_SCHEMA_AND_UNIT_AUTHORITY.md
KRAKEN_ANALYTICS_STORAGE_BENCHMARK.csv
KRAKEN_ANALYTICS_ACQUISITION_PLAN.json
KRAKEN_ANALYTICS_JOB_LEDGER.sqlite
KRAKEN_ANALYTICS_REQUEST_LEDGER.parquet
KRAKEN_ANALYTICS_COVERAGE_MATRIX.csv
KRAKEN_ANALYTICS_GAP_REGISTER.parquet
KRAKEN_ANALYTICS_DATA_MANIFEST.json
KRAKEN_ANALYTICS_ACQUISITION_SUMMARY.md
HYPOTHESIS_DATA_READINESS_UPDATE.csv
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large raw responses, Parquet data, SQLite ledger, logs, and gap tapes remain local and hash-manifested; omit them from Drive. Include compact schemas, summaries, coverage tables, and manifests in the closed handoff.

## Validation

Use synthetic and bounded endpoint fixtures for:

- `more` and continuation paging;
- inclusive endpoints and duplicate boundaries;
- conflicting duplicates;
- empty and unsupported responses;
- exact lower/upper protected bounds;
- schema drift;
- unit-status gating;
- atomic writes and crash recovery;
- stale job recovery;
- exact request retries;
- raw and Parquet hash verification;
- monthly compaction;
- disk/inode safety stops;
- request and memory caps;
- deterministic replay;
- zero economic fields.

Run relevant protected-boundary, manifest, identity, PF semantic, funding-authority, archive, and continuity tests.

Require independent review before acquisition and after finalization.

## Data-readiness implications

Update capability only. Do not execute any hypothesis.

Report bounded readiness for:

```text
C04 crowding versus price progress
C08 basis/dislocation state
C09 liquidation and forced-flow completion
C14 basis/funding-boundary survival
```

Funding remains limited to existing authority. Do not claim that basis is funding.

Earlier C01/C02 and older price-pattern families may be used only for future overlap/control design. Do not filter, retune, or rescue them using these newly acquired variables on the already inspected sample.

## Decision vocabulary

Return exactly one:

```text
historical_analytics_acquisition_complete
acquisition_partial_resume_ready
blocked_by_units_pagination_or_storage
```

No economic work is authorized.

## Integration and Drive handoff

After review:

- update factual source, data-capability, family-readiness, acquisition, and continuity records;
- create reviewed task-scoped commits;
- non-force push under the standing workflow;
- package compact evidence using `drive_handoff: approved_default`;
- round-trip verify bytes and SHA-256;
- retain all local source and normalized data.

## Final response

```text
status:
actual_starting_commit:
terminal_record_status:
frozen_symbol_inventory:
approved_metrics_and_units:
endpoint_pagination_and_boundaries:
storage_projection_and_gate:
acquisition_scope_completed:
request_count_and_bytes:
raw_and_parquet_roots:
row_counts_by_metric_resolution:
coverage_and_gap_summary:
protected_rows_opened: no
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
