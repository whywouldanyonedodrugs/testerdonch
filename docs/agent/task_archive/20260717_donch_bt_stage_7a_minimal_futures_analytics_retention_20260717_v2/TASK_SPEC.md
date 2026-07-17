# Stage 7A — Minimal Kraken Futures Analytics Retention Probe

```text
task_id: donch_bt_stage_7a_minimal_futures_analytics_retention_20260717_v2
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
```

## Supersession

This prompt supersedes, before execution:

```text
DONCH_TO_BACKTESTING_Stage_7A_Kraken_Futures_Public_Analytics_Authority_Preflight_2026-07-17_v1.md
SHA-256:
224e91ee8d232b5f32d18ea7885d94516e47173bb05f7e9df2270e266d1f7e63
```

Do not execute the broader v1 request matrix or public order-event pilot.

## Objective

Answer one narrow question:

> Does Kraken's official public Futures analytics endpoint actually return reproducible pre-2026 historical rows for open interest, funding, liquidation volume, and futures basis?

This is a retention and source-authority probe only. Do not acquire a full history, calculate signals or returns, join to Kraken price paths, select thresholds, or run economics.

## Archive context

Preserve exactly:

```text
C01:
    level3_no_primary_pass_stop

C02:
    level3_no_primary_pass_stop

C03:
    C03_PIT_authority_unavailable

C16:
    C16_flow_authority_unavailable

current expected starting commit:
    f0c12311f31c24c2683d180166450bbacf7389bf
```

Historical full-period OI remains `unavailable`. Exact funding remains verified only for the existing narrow late-2025 slice. This task tests whether the newly identified public analytics interface changes those capability records.

## archive_context

```text
decision_ids:
    C01 Level-3 stop
    C02 Level-3 stop
    C03 PIT authority unavailable
    C16 official-flow authority unavailable

evaluation_conclusion:
    earlier project evidence did not establish usable 2023-2025
    historical OI or full-period exact funding

source_to_probe:
    official Kraken Futures Charts market-analytics endpoint

rejected_alternatives:
    broad 700-request audit
    public order-event pilot
    current since-inception downloads
    private/authenticated endpoints
    paid vendors
    inferring availability from documentation alone

approval_scope:
    24 fixed data requests plus one exact deterministic replay
    and official documentation snapshots only
```

## Start checks

Verify:

- repository root and applicable `AGENTS.md` chain;
- clean synchronized `main`;
- current machine contracts;
- current data-capability and continuity records;
- official endpoint documentation and supported parameter names.

Record the actual starting commit. Stop if the endpoint cannot be called with an explicit upper time bound before any response body is opened.

## Official endpoint

Use only the official public Kraken Futures analytics interface documented for:

```text
/api/charts/v1/analytics/:symbol/:analytics_type
```

Archive the exact official documentation page and schema before probing.

Do not call:

- public order-event history;
- raw Futures trade history;
- private or authenticated endpoints;
- trading endpoints;
- requests without both lower and upper pre-2026 bounds.

## Frozen probe matrix

### Symbols

```text
PF_XBTUSD
PF_ETHUSD
```

### Analytics types

Use the exact endpoint vocabulary verified from the official documentation for:

```text
open interest
funding
liquidation volume
futures basis
```

If the documented route names differ, record the official names and use only those exact names. Do not substitute related metrics.

### Interval

```text
1 hour
```

### UTC windows

```text
2023-06-15T00:00:00Z to 2023-06-16T00:00:00Z
2024-06-15T00:00:00Z to 2024-06-16T00:00:00Z
2025-07-15T00:00:00Z to 2025-07-16T00:00:00Z
```

This produces exactly:

```text
2 symbols × 4 types × 3 windows = 24 core requests
```

Do not add other symbols, intervals, dates, or analytics types.

## Request limits

```text
core requests:
    24

deterministic replay:
    the same 24 requests once

maximum data requests:
    48

maximum total downloaded response bytes:
    50 MB
```

Documentation snapshots and ordinary HTTP headers may be archived separately.

Use deterministic throttling. No retry may change a date, symbol, type, interval, or bound. Network retries may repeat the exact request at most twice and must be logged separately.

## Required checks per request

Record:

```text
request_id
symbol
analytics_type
interval
since
to
full request URL with secrets absent
HTTP status
response content type
response bytes
response SHA-256
schema keys
row count
first timestamp
last timestamp
minimum and maximum returned timestamp
null/non-finite counts
server error fields
upper-bound honored
2026 rows present
```

Classify each cell:

```text
verified_historical_rows
empty_valid_response
unsupported_type_or_symbol
recent_only_or_bound_ignored
schema_or_unit_ambiguous
request_failed
blocked
```

Never infer historical availability from HTTP 200 alone.

## Semantics and units

For every returned field, record only:

```text
verified
inferred
unavailable
blocked
```

A unit or sign may be `verified` only from official schema text or an existing repository authority artifact. Do not infer units by fitting values to prices or outcomes.

This task does not need to prove final economic usability. It must determine whether historical rows exist and whether their timestamps, schema, and basic semantics are stable enough to justify a later bounded audit.

## Deterministic replay

After the first 24-request matrix completes:

1. rerun the exact same 24 requests;
2. compare status, schema, row count, timestamp bounds, and response SHA-256;
3. explain any differences;
4. do not perform a third matrix.

If the endpoint is documented as mutable, a hash difference may be reported without being treated automatically as failure, but the changed rows and source behavior must be explicit.

## No cross-source economic analysis

Permitted:

- compare a returned funding timestamp with the existing exact-funding timestamp convention;
- compare analytics trade dates with the requested bounds;
- verify that BTC and ETH identities are correct.

Prohibited:

- joining to future prices or returns;
- checking whether OI, funding, liquidations, or basis predict anything;
- selecting a date because its values appear interesting;
- extending the probe after seeing the responses.

## Required outputs

```text
KRAKEN_ANALYTICS_DOC_SNAPSHOT_INDEX.csv
KRAKEN_ANALYTICS_MINIMAL_REQUEST_LEDGER.csv
KRAKEN_ANALYTICS_RETENTION_MATRIX.csv
KRAKEN_ANALYTICS_SCHEMA_AND_UNIT_NOTE.md
KRAKEN_ANALYTICS_REPLAY_COMPARISON.csv
KRAKEN_ANALYTICS_AUTHORITY_DECISION.md
HYPOTHESIS_DATA_READINESS_UPDATE.csv
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Raw responses remain local and hash-manifested. Do not upload response bodies to Drive.

No output may contain:

- post-request market returns;
- PnL;
- MAE/MFE;
- predictive labels;
- economic rankings;
- protected-period values.

## Decision rules

Return exactly one:

```text
ready_for_bounded_historical_analytics_audit
```

only when all four analytics types return valid historical rows for both BTC and ETH in 2023, 2024, and 2025, every upper bound is honored, and replay is sufficiently stable for a larger audit.

Return:

```text
historical_public_analytics_unavailable
```

when the endpoint is empty, unsupported, recent-only, ignores bounds, or otherwise fails to establish pre-2026 historical availability.

Return:

```text
partial_historical_analytics_requires_review
```

when only some types, symbols, or years are historically available.

No result authorizes full acquisition, signal design, or economic work.

## Validation and review

Use synthetic fixtures for:

- explicit lower and upper bounds;
- default-to-current rejection;
- 2026/mixed-payload rejection;
- valid empty responses;
- ignored-bound detection;
- schema changes;
- deterministic replay comparison;
- request and byte caps;
- zero economic fields.

Run relevant protected-boundary, source-ledger, funding-authority, manifest, continuity, and archive tests.

Require an independent review of:

- documentation snapshot;
- exact 24-request matrix;
- bounds;
- response classifications;
- replay results;
- updated capability claims;
- absence of economic and protected data.

## Integration and Drive handoff

After review:

- update only factual source, data-capability, family-readiness, and continuity records;
- create task-scoped reviewed commit(s);
- non-force push under the standing workflow;
- use `drive_handoff: approved_default`;
- round-trip verify all uploaded files;
- retain raw responses and the local archive.

## Final response

```text
status:
actual_starting_commit:
superseded_prompt_not_executed:
terminal_record_status:
documentation_snapshot:
request_matrix_completed:
request_budget_used:
per_type_symbol_year_results:
upper_bound_audit:
replay_result:
protected_rows_opened: no
economic_outputs_computed: no
data_capability_changes:
recommended_next_task:
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
human_approval_required:
```
