# Stage 2A1 — Final 2025 Day and C01 Reference-Panel Authority

```text
task_id: donch_bt_stage_2a1_c01_reference_panel_20260717_v1
target_environment: backtesting Codex
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
```

## Objective

Produce a non-economic, authority-bounded C01 reference panel by:

1. acquiring the missing final rankable 2025 trade and mark candles for `PF_XBTUSD` and `PF_ETHUSD` without opening any 2026 row;
2. adding official Kraken terminal-lifecycle evidence for those two contracts; and
3. replacing the impossible “uninterrupted tradeability throughout 2023–2025” claim with a precise split between:
   - structural contract identity/lifecycle; and
   - event-time operational availability.

Do not implement C01 features, candidates, controls, returns, or economic evaluation.

## Controlling decision

Stage 2A is accepted as correctly fail-closed:

```text
task: donch_bt_stage_2a_u2_lifecycle_20260716_v1
commit: 2a15da802d630ce6d11c3ac8a6a0fcfba78aff5f
result: completed_fail_closed_no_defensible_u2_cohort
```

The unchanged U2 standard requiring proof of uninterrupted historical tradeability remains unavailable. Do not claim that sparse snapshots or current status prove this.

For C01, use a different bounded object:

```text
reference_panel_id: kraken_c01_reference_panel_v1
members: PF_XBTUSD, PF_ETHUSD
role: causal market-reference/factor series
continuous_tradeability_claim: no
survivorship_free_claim: no
```

These instruments are reference factors, not the future C01 candidate universe.

## Authority interpretation

### Structural lifecycle

Use official Kraken evidence to establish only:

- stable instrument identity;
- official opening date;
- absence or presence in Kraken's cumulative derivatives delisting/settlement record;
- known symbol migration, terminal settlement, or resumed-trading note.

The official Kraken derivatives delistings page may be used as a retrospective terminal-event ledger. Cache the exact page, access time, response metadata, and SHA-256. Use archived official copies where available and useful.

Do not infer that absence from the delisting table proves there was never a temporary post-only, maintenance, or trading interruption.

### Operational availability

Operational tradeability is evaluated at the event interval, not assumed continuously for the entire three-year period.

A later C01 event may use a reference series only when:

```text
trade bars required by the feature window are present
mark bars required by the feature/risk window are present
timestamps are inside [2023-01-01, 2026-01-01)
no known contract-specific terminal state applies
no known maintenance/status interval invalidates the required window
all required rows pass current loader authority
```

Missing or invalid windows fail closed at event time. Temporary platform outages do not remove the contract from the reference panel for all years.

Do not build a complete status-history or outage platform in this task.

## Final-day acquisition

Use the official public Kraken Futures Market Candles endpoint:

```text
GET https://futures.kraken.com/api/charts/v1/{tick_type}/{symbol}/5m
```

Acquire exactly these four slices:

```text
symbol: PF_XBTUSD, tick_type: trade
symbol: PF_XBTUSD, tick_type: mark
symbol: PF_ETHUSD, tick_type: trade
symbol: PF_ETHUSD, tick_type: mark
```

Date bounds:

```text
from: 1767139200     # 2025-12-31T00:00:00Z
to:   1767225599     # 2025-12-31T23:59:59Z
protected_start_ms: 1767225600000
```

Never request an upper bound at or after 2026-01-01.

Fail closed if:

- any returned candle timestamp is `>= 1767225600000`;
- the response includes another symbol or tick type;
- pagination or `more_candles` cannot be completed without changing the upper bound;
- response identity, schema, or time semantics are ambiguous;
- the source returns a mixed 2025/2026 object.

Preserve raw response bytes, headers, URL, access time, status, and SHA-256. Normalize under the existing Kraken trade/mark schema and authority metadata.

Do not open the previously identified mixed protected chunk.

## Scope

Allowed:

- inspect current repository and Stage 2A archive;
- public read-only Kraken Futures candle requests for the four exact slices;
- public read-only Kraken derivatives delisting/support page and archived official copies;
- minimal parser/normalizer and tests if existing code cannot ingest the bounded slices;
- factual updates to lifecycle/data-capability authority records;
- task archive and approved default Drive handoff.

Forbidden:

- other symbols or dates;
- 2026 data;
- public-order-event bulk acquisition;
- complete historical status reconstruction;
- candidate-universe construction;
- C01 residual/path feature implementation;
- candidate counts or returns;
- package-protocol repair;
- C02/C03 work;
- capture;
- private endpoints;
- paid data;
- governance or documentation restructuring beyond the exact factual authority artifact.

## Required outputs

Use repository conventions and retain:

```text
C01_REFERENCE_PANEL_AUTHORITY.csv
C01_REFERENCE_PANEL_CLAIM_BOUNDARY.md
FINAL_2025_DAY_SOURCE_LEDGER.csv
FINAL_2025_DAY_COVERAGE_VALIDATION.md
TERMINAL_LIFECYCLE_SOURCE_LEDGER.csv
TERMINAL_LIFECYCLE_VALIDATION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Required reference-panel fields:

```text
reference_panel_id
instrument_id
canonical_asset_id
Kraken_symbol
role
official_opening_date
identity_confidence
terminal_lifecycle_status
terminal_lifecycle_source
terminal_lifecycle_source_sha256
trade_coverage_start_utc
trade_coverage_end_utc
mark_coverage_start_utc
mark_coverage_end_utc
continuous_tradeability_claim
operational_availability_rule
survivorship_cap
included
inclusion_reason
permitted_claim
prohibited_claim
```

## Tests

Use synthetic fixtures for all boundary logic.

Verify:

1. exact `from` and `to` request bounds;
2. any 2026 timestamp causes rejection before normalization;
3. wrong symbol/tick type causes rejection;
4. raw response and normalized output hashes are stable;
5. duplicate timestamps and out-of-order rows fail;
6. 5-minute interval coverage and any genuine missing intervals are reported, not silently filled;
7. trade and mark remain separate;
8. terminal lifecycle parser identifies included, delisted, and resumed examples;
9. absence from the terminal table is never converted into a no-outage claim;
10. no candidate, return, funding, or protected reader is called.

Run focused tests plus the smallest relevant repository-supported regression/guard suite.

## Acceptance criteria

```text
four exact final-day slices acquired or explicitly blocked
returned 2026+ rows: zero
protected payloads opened: zero
trade/mark coverage through 2026-01-01 exclusive: verified or explicitly incomplete
terminal lifecycle status for PF_XBTUSD/PF_ETHUSD: bounded and sourced
continuous tradeability claimed: no
reference panel role: factor/reference only
economic outputs computed: zero
independent review: approve
```

A blocked final-day endpoint response is an acceptable task result. Do not use a mixed protected file as fallback.

## Integration and handoff

After tests and review pass:

- create one task-scoped commit;
- integrate and non-force push only under the repository's standing reviewed-task workflow;
- package the closed task archive;
- upload using `drive_handoff: approved_default`;
- verify remote content according to `docs/agent/DRIVE_HANDOFF_TARGET.md`;
- keep the local package.

## Next task

If the reference panel and final-day coverage pass, recommend:

```text
Stage_2B_C01_identity_multiplicity_and_feature_foundation
```

That later task must define the capped candidate cohort separately and must use the authority label:

```text
current_roster_bar_existence_cohort
```

unless stronger PIT lifecycle evidence is independently acquired.

Do not implement Stage 2B here.

## Final response

```text
status:
actual_starting_commit:
final_day_requests:
final_day_rows_by_symbol_and_tick_type:
returned_2026_rows:
trade_coverage_end:
mark_coverage_end:
terminal_lifecycle_sources:
reference_panel_id_and_hash:
continuous_tradeability_claim: no
survivorship_free_claim: no
files_changed:
tests_and_results:
protected_outcomes_opened: no
economic_outputs_computed: no
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
artifacts_and_sha256:
remaining_limits:
next_recommended_task:
human_approval_required:
```
