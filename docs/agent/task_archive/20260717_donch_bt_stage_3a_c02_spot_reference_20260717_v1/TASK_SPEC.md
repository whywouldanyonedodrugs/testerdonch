# Stage 3A — C02 Kraken Spot-Reference Data Authority

```text
task_id: donch_bt_stage_3a_c02_spot_reference_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
capture_access: no
public_data_acquisition: official Kraken rankable-period data only
```

## Objective

Establish whether C02 (`spot_led_vs_perp_led_impulse`) has a defensible official Kraken historical data path, and acquire the eligible 2023–2025 spot-reference panel when the source passes validation.

Do not generate C02 signals, inspect lead/lag performance, calculate forward returns, select thresholds, or run an economic screen.

## Current decision to preserve

```text
C01 family:
    C01_debetaed_residual_shock_path_bifurcation

C01 Level-3 decision:
    level3_no_primary_pass_stop

C01 authoritative run root:
    results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227

C01 closure commit:
    b4e6c24c7c1f6c54dea931d824383590d087b819
```

C01 must not be retuned, subset-rescued, or reopened. Confirm that the durable family/multiplicity/continuity records contain this terminal decision; update only missing factual records.

## C02 data question

Determine whether official Kraken public sources can provide point-in-time, reproducible spot trades or bars for:

```text
venue: Kraken
interval: [2023-01-01, 2026-01-01)
quote: USD only
resolution target: 5 minutes
```

The spot series must be same-asset Kraken spot references for Kraken PF contracts. Do not use another venue, synthetic index, USDT substitution, paid vendor data, or 2026 observations.

## Phase A — Source and identity preflight

Use official Kraken sources only:

- official downloadable historical market data;
- official public Spot API or archived official data endpoint;
- official asset-pair/instrument metadata;
- official documentation for timestamp, price, volume, and trade semantics.

Archive source URLs, access times, response metadata, exact files, size, and SHA-256.

Create a point-in-time mapping with:

```text
canonical_asset_id
PF_symbol
Kraken_spot_pair
spot_base
spot_quote
pair_identity_source
observed_or_verified_start
observed_or_verified_end
symbol_migrations
known_status_intervals
identity_confidence
historical_authority_status
inclusion_or_exclusion_reason
```

Use only USD-quoted Kraken spot pairs. Preserve XBT/BTC and other ticker migrations explicitly. Current pair existence may support identity, but cannot prove uninterrupted historical availability.

## Phase B — Frozen pilot

Before downloading pilot prices, freeze:

```text
PF_XBTUSD <-> XBT/USD
PF_ETHUSD <-> ETH/USD
plus the first two additional eligible crypto assets by canonical asset ID
that have both:
    a mechanism-proof-cohort PF contract
    an official Kraken USD spot pair
```

Use deterministic canonical-asset ordering, not price behavior or expected usefulness.

For each pilot pair, acquire at least:

```text
2023-01-01 through 2023-01-07
2024-06-01 through 2024-06-07
2025-12-24 through 2026-01-01 exclusive
```

Validate:

- source identity and rankable bounds before read;
- timestamps and timezone;
- exact pair and quote currency;
- duplicate/out-of-order records;
- price and volume semantics;
- gaps and partial intervals;
- deterministic 5-minute bar construction where raw trades are used;
- no row at or after 2026-01-01;
- no mixed protected payload.

Stop if the source cannot satisfy these checks without opening 2026 data.

## Phase C — Rankable acquisition

Proceed only after the pilot and independent review pass.

Acquire the full rankable interval for every pair that:

- maps to the Stage 2C mechanism-proof PF cohort;
- has an official Kraken USD spot identity;
- has defensible data coverage for at least one bounded eligible interval;
- has no unresolved ticker or unit ambiguity for that interval.

Unknown or interrupted intervals remain masked. Do not fabricate continuous coverage.

Store immutable raw identities and deterministic normalized 5-minute spot bars under repository-approved data roots. Update `data_paths()`/authority manifests only through the existing protected-boundary system.

The acquisition panel is not survivorship-free unless independently proven. Use:

```text
Kraken_USD_spot_bar_existence_panel
```

## Required outputs

```text
C02_SPOT_SOURCE_LEDGER.csv
C02_SPOT_PAIR_AUTHORITY.csv
C02_SPOT_PILOT_VALIDATION.md
C02_SPOT_COVERAGE_MATRIX.csv
C02_SPOT_DATA_MANIFEST.json
C02_SPOT_AND_PF_ALIGNMENT_CONTRACT.md
C02_DATA_CAPABILITY_DECISION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Report exact symbols, date coverage, missing intervals, sizes, hashes, and exclusions.

No output may contain:

- future returns;
- spot-versus-perp lead/lag labels;
- predictive scores;
- chosen horizons or thresholds;
- candidate ranking;
- C02 economic conclusions.

## Tests

Use synthetic fixtures for:

- spot/PF canonical identity;
- ticker migrations;
- USD-only enforcement;
- rankable-period filtering before payload read;
- trade-to-5m aggregation;
- duplicates, gaps, and boundary rows;
- deterministic manifests and hashes;
- unknown lifecycle intervals remaining unknown;
- no protected, non-Kraken, or outcome-derived rows.

Run focused tests and the relevant loader/protected-boundary guards.

## Acceptance criteria

```text
official spot source authority: verified or precisely blocked
pilot pairs frozen before price read: yes
protected rows opened: zero
economic outputs computed: zero
full acquisition performed only after pilot approval: yes
spot/PF mapping deterministic: yes
coverage and uncertainty explicit: yes
independent review: approve
```

A valid result may be `C02_historical_data_path_unavailable`. Do not weaken the source contract merely to populate the panel.

## Integration and handoff

After tests and review:

- create task-scoped commit(s) under the standing reviewed-task workflow;
- non-force push only after review;
- upload the closed archive with `drive_handoff: approved_default`;
- omit large raw/normalized market payloads from Drive and retain them locally with hashes;
- update factual data-capability and continuity records.

## Final status

Return one:

```text
ready_for_C02_non_economic_generator_contract
C02_historical_data_path_unavailable
blocked_with_exact_data_remedy
```

No economic run is authorized.

## Final response

```text
status:
actual_starting_commit:
C01_terminal_record_status:
official_sources:
pilot_pairs:
pilot_results:
spot_pair_count:
PF_mapped_pair_count:
coverage_summary:
protected_rows_opened: no
economic_outputs_computed: no
data_roots_and_manifest_hashes:
tests_and_review:
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
next_status:
human_approval_required:
```
