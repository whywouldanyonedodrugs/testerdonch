# Stage 19 — Ingest the Local Official Kraken Funding Export, Regenerate the Campaign Packet, and Audit the Wider Historical Data Surface

```text
task_id: donch_bt_stage_19_local_official_funding_export_packet_20260720_v2
repository: /opt/testerdonch
mode: direct_apply_in_isolated_worktree
drive_handoff: approved_default

local_official_funding_export_read_authorized: yes
protected_funding_partition_engineering_authorized: yes — exact bounded purpose below
official_public_metadata_web_access_authorized: yes
other_bulk_market_data_download_authorized: no
new_strategy_outcome_access: no
phase_2_or_later_campaign_economics_authorized: no
protected_strategy_price_or_return_access: no
Capitalcom_payload_access: no
Telegram_real_message_authorized: no
orders_or_live_trading: no

funding_contract_replacement_authorized: yes
campaign_packet_regeneration_authorized: yes
historical_terminal_decision_change_authorized: no
commit_authorized: yes
push_authorized: yes — non-force
```

## Objective

Use the human-transferred official Kraken funding export already present on the host:

```text
/opt/testerdonch/research_inputs/exports.zip
```

Do not download another copy unless this exact file is missing or fails integrity checks. If the file is absent, stop with the exact path required.

The human states that `exports.zip` was downloaded through Kraken's official **Export historical funding rates** page and uploaded to the repository's `research_inputs` directory.

The task must:

1. verify, inventory and physically partition the export;
2. establish complete funding economics without guessing a single row timestamp interpretation;
3. regenerate every funding-dependent campaign object and a non-authorizing approval packet;
4. run a synthetic Phase 2–5 canary and independent review;
5. perform a separate metadata-only audit of other official Kraken historical-data surfaces that could improve future research.

Do not launch campaign economics.

## Sample anchors supplied to Donch

Three files independently inspected from the same export have this exact schema:

```text
timestamp,tradeable,absolute_rate,relative_rate
```

Sample hashes:

```text
PF_DEXEUSD.csv
    SHA-256 297f5badb99207349f71307e0496cf35463454f69a107fe429d8cf402c8ad0f8
    rows 8,664
    2025-02-04 18:00:00 through 2026-02-01 00:00:00

PF_OPENUSD.csv
    SHA-256 6fc1d276a9f4207a28be9f2504192360785abb6a3d9bd3ee6d91a8b4b0671c31
    rows 2,743
    2025-10-09 13:00:00 through 2026-02-01 00:00:00

PF_ZBTUSD.csv
    SHA-256 c2877d184c5a5e7e185b7d6d76601fb5d31c436b096c6c8f2da69386cd052315
    rows 2,073
    2025-11-06 14:00:00 through 2026-02-01 00:00:00
```

Observed properties in all three samples:

- no nulls;
- one row per UTC hour apart from a few missing hours;
- no duplicate timestamps;
- signed `absolute_rate` and `relative_rate` always agree;
- for nonzero rows, `absolute_rate / relative_rate` is positive and price-like;
- each file contains 745 rows at or after `2026-01-01T00:00:00Z`.

Use these only as integrity anchors. The full ZIP controls final coverage.

## Current authority to verify

```text
expected origin/main and task start:
    3f5e94eb8e6b8becb4dfaa5457742682ac31f7e9

Stage-18 terminal status:
    blocked_no_safe_rankable_absolute_funding_row_groups

Stage-18 artifact manifest:
    79e779663d3b86a1df07fc9146f1a2f2f631ed57dc9d64e3f6d1f785cd3bfdcd

Stage-18 blocked diagnostic-root manifest:
    8b890636fdf3799a57be576f5d557874883e4bb11a155f7ab080bcecc8534609

Stage-16 economic translation registry:
    9c07f8695b117afe61f03354e0b6ab39a4c11bd4f0335f9f066cca33560ff1f8

Stage-16 campaign manifest:
    cc07499c671cf39b8ceaee91156f141dcc2c5532142af29a38a4f6830b73f23d

Stage-16 approval packet:
    c01281e50f40f95b922a04ed01c5b3d28ed325577891eed8e2ca5d32286965ca

Stage-16 search/bin specification:
    1cc994543371a8db86f428957a27d3a62d93f390e30f99460aa3e1d973b8c67b

Stage-16 estimator/rule inventory:
    b30b7c115d6d1ed765542c44d791c44117a0387f71ae4b33ec1219d4243a3b

Stage-16 inner-fold map:
    6ee8cbc53eac9326f904eec760ae82861a533b8165be6e1ccfc440dd0fa32ba0

Stage-16 utility/Pareto contract:
    8b9d6afa6c1ea6676cdac9b1703223ef1d47d929988c8e419fc1c112e4a3f093

Stage-16 boundary contract:
    460eb30f45c232c79729765b8724ddd206ed09671bb98f485b2175bf83cc5e75

analytics manifest:
    f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d

authorized cohort:
    5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636
```

Verify repository root, `AGENTS.md`, Git/remotes/worktrees, clean status, local ZIP path, archive convention, supported commands, data roots and Drive target.

Preserve all Stage 17/18 incidents and negative findings unchanged.

# Part A — Verify and partition the local export

## 1. Local ZIP authority

Record:

```text
local path
byte size
SHA-256
filesystem timestamps
ZIP member count
compressed and uncompressed bytes
```

Fetch the official Kraken support page metadata only:

```text
https://support.kraken.com/articles/export-historical-funding-rates
```

Record page retrieval time, page hash and the current official export-link host. Do not replace the local ZIP merely because the current upstream file has changed.

Authority classification:

```text
human_transferred_official_export
+ official_support_page_provenance
+ local_content_hash
```

Do not claim independent remote-byte identity unless a published content hash or exact remote-byte comparison exists.

## 2. ZIP security

Before extraction reject:

- absolute paths;
- `..` traversal;
- symlinks;
- duplicate member names;
- encrypted members;
- unreasonable compression ratios;
- non-CSV unexpected payloads unless separately explained.

Inventory every member and hash its uncompressed bytes.

Verify the three supplied sample-member hashes when those names exist. A mismatch is a blocker requiring explanation.

## 3. Full schema and coverage audit

For every CSV verify:

```text
header exactly:
    timestamp,tradeable,absolute_rate,relative_rate

filename-symbol equality
tradeable-column constancy
UTC-hour timestamp parsing
strict timestamp ordering
duplicates
gaps
finite decimal fields
absolute/relative sign consistency
zero/zero consistency
first and last timestamp
rankable and protected row counts
```

Use decimal-safe parsing for rates. Do not downcast source authority to binary floating point.

Produce the exact ZIP symbol inventory and compare it with:

```text
479 reconciled identities
460 K0 included identities
187 authorized campaign symbols
```

One explicit mapping or exclusion reason per campaign symbol is required.

## 4. Bounded protected funding partition

This task explicitly authorizes opening the official export's `2026+` **funding CSV rows only** for physical partition engineering.

It does not authorize:

- protected funding statistics;
- protected funding calibration;
- joins to protected prices or strategy identities;
- protected return analysis;
- protected feature or threshold selection.

Implement raw-line streaming where possible:

1. parse header;
2. parse timestamp and tradeable identity;
3. route the raw row physically;
4. convert rate fields numerically only for rankable rows.

Report honestly:

```text
protected_funding_rows_opened_for_partition: yes
protected_funding_values_used_for_statistics: 0
protected_strategy_price_or_return_rows_opened: 0
```

Partitions:

```text
rankable_2023_2025:
    timestamp < 2026-01-01T00:00:00Z

protected_2026_plus_quarantine:
    timestamp >= 2026-01-01T00:00:00Z

unknown_or_invalid:
    null, malformed, ambiguous or unparseable timestamp
```

Create immutable physical packages. Future campaign code may access only the rankable package.

# Part B — Freeze exact funding under timestamp ambiguity

## 5. Official funding semantics

Bind the official linear-perpetual definitions:

```text
absolute_rate:
    USD funding received by a one-contract-unit short position for one hour

absolute payout:
    contract quantity * absolute_rate * time fraction held

positive rate:
    longs pay shorts

negative rate:
    shorts pay longs

funding:
    continuous, settled hourly or when net position changes
```

Treat export timestamps as UTC hourly boundaries.

The unresolved question is only whether row timestamp `t` applies to:

```text
alignment_start:
    [t, t + 1 hour)

alignment_end:
    [t - 1 hour, t)
```

Do not choose one by inference.

## 6. Exact cashflow under both alignments

For every trade and each alignment:

```text
position_sign:
    +1 long
    -1 short

held_fraction:
    overlap_seconds(trade_interval, funding_period) / 3600

signed_funding_cashflow_bps:
    -position_sign
    * absolute_rate_usd_per_contract_unit_per_hour
    * held_fraction
    / entry_trade_open_usd_per_contract_unit
    * 10000
```

Sum across all overlapping periods.

Verify that the PF market's quoted price and `absolute_rate` use the same contract/base unit. Use:

- official Kraken instrument/base-unit metadata;
- export `absolute_rate / relative_rate` implied price where nonzero;
- contemporaneous rankable trade/mark prices as an outcome-free unit sanity check.

Do not infer a hidden multiplier silently. A symbol with unresolved units is mechanically excluded before outcomes.

## 7. Candidate-selection funding rule

Funding must never rescue a candidate.

For each trade compute:

```text
signed_start_alignment_bps
signed_end_alignment_bps

adverse_exact_funding_bps:
    min(0, signed_start_alignment_bps, signed_end_alignment_bps)
```

Primary funding-adjusted net:

```text
gross_return_bps
- 14
+ adverse_exact_funding_bps
+ missing-hour base fallback cost
```

Stress net:

```text
gross_return_bps
- 32
+ adverse_exact_funding_bps
+ missing-hour stress fallback cost
```

The exact signed start/end alignment results remain diagnostics. Favorable funding credits cannot activate, rank, rescue or promote a candidate.

This avoids selecting a timestamp interpretation and avoids granting favourable funding alpha.

## 8. Missing-hour fallback

For expected hourly periods lacking an export row, calibrate from rankable rows only:

```text
base gap allowance:
    symbol q95 of abs(relative_rate) * 10000 bps/hour

stress gap allowance:
    symbol q99 of abs(relative_rate) * 10000 bps/hour

quantile:
    Hyndman-Fan type 7 / linear interpolation

minimum symbol observations:
    720

fallback:
    equal-symbol-weighted pooled distribution among unit-compatible PF symbols
```

Gap cost:

```text
-base_or_stress_allowance_bps_per_hour * missing_overlap_hours
```

Requirements:

- always nonpositive;
- same for long and short;
- never a credit;
- no protected rows;
- no candidate-, fold-, context- or return-conditioned calibration;
- complete campaign-symbol coverage or event rejection.

This fallback applies only to missing export hours, not uniformly to all observed hours.

## 9. Reporting partitions

Report:

```text
pre_funding_14bps
adverse_exact_dual_alignment_primary
adverse_exact_dual_alignment_stress
signed_start_alignment_diagnostic
signed_end_alignment_diagnostic
timestamp_alignment_sensitivity
missing_hour_allowance_contribution
```

The row timestamp ambiguity is not hidden. A candidate positive under only one alignment is tagged `funding_timestamp_alignment_sensitive`; it is not promoted as robust funding-adjusted evidence.

# Part C — Regenerate campaign authority

## 10. Funding-dependent objects

Regenerate, rather than edit in place:

```text
FUNDING_SOURCE_AND_PARTITION_MANIFEST.json
FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json
FUNDING_GAP_ALLOWANCE_TABLE.csv
FUNDING_COST_AND_COVERAGE_CONTRACT.json
BOUNDARY_AND_MISSINGNESS_CONTRACT.json
DEVELOPMENT_METRIC_CONTRACT.json
UTILITY_AND_PARETO_CONTRACT.json
ECONOMIC_TRANSLATION_REGISTRY.json
CAMPAIGN_MANIFEST.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md
RESOURCE_PROJECTION.json
CAMPAIGN_PACKET_LAUNCH_READINESS.md
```

All funding-dependent economic-address hashes must change.

Preserve unless a pure dependency hash changes:

- 186 executable cells;
- 42 inherited non-executable KDX attempts;
- search bins and rule grammar;
- inner/outer folds;
- instruments and sides;
- entries and exits;
- candidate beam and Pareto ordering;
- Telegram/supervision contract;
- route policy and historical terminal decisions.

The replacement packet remains non-authorizing and requires a new exact human approval.

## 11. Coverage gates

Before packet publication require:

```text
187/187 campaign symbols mapped or explicitly mechanically excluded
rankable funding export physically separate
zero protected rows in rankable package
all observed trade funding periods accounted for by export row or gap fallback
unit compatibility verified
dual-alignment arithmetic deterministic
```

Exact funding availability by symbol/year/fold is reported but does not itself select strategies.

## 12. Synthetic canary and review

Run a synthetic Phase 2–5 canary proving:

- both timestamp alignments;
- partial-hour accrual;
- long/short and positive/negative-rate arithmetic;
- favorable funding ignored for selection;
- missing-hour q95/q99 fallback;
- complete economic-address stability;
- 186-cell campaign determinism;
- no runtime semantic discretion.

Independent reviewer must explicitly decide:

```text
local export authority accepted/rejected
physical partition accepted/rejected
absolute-rate units accepted/rejected
dual-alignment contract accepted/rejected
gap allowance accepted/rejected
campaign-symbol coverage accepted/rejected
packet launch-complete after new approval
```

No residual semantic decision may be deferred to campaign runtime.

# Part D — Wider Kraken historical-data surface audit

## 13. Scope

After the funding packet work is complete or conclusively blocked, perform a metadata-only official-source audit. Do not bulk-download additional datasets.

Use official Kraken support and API documentation only.

Audit at minimum:

```text
derivatives funding bulk export
derivatives trade/mark/spot candles
market analytics:
    open-interest
    aggressor-differential
    trade-volume
    trade-count
    liquidation-volume
    rolling-volatility
    long-short-ratio
    long-short-info
    cvd
    top-traders
    orderbook
    spreads
    liquidity
    slippage
    future-basis
    funding
public order-event history
recent futures/index trade-history endpoint
spot bulk time-and-sales ZIPs
spot bulk OHLCVT ZIPs
instrument and lifecycle archives/changelogs
```

For each record:

```text
data_type
official page or endpoint
bulk export available
platform: derivatives / spot / account
public or authenticated
historical query controls
documented retention
available intervals
schema
PIT and lifecycle limitations
current Donch acquisition status
whether Stage 7C already covers it
likely incremental value
recommended next action
```

Important distinctions:

- Do not treat Kraken's spot bulk time-and-sales or OHLCVT downloads as PF derivatives data unless member inventory proves PF coverage.
- They may still be valuable as a spot/reference panel.
- The recent `/derivatives/api/v3/history` endpoint documents only about seven days or the latest engine restart and is not a historical backfill source.
- The Market Analytics endpoint advertises many additional analytics, but documentation of an endpoint is not proof of full historical retention or complete local acquisition.
- No current support-page evidence should be treated as proof that bulk ZIP exports exist for OI, basis, liquidation or the other analytics.

Outputs:

```text
KRAKEN_OFFICIAL_HISTORICAL_DATA_SURFACE_AUDIT.md
KRAKEN_OFFICIAL_HISTORICAL_DATA_SURFACE_REGISTRY.csv
KRAKEN_NEXT_DATA_ACQUISITION_PRIORITIES.md
```

Rank priorities using data authority, mechanism value, overlap with existing Stage 7C data, expected historical coverage, cost and storage—not strategy outcomes.

No additional acquisition is authorized in this task.

# Part E — Durable records, Git and handoff

Update surgically:

- funding capability/source registry;
- protected-purpose/partition record;
- continuity and current decisions;
- Kraken analytics/funding learnings;
- campaign packet lineage;
- data-acquisition candidate registry.

Preserve all prior terminal decisions and incident records.

Create separate reviewed commits for:

1. local export ingestion and physical partition;
2. funding engine and packet regeneration;
3. wider official-data audit and documentation;
4. archive/manifest closure;
5. Drive verification.

Non-force push only. Finish with clean main and worktree.

Complete `drive_handoff: approved_default`, round-trip verify compact files, and retain the large local ZIP and physical partitions locally with hash manifests.

# Required outputs

```text
LOCAL_OFFICIAL_FUNDING_EXPORT_AUTHORITY.md
OFFICIAL_EXPORT_SOURCE_MANIFEST.json
OFFICIAL_EXPORT_MEMBER_INVENTORY.csv
OFFICIAL_EXPORT_SCHEMA.json
OFFICIAL_EXPORT_COVERAGE.csv
OFFICIAL_EXPORT_PARTITION_LEDGER.csv
RANKABLE_FUNDING_PACKAGE_MANIFEST.json
PROTECTED_FUNDING_PARTITION_AUDIT.json
FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json
FUNDING_GAP_ALLOWANCE_TABLE.csv
FUNDING_COST_AND_COVERAGE_CONTRACT.json
FUNDING_CALIBRATION_AND_ALIGNMENT_SUMMARY.md
BOUNDARY_AND_MISSINGNESS_CONTRACT.json
DEVELOPMENT_METRIC_CONTRACT.json
UTILITY_AND_PARETO_CONTRACT.json
ECONOMIC_TRANSLATION_REGISTRY.json
CAMPAIGN_MANIFEST.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json
FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md
RESOURCE_PROJECTION.json
CAMPAIGN_PACKET_LAUNCH_READINESS.md
SYNTHETIC_CANARY.json
KRAKEN_OFFICIAL_HISTORICAL_DATA_SURFACE_AUDIT.md
KRAKEN_OFFICIAL_HISTORICAL_DATA_SURFACE_REGISTRY.csv
KRAKEN_NEXT_DATA_ACQUISITION_PRIORITIES.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

Large ZIP, rankable funding package and protected quarantine remain local and hash-manifested.

# Stop conditions

Stop globally for:

- missing or ambiguous local ZIP path;
- ZIP/member integrity failure;
- sample-anchor mismatch without explanation;
- export schema lacks verified absolute and relative rate fields;
- protected funding used for statistics;
- protected strategy-price/return access;
- unresolved PF contract/base-unit mapping;
- inability to cover a campaign symbol;
- funding or campaign nondeterminism;
- packet requires a new runtime semantic choice.

Do not fall back to the old mixed Parquet source, zero funding, mark/spot substitution or the retired relative-rate imputation model.

Do not launch the campaign.

# Final response

```text
status:
actual_starting_commit:
local_export_path_size_sha256:
official_page_provenance:
ZIP_members_and_symbols:
sample_anchor_verification:
protected_funding_rows_opened_for_partition:
protected_funding_values_used_for_statistics: 0
protected_strategy_price_or_return_rows_opened: 0
rankable_funding_package_hash:
rankable_rows_date_range_and_symbol_coverage:
campaign_symbol_mapping:
absolute_rate_unit_verification:
dual_alignment_contract_hash:
gap_allowance_method_and_hash:
funding_contract_hash:
economic_translation_registry_hash:
replacement_campaign_manifest_hash:
replacement_approval_packet_hash:
cell_count_and_resource_projection:
synthetic_canary:
launch_readiness_review:
wider_data_surface_audit:
recommended_next_acquisitions:
economic_outputs_computed: no
Capitalcom_payload_opened: no
Telegram_messages_sent: no
tests_and_reviews:
files_and_commits:
origin_main_updated:
original_checkout_final_status:
task_worktree_final_status:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
human_approval_required:
```
