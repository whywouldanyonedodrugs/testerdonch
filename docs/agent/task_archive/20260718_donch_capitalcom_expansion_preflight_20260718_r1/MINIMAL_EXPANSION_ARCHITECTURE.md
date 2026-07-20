# Minimal Expansion Architecture

## Decision

Use a shared evidence/identity boundary plus two source-specific adapters. Do not build a universal simulator and do not refactor current Kraken family runners.

## What stays unchanged

- Existing Kraken data roots, manifests, readers, symbol identities, lifecycle artifacts, family definitions, decisions and run roots.
- `signal_state_contract_v1_20260715` mechanics.
- Existing event/control outcome ledgers and hashes.
- Kraken trade/mark/index/funding semantics and Kraken-only manuals where they describe Kraken behavior.

## What becomes platform-aware

Only new shared boundary records:

- dataset authority: `platform`, `source_dataset_id`, interval, purpose, protected classification, schema/content hash;
- outer instrument identity: `instrument_uid`, `platform`, native instrument ID, type/form, currency, validity and calendar;
- new run/event/control records: explicit `platform` or directed `source_platform`/`target_platform`;
- translation IDs: new Capital.com and cross-platform identities linked to preserved Kraken hypothesis IDs.

Existing Kraken hashes are not recomputed. Platform-aware outer IDs wrap them when cross-platform comparison is required.

## What remains source-specific

### Kraken adapter

Current `PF_*` discovery, trade/mark separation, perpetual funding, lifecycle and 24/7 execution remain unchanged.

### Capital.com adapter

A new adapter must require manifest-authorized:

- `platform_epic`, instrument type/name, currency and contract form;
- bid and ask OHLC separately;
- bar interval, bar open/close timestamps and availability timestamp;
- platform calendar, expiry/status and metadata snapshot hash;
- volume with semantic status (`verified`, `unverified`, or unavailable);
- spread from bid/ask only;
- financing/corporate-action/trading-rule status as explicit known/unknown fields.

It must never label bid/ask bars as exchange trades, use midpoint as a fill without an explicit diagnostic mode, or infer historical rules from current metadata.

## Smallest identity and mapping layer

Two additive tables are sufficient:

1. `instrument_identity.csv`: `instrument_uid`, `platform`, `platform_instrument_id`, `asset_class`, `contract_form`, `underlying_uid`, `quote_currency`, `expiry_or_undated`, `valid_from_utc`, `valid_to_utc`, `calendar_id`, `metadata_hash`.
2. `directed_instrument_relationships.csv`: source UID, target UID, relationship type/rationale, effective interval, confidence and provenance.

Only researched relationships are entered. No all-pairs catalog is justified.

## Physical data layout

Eventually use a source-separated external root, not the repository:

```text
/opt/parquet/capitalcom/
  raw/
  normalized/bid_ask_bars/
  manifests/
  reference/instruments/
  reference/calendars/
  qc/
```

Kraken remains under `/opt/parquet/kraken_derivatives` and `/opt/parquet/kraken_spot_reference`.

## Directed cross-platform contract

Each test must freeze:

- source/target platform and instrument UID;
- source event time and source availability time;
- minimum information lag;
- target calendar and first executable target quote;
- target bid/ask execution side and costs;
- holding/exit semantics;
- currency conversion and corporate-action treatment;
- common-factor, reverse-direction, session/reopen and timestamp-null controls;
- multiplicity family and falsification rule.

`A -> B` and `B -> A` are distinct contracts.

## Protected rows fail before load

The manifest authority validator must reject before a payload reader is called when:

- purpose is unknown, mixed, protected, calibration-only or external-unrankable;
- file interval reaches 2026+ unless the end is proven exclusive at the cutoff;
- platform is missing or conflicts with the selected adapter;
- content/schema hash or rankable classification is missing;
- a Capital.com file mixes rankable and protected purposes.

After a rankable file is opened, row guards must remove pre-2023 and wrong-platform rows before feature selection or scoring, then reassert `< 2026-01-01`. Funding/financing receives equivalent treatment.

## Bounded later patch scope

First code patch only:

- new `tools/qlmg_rankable_source_contract.py`;
- compatibility call from `tools/run_kraken_family_engine_aggregate_first_sweep.py` with no changed Kraken output;
- new `tools/capitalcom_data_adapter.py` using synthetic fixtures only;
- new `unit_tests/test_rankable_source_contract.py`;
- new `unit_tests/test_capitalcom_data_adapter.py`;
- repository contract documentation explicitly approved in a later task.

Stop before instrument-map population or real data import.
