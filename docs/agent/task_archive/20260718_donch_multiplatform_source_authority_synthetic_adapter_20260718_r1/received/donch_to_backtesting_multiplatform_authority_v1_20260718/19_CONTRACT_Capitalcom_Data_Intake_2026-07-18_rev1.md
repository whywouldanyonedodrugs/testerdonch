---
status: proposed new data-intake contract; not applied
date: 2026-07-18
revision: 1.0
scope: credential-free Capital.com acquisition handoff and metadata validation before import
authority: multi-platform operating contract; official Capital.com API/pricing documentation; repository preflight
supersedes: none
provenance: approved policy; expansion plan; official Capital.com documentation accessed 2026-07-18
known limitations: actual downloader implementation, account/legal environment, files and coverage are not inspected here
---

# Capital.com Data Intake Contract

## Purpose

This contract governs the closed handoff from the Capital.com acquisition host to Donch/backtesting. It does not authorize trading, account mutation, economic analysis or protected-outcome use.

## Required package

```text
READ_FIRST.md
ACQUISITION_MANIFEST.csv or parquet
INSTRUMENT_MANIFEST.csv
SCHEMA_DICTIONARY.md or json
COVERAGE_AND_GAPS.csv
CALENDAR_AND_TIMEZONE_AUTHORITY.csv
METADATA_SNAPSHOT_MANIFEST.csv
FINANCING_AND_CASHFLOW_STATUS.csv
CORPORATE_ACTION_STATUS.csv
PROTECTED_PERIOD_AUDIT.json
COMMANDS_AND_RESULTS.md
SECRET_SCAN.md
TRANSFER_MANIFEST.json
```

## Per-file manifest fields

```text
relative_path
byte_size
sha256
platform
account_or_legal_environment_label
native_epic
instrument_type
resolution
schema_version
schema_hash
minimum_snapshot_time_utc
maximum_snapshot_time_utc
interval_end_semantics
purpose
protected_classification
row_count
known_gap_count
source_endpoint_or_method
acquisition_run_id
```

Mixed rankable/protected files are rejected unless they can be physically repartitioned without viewing strategy outcomes and the new files receive independent manifests.

## Instrument fields

At minimum: epic, symbol/name, instrument type, currency, expiry/undated form, current status, opening-hours zone, metadata snapshot time/hash, and the configured account/legal-environment label. Historical validity remains unknown unless separately sourced.

## Price fields

Bid and ask OHLC remain separate. Record `snapshotTimeUTC` or equivalent availability fields, resolution and pagination method. The API documents a maximum of 1,000 historical values per response, so coverage and pagination reconciliation are mandatory. Volume is `unverified_semantics` until a primary source or audit establishes its meaning.

## Economic-semantics status

For each instrument/class, label financing, premium adjustments, dividends/corporate actions, currency conversion, expiry/roll, margin/dealing rules and historical rule versions as:

```text
verified_historical
verified_current_only
partially_verified
unavailable
not_applicable
```

Do not infer historical costs from current market details. Missing fields block or cap later holding periods and claims.

## Security

No API key, custom password, account identifier, session token, OAuth material, cookies, raw configuration secret, private endpoint response or order/account-mutation payload enters the package. Capital.com documents that API keys are trading-capable and no read-only key privilege is available; research code must therefore remain separated from credentialed acquisition.

## Acceptance

The receiving agent verifies package and per-file hashes, schemas, intervals, protected purposes, duplicates, coverage and secret scan before reading market payload. Failed verification leaves import disabled and preserves the package for repair.

## Official source trace

- Capital.com Public API: `https://open-api.capital.com/`
- Capital.com EU charges and fees: `https://capital.com/en-eu/ways-to-trade/fees-and-charges`
- Capital.com market pricing: `https://capital.com/en-eu/about-us/how-our-markets-are-priced`
- Dividend adjustments: `https://help.capital.com/hc/en-us/articles/17048526103314-Dividend-adjustments`

These sources document current capabilities and mechanics. They do not establish the acquired historical dataset.
