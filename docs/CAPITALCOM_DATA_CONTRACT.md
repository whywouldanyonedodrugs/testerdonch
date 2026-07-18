# Capital.com Data Contract

This document defines an intake boundary. It does not authorize acquisition, economic research, account access, or order activity.

## Authority

Capital.com rows are eligible for rankable loading only when a verified source manifest identifies the configured account/legal environment, platform, dataset, rankable purpose, event-time interval, schema hash, and content hash. Real payload import remains blocked until a separate verified acquisition package is approved.

The rankable interval is `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`. Capital.com data from 2026 onward default to `data_engineering_only`. Mixed-purpose or mixed-interval files fail closed.

## Row Contract

Each bar retains:

- `platform=Capital.com`, epic, instrument type/name, currency, and contract form;
- bid and ask OHLC separately;
- UTC bar start, bar end, and availability timestamps;
- calendar, metadata snapshot hash, expiry/undated state, and market status;
- explicit volume-semantic, financing, and corporate-action status.

Capital.com rows are OTC CFD quotes, not exchange trades. A hypothetical buy uses ask and a sell uses bid. Midpoint execution is not the default. Missing instrument, calendar, metadata, expiry, financing, corporate-action, or bid/ask authority fails closed. Volume remains `unverified` unless the source fixture explicitly proves its semantics.

## Execution Clock

The first-executable-target rule selects the first manifest-authorized quote whose availability is at or after the decision target and whose market status is executable. Closed periods are skipped; no synthetic fill is inserted at a closed endpoint.

## Prohibitions

The adapter contains no API authentication, account, order, or deployment client. Acquisition, private-account actions, execution assumptions, costs, financing, corporate actions, universe construction, and economic scoring require separate authorization and platform-specific contracts.
