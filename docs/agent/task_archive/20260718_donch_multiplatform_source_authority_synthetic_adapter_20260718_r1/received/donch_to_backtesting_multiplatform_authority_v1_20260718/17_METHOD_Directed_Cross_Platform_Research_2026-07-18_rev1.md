---
status: proposed new method; not applied
date: 2026-07-18
revision: 1.0
scope: directed source-to-target Kraken/Capital.com research and controls
authority: multi-platform operating contract and test/evidence manual
supersedes: none
provenance: approved policy; expansion plan; repository preflight
known limitations: no cross-platform hypothesis has been tested and Capital.com data authority is unavailable
---

# Directed Cross-Platform Research Method

## Core identity

A cross-platform hypothesis is a directed information-flow claim, not a generic correlation:

```text
translation_id
parent_hypothesis_id
source_platform
source_instrument_uid
source_observable
source_event_ts
source_available_ts
minimum_information_lag
target_platform
target_instrument_uid
target_calendar_id
target_first_executable_ts
target_entry_side_and_quote
target_exit_rule
currency_and_cashflow_contract
common_factor_controls
reverse_direction_control
session_reopen_control
timestamp_null_control
multiplicity_family
falsification_rule
```

`Capital.com -> Kraken` and `Kraken -> Capital.com` are separate translation IDs.

## Clock rules

- Use the timestamp when source information was observable, not a later bar label.
- Apply the frozen minimum lag before target eligibility.
- If the target is closed, use the first executable target bid/ask quote after reopening.
- Do not use a target bar close, high or low before it was available.
- Record timezone and DST authority for each calendar interval.
- Date-only events require a delay rule consistent with source precision.

## Instrument relationship rules

Only named research relationships enter the relationship registry. Each row needs effective dates, rationale, confidence and provenance. Do not create all-pairs mappings or infer economic equivalence from similar names.

Relationship types may include:

```text
same_underlying_different_contract
reference_or_hedge_relationship
macro_driver_candidate
sector_or_supply_chain_candidate
regulated_access_transmission
risk_factor_proxy
```

A relationship is a hypothesis input, not proof of lead-lag.

## Target execution

Kraken targets use Kraken price and funding semantics. Capital.com targets use executable bid/ask sides, calendars, financing, corporate actions and conversion. Same-underlying price differences begin as calibration and data-quality questions unless a separate economic mechanism is frozen.

## Required controls

- target own-lag/autocorrelation;
- broad BTC/ETH or relevant macro factor;
- source and target common-factor exposure;
- reverse direction;
- matched source timestamps without the event;
- target session/reopen and day-of-week;
- delayed timestamps consistent with data uncertainty;
- instrument and episode clustering;
- multiple direction/pair attempts.

## Falsification

A directed hypothesis fails when the source does not add stable information beyond target history, common factors, calendars, and matched timestamps under realistic target execution. A contemporaneous correlation or same-underlying price relation alone is insufficient.

## Claim boundary

Allowed wording states the exact direction, instruments, interval, controls and execution contract. Do not call a result portable, universal, cross-venue validated or causal beyond the frozen design.
