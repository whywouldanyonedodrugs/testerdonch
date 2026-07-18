---
status: proposed current operating authority after application
date: 2026-07-18
revision: 2.0
scope: Donch planning, Kraken and Capital.com historical research, directed cross-platform research, and protected-data governance
authority: exact human approval of 2026-07-18 plus higher machine-enforced contracts and finalized evidence
supersedes: 01_AUTHORITY_QLMG_Operating_Contract_2026-07-16.md after verified application
provenance: 2026-07-18 approved policy; repository preflight; 2026-07-16 contract and source audits
known limitations: Capital.com acquisition manifests, historical costs, corporate actions, lifecycle and trading-rule histories remain unavailable
---

# Multi-Platform Operating Contract

## Binding program contract

```text
research_platforms:
  - Kraken derivatives
  - Capital.com instruments present in a verified acquisition manifest for the configured account and legal environment
rankable_interval: [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
protected_period_start: 2026-01-01T00:00:00Z
kraken_july_2026_purpose: execution_calibration_only
capitalcom_2026_plus_default_purpose: data_engineering_only
paid_historical_vendor_data: prohibited
live_trading_orders_private_account_actions: not_authorized
existing_kraken_lineage: preserve_unchanged
cross_platform_contract: directed_source_to_target_required
```

“All Capital.com instruments” is acquisition and inventory scope. It is not a research universe, an economic search authorization, or evidence that every instrument has adequate history or reconstructable execution economics.

## Platform contract

### Kraken

Kraken derivatives retain their existing trade/last, mark, index, perpetual funding, lifecycle, 24/7 calendar, execution, and capture semantics. Existing Kraken source identities, family decisions, hashes, roots, and negative findings remain unchanged.

### Capital.com

Capital.com is a separate CFD source and target. Its adapter must preserve bid and ask, instrument epic, contract form, calendar, expiry/status, financing, dividend/corporate-action adjustments, currency conversion, and account/legal-environment metadata. A current metadata value does not establish its historical value. Mid prices are diagnostic only unless an explicitly approved contract uses them for a non-fill calculation.

### Directed cross-platform research

Every cross-platform hypothesis declares source platform, source instrument, source availability time, target platform, target instrument, minimum lag, first executable target quote, target execution semantics, controls, falsification, and multiplicity family. `A -> B` and `B -> A` are distinct identities and require separate frozen economic approvals.

## Research and protected periods

Pre-2023 material may support priors and engineering context. Rankable work is confined to the common interval. All 2026+ data are protected and purpose-labelled before load.

Allowed 2026 purposes:

```text
protected_holdout
execution_calibration_only
prospective_shadow
data_engineering_only
```

Kraken July 2026 defaults to `execution_calibration_only`. Capital.com 2026+ defaults to `data_engineering_only`. Neither may be used for directional discovery, threshold selection, feature selection from outcomes, family ranking, or reopening a historical translation.

## Evidence contract

Every claim states:

- source platform and, when applicable, target platform;
- instrument and contract form;
- hypothesis and translation lineage;
- exact interval and protected-row count;
- source manifests, schemas, hashes, and point-in-time eligibility;
- price fields and source-specific cashflows;
- event, control, and episode identities;
- costs and execution assumptions;
- evidence, reproducibility, validation, and deployment states;
- multiplicity, concentration, limitations, and permitted next action.

No evidence transfers automatically between platforms. A Capital.com result may inform a mechanism but does not overwrite a Kraken result or prove Kraken execution.

## Environments

- Donch: authority, synthesis, task construction, continuity, and review.
- Backtesting repository: authorized source-contract engineering, platform adapters, historical tests, and evidence artifacts.
- Kraken capture repository: public Kraken capture and execution calibration only.
- Capital.com acquisition host: approved public/account-authenticated acquisition and data packaging only; no trading actions.
- Drive: archive and handoff, not authority.
- Human: trading judgment, broad direction, conflict resolution, and exact consequential approvals.

## Approval gates

Separate explicit approval remains required before:

- any economic screen or backtest;
- any protected-outcome access or protected-purpose change;
- real Capital.com data import into decision-bearing research;
- capture restart or network-write production test;
- private-account mutation, demo or real order, or risk change;
- destructive archive action;
- repository commit unless a task explicitly authorizes it;
- push, merge, deployment, or publication;
- live Donch source or instruction replacement.

The approved first repository task is limited to the source authority boundary and synthetic Capital.com fixtures. It does not authorize real data import or economics.

## No-vendor rule

Paid historical vendor data remain prohibited. Missing data lead to a bounded route such as `data_feasibility_required`, `blocked_by_missing_semantics`, `prospective_only`, `redesign_to_less_sensitive`, or `candidate_library_only`. Missing evidence is not zero cost.

## Current maturity

No current strategy is validation-grade or live-ready. Existing Kraken family decisions remain binding at their exact translation level. New Capital.com or cross-platform translations receive new identities and begin as untested.
