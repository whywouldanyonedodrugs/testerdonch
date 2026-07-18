---
status: proposed current continuity after application
date: 2026-07-18
revision: rev9
scope: cross-environment Donch state after multi-platform approval and repository preflight
authority: current narrative continuity below machine contracts, finalized manifests, run roots, hashes and ledgers
supersedes: 02_STATE_Master_Continuity_Brief_2026-07-16_rev8.md after verified application
provenance: rev8; approved 2026-07-18 policy; verified Capital.com expansion preflight handoff
known limitations: Capital.com downloader output is managed elsewhere and no verified acquisition handoff has yet been received
---

# Master Continuity Brief, rev9

## Current decision

The program now has an approved two-platform research policy, but only the non-economic source-authority and synthetic-adapter task is authorized. No economic run, real Capital.com import, protected-outcome access, order, risk change, push, merge, deployment, or live Donch source replacement is authorized.

## Fixed scope

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

## Repository preflight

Verified on 2026-07-18:

- repository `/opt/testerdonch`;
- clean `main` at `9df9c4f319cceffce84fc4567ddcaec9333c2bb7`;
- synchronized with `origin/main`;
- applicable root `AGENTS.md`;
- existing regression baseline: 49 passed, 0 failed after correction of one nonexistent test-module command;
- no protected outcomes, economic commands, Capital.com payloads, or decision-bearing changes;
- Drive handoff package hash `3f3adc45590cbf7c7f1e9e26f7baf06f066b729345451a65973a88a3eb2ec2f8` independently verified.

Repository state must still be rechecked at the start of implementation.

## Minimal architecture decision

Use a shared authority/evidence boundary and separate source adapters. Do not create a universal simulator or refactor established Kraken runners.

First repository patch:

- source-neutral pre-open manifest authority guard;
- Kraken compatibility call with unchanged behavior;
- synthetic Capital.com bid/ask adapter boundary;
- focused tests and repository contract documentation;
- isolated local reviewed commit; no push.

Existing Kraken identities, results, roots, and hashes remain unchanged. New platform-aware outer identities wrap them only when needed.

## Capital.com data state

The downloader is active in a separate workflow, but no closed acquisition handoff is available here. Consequently, the following are `unavailable` for research authority:

- acquired file manifest and per-file hashes;
- actual instrument epics and coverage;
- normalized schema and gap report;
- historical calendars, expiry and market-status history;
- historical financing and currency-conversion semantics;
- dividend/corporate-action adjustments;
- volume semantics;
- rankable/protected physical partitioning.

Real import and economics remain blocked until the data-intake contract is satisfied.

## Research state

All Kraken decisions from rev8 remain unchanged. No Capital.com or cross-platform translation has evidence. The hypothesis registry rev2 adds route fields only. A route such as `translation_candidate_unreviewed` means the mechanism may warrant later translation review; it does not mean the Kraken result was portable or that a test is approved.

## Protected state

- Kraken July 2026: `execution_calibration_only`.
- Capital.com 2026+: `data_engineering_only` by default.
- Protected rows must fail before rankable payload load whenever manifest partitioning permits.
- No 2026 outcome can choose instrument, threshold, direction, feature, family, or economic priority.

## Immediate authorized action

Execute `Multi-Platform Rankable Source Authority Contract and Synthetic Adapter Boundary v1` under the exact approved task specification. Then review the actual Drive package and local commit before deciding whether to accept or apply the next phase.

## Next blocked phase

Capital.com metadata intake becomes eligible only after the downloader produces a closed, credential-free package meeting `19_CONTRACT_Capitalcom_Data_Intake_2026-07-18_rev1.md`. Market-payload normalization and economic research remain later, separate approvals.
