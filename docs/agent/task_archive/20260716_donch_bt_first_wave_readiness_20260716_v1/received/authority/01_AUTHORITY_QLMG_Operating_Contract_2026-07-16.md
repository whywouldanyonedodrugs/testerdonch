---
status: current operating authority
date: 2026-07-16
revision: 1.0
scope: Donch analysis, Kraken historical research, and public forward-capture governance
authority: user master prompt plus higher machine-enforced contracts where applicable
supersedes: active venue and period guidance in testmanual(1).txt and all Bybit-primary operating language
provenance: master prompt; QLMG interim reassessment; rev7 continuity brief; backtesterreport.md; Captureforwardreport.md
known limitations: repository-local machine contracts were not directly inspected in this task and remain higher authority when verified
---

# QLMG Operating Contract

## Purpose

This contract governs planning and handoffs across the Donch project, the historical backtesting repository, the forward-capture repository, and the human operator. It is an operating boundary, not a strategy specification.

## Binding constraints

```text
execution_venue: Kraken only
rankable_history_start: 2023-01-01T00:00:00Z
rankable_history_end: 2025-12-31T23:59:59.999999Z
protected_period_start: 2026-01-01T00:00:00Z
july_2026_capture_purpose: execution_calibration_only
paid_historical_vendor_data: prohibited
live_trading: not_authorized
live_repository_scope: public_forward_capture_only
private_kraken_endpoints: not_authorized_by_this_contract
orders: prohibited
old_donch_v3_s1: legacy_provenance_only
```

Pre-2023 information may inform literature review, historical illustration, and mechanism priors. It must not enter rankable Kraken results, tune thresholds, or serve as a manufactured validation sample.

External public data may be used as a point-in-time explanatory input. All proposed execution remains Kraken-only. Another venue's results, funding cycle, order semantics, or liquidity cannot be treated as Kraken evidence.

## Environment responsibilities

### Donch project

Analysis, planning, source governance, verification design, prompt construction, and human-readable continuity. It does not implement repository changes and does not imply shared filesystem access.

### Backtesting repository

Historical Kraken data engineering, mechanical QA, evidence contracts, reproducible research infrastructure, and report generation. Economic runs require an approved, frozen hypothesis contract and explicit authority.

### Forward-capture repository

Public Kraken market-data capture, schema and gap tracking, storage integrity, upload verification, and strategy-agnostic execution calibration. It has no order or live-risk mandate.

### Human operator

Chooses consequential policy, approves transfers and repository application, reviews plans and evidence, and authorizes any later holdout or deployment step. Technical uncertainty should be resolved through repository evidence and tests rather than shifted to the operator as an implementation guess.

## Authority and conflict resolution

Use this order:

1. machine-enforced contracts and finalized run manifests;
2. authoritative run roots, hashes, ledgers, and audit artifacts;
3. this contract and the current continuity brief;
4. current manuals and data-capability reports;
5. current strategic audits and agent reports;
6. research reports, catalogues, and public-source synthesis;
7. superseded or legacy material.

When two authorities conflict:

1. preserve both source identities;
2. identify the higher authority;
3. record the resolution and scope;
4. do not rewrite lower-authority history as though it never existed.

Known resolution: `testmanual(1).txt` names Bybit as primary. That venue guidance is superseded. Its evidence-contract and no-vendor principles remain useful where they do not conflict with current Kraken-only policy.

## Protected-data firewall

Data from 2026 onward must carry an explicit purpose label. Allowed labels are:

```text
protected_holdout
execution_calibration_only
prospective_shadow
data_engineering_only
```

The July 2026 capture defaults to `execution_calibration_only`.

Permitted uses include parser work, schema checks, gap and timestamp analysis, book reconstruction, strategy-agnostic spread/slippage surfaces, latency/staleness work, venue-rule checks, storage engineering, and separately reported observation of rules frozen before outcomes were viewed.

Prohibited uses include directional discovery, threshold selection, feature selection from forward returns, family ranking, symbol or session selection by performance, reopening a historical family, and any use of post-entry protected outcomes in promotion decisions.

Protected access must fail closed. If a task cannot prove that outcome-bearing rows are excluded, it must stop before reading them.

## Research and evidence contract

Every claim must identify:

- hypothesis identity and family lineage;
- venue and time scope;
- data sources and exactness partition;
- point-in-time universe rule;
- event/control identity;
- costs and execution assumptions;
- evidence level;
- reproducibility state;
- protected-data status;
- multiplicity and episode-overlap treatment;
- limitations and permitted next action.

The following are separate decisions:

| State | Meaning | Current program status |
|---|---|---|
| Hypothesis present | Mechanism is recorded and falsifiable. | Many |
| Evidence available | A bounded observation or ledger supports a claim. | Mixed by family |
| Reproducible | Inputs, code, config, universe, funding, and outputs are reconstructable. | Uneven; five family lineages lack recorded hashes |
| Validation-grade | Frozen candidate passed the approved protocol without protected-data contamination. | None |
| Deployment-approved | Human-approved live implementation and risk contract exist. | None; live trading is not authorized |

Absence of validation does not erase a hypothesis. Presence of evidence does not imply reproducibility. Reproducibility does not imply validation. Validation would not by itself authorize deployment.

## Machine-readable authority

Finalized manifests, schemas, ledgers, hashes, run roots, and fail-closed guards outrank prose summaries about their contents. Prose may explain a machine artifact but cannot change its rows or status.

No performance label may be inferred from a filename such as `final`, `validated`, `pass`, or `ready`. Use the artifact's current machine fields and verification record.

## No-vendor rule

Paid historical depth, trade, liquidation, OI, or similar vendor feeds will not be purchased. Missing data must lead to one of:

```text
progress_with_current_data
redesign_to_less_depth_sensitive
needs_capture_substitute
prospective_only
candidate_library_only
discard_current_translation_no_vendor_path
```

`waiting_for_vendor_data` is not a terminal plan.

## Research conduct

- Freeze hypotheses before an economic run.
- Keep early screening train-only and kill-only.
- Do not confer evidence by running more parameter combinations.
- Use real controls and event-level trade rows.
- Reconstruct point-in-time eligibility and lifecycle.
- Separate exact, mixed, and imputed funding.
- Treat pooled definition rows as diagnostic, not a portfolio or independent sample.
- Track effective trials and cross-family market episodes.
- Preserve closed translations and prohibited same-sample actions.
- Keep capacity secondary while treating execution realism, bad wicks, funding, lifecycle, liquidation, spread, slippage, and data integrity as primary.

## Approval gates

Human approval is required before:

- a new economic screen or backtest;
- any protected holdout access;
- redefining the protected-data policy;
- capture restart or a network-write smoke test;
- private Kraken access;
- any order or live-risk action;
- applying ready-to-use files to a remote repository with unknown Git state;
- uploading, replacing, or deleting Donch project sources;
- Git commit, push, merge, or production deployment.

This contract supplies no approval for those actions.

## Required completion language

Agents must distinguish `verified`, `inferred`, `proposed`, `unavailable`, and `blocked`. They must not claim a command ran, a test passed, a source was authoritative, or a repository changed without direct evidence.
