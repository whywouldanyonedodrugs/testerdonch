---
status: current readiness audit
date: 2026-07-16
revision: 1.1
scope: historical backtesting repository evidence, data, package, and infrastructure readiness
authority: backtesterreport.md reconciled with package preflight and the operating contract
supersedes: any implication that archive existence or zero recomputation mismatches makes the package release-ready
provenance: read-only backtesting audit; external-review package manifests; July 16 research audit
known limitations: read-only VS Code inspection verified `/opt/testerdonch`, branch `main`, and a heavily dirty working tree, but current commit/remotes, complete repository instructions, supported commands, and code compatibility remain unavailable; no economic run ran; no review Parquet or protected row was intentionally opened, but a generic workbook schema preflight unintentionally surfaced a limited first-row view from outcome-bearing Current Results; values were discarded and not used, period unknown, and the sheet was excluded thereafter
---

# Backtester and Evidence Readiness Audit

## Executive decision

```text
package_status: blocked_by_protocol_issue
package_release_ready: false
new_economic_run_authorized: no
protected_strategy_outcomes_found_in_review_package: no
historical_data_readiness: price_and_mark_event_research_with_lifecycle_funding_and_execution_caps
```

Archive creation, hash validation, zero recomputation mismatches, and zero protected rows are useful facts. They do not close the release protocol.

## Current repository application state

Read-only VS Code Remote SSH inspection verified `/opt/testerdonch` as the open Git repository on branch `main`. Source Control reported 239 pending changes, including at least 139 staged. Quick Open found no exact root `AGENTS.md` and no exact collision for the proposed `DATA_AND_PROTECTED_PERIOD_RULES.md`, but this is not a complete filesystem or instruction-chain audit. The current commit, remotes, supported test commands, and ownership of the existing changes were not established.

The local backtesting harness must not be overlaid on this checkout. Application requires an isolated worktree or other clean reviewed checkpoint, followed by repository-native checks and separate approval. No repository file was changed during the inspection.

## Review-package facts

The reported full package contains 386 manifest records and 14 family directories. Standardized ledgers are present for definition identity, candidates, controls, candidate-control matching, paths/MAE-MFE, verification-window indexes, and family-local identity/overlap.

Open evidence gaps:

| Gap | Scope | Consequence |
|---|---|---|
| Raw event-window trade-bar extracts | All 14 families | Independent event-level market-data verification cannot be completed from the package. |
| Raw event-window mark extracts | All 14 | Mark-aware verification remains indirect. |
| Raw event-window index/spot extracts | All 14 | Historical source is unavailable and no extract exists. |
| Raw event-window exact-funding extracts | All 14 | Funding join cannot be independently rechecked from raw windows. |
| Full MAE/MFE population | TSMOM | Path evidence is incomplete. |
| Prior-high runner source snapshot | One expected tool | Code-lineage package is incomplete. |
| Test and failure counts | All 14 | A blank matrix cannot establish that tests ran or passed. |
| Reproducibility hashes | TSMOM, A1, prior-high, C2, LFBS | Commit, code, config, data, universe, and funding identity are not recorded. |
| Cross-family episode identity | Program-wide | Definition and family rows may reuse the same market episode without one canonical cluster ID. |

The four families whose recomputation status was `source_summary_unavailable` were TSMOM, A1/compression, prior-high/reclaim, and C2. A pass-like label that includes “source not recorded” is not an affirmative verification result.

## Package interpretation rules

- `candidate rows` and `control rows` are definition-event records, not independent trials.
- Pooled definition means are diagnostic, not portfolio returns or family expectancy.
- Family-local exact overlap does not answer nearby cross-family episode overlap.
- A standardized result package is not the same as a self-contained raw verification package.
- A hash pass proves identity against the manifest, not correctness of the economic interpretation.
- Zero protected rows in the package does not remove the need for a fail-closed protected input boundary.

## Historical data readiness

Available for the acquired current-roster cohort:

- official 5-minute Kraken trade/last bars for 2023-2025;
- official 5-minute mark bars for 2023-2025;
- exact funding from about 2025-06-26 onward;
- frozen fee, funding-imputation, and slippage scenarios;
- current instrument metadata and opening dates.

Not acquired as usable 2023-2025 authority:

- index/spot, OI, basis, typed forced flow, CVD, spread, depth, observed liquidity, observed slippage;
- archived delistings, suspensions, settlements, historical specification changes;
- a survivorship-free universe.

The analytics probe omitted a valid interval. The correct conclusion is `not acquired; availability unresolved`, not `Kraken has no such data`.

## Protected-data boundary

The review package reports zero protected-period candidate rows and family ledgers end by 2025-12-31. The common loader filters rankable trade/mark rows below 2026.

Two caps remain:

1. the 2026 current instrument roster influenced which historical symbols were acquired;
2. funding files can be read into memory before the date filter because they are not date-sharded.

Required repair is physical input separation or predicate pushdown before protected rows are loaded. Do not rely only on downstream scoring filters.

## Mandatory outcome-free closure

### Package protocol

1. Decide whether raw event-window extracts are required; generate them without economic reselection or narrow the release contract.
2. Restore the missing prior-high source snapshot or record why it cannot be reconstructed.
3. Record exact test commands, counts, failures, exit codes, environment, and timestamps.
4. Recover or explicitly mark irrecoverable the five-family reproducibility fields.
5. Complete or accurately cap TSMOM MAE/MFE coverage.

### Universe and protected inputs

1. Build a PIT Kraken instrument/lifecycle authority from archived public sources, with uncertainty.
2. If that is not possible, label every cross-sectional result `current_roster_bar_existence_cohort`, never `survivorship_free`.
3. Partition protected files physically and make rankable loaders fail closed.
4. Ensure imputed funding cannot activate a feature, gate, or promotion result.

### Multiplicity and identity

1. Assign one canonical episode ID across all families.
2. Record exact-event overlap separately from nearby episode overlap.
3. Register every tried family, component, and adaptive branch.
4. Report effective trials and cluster-aware uncertainty.

### Reporting

1. Produce definition-level and episode-cluster-level summaries.
2. Keep row-weighted pools diagnostic.
3. Freeze risk denominators and cost scenarios before outcomes.
4. Separate evidence level, reproducibility state, validation state, and deployment authority.

## Infrastructure capability state

The backtesting audit reported:

| Capability | State | Current implication |
|---|---|---|
| Continuous cross-sectional ranks | Ready | Can support PIT ranks after universe repair. |
| Exact multi-lot exits and adds | Missing | Immediate starter plus retest add cannot be tested exactly yet. |
| Shared component ablation | Partial | Family-specific controls exist; no universal component-vector engine. |
| Cross-family episode IDs | Missing | Multiplicity and diversification claims remain capped. |
| PIT catalyst joins | Partial | C2-specific ingestion exists; not general. |
| PIT sector joins | Partial | Seed exists; not a full historical Kraken authority. |

Implement infrastructure only for a repeated, approved need. Deterministic identity, protected-input separation, lifecycle authority, and verification are required now. Hypothesis-specific OI, liquidation, or exact multi-lot engines can wait for an approved contract that needs them.

## Acceptance criteria for readiness review

Readiness may be reconsidered when:

- every open package field has a verified value or explicit irrecoverable status;
- raw-verification claim and payload agree;
- the current roster cap is repaired or stated everywhere;
- protected rows cannot enter rankable loaders;
- cross-family episode and multiplicity records exist;
- the central family registry matches current roots and closed actions;
- a fresh independent review finds no silent downgrade of the protocol.

Passing this review would permit a human decision about a new economic task. It would not itself authorize one.
