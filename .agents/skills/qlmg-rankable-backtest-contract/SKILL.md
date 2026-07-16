---
name: qlmg-rankable-backtest-contract
description: Enforce the QLMG evidence contract for any proposed, implemented, reviewed, or executed rankable historical backtest. Use when work touches signal scoring, tuning, thresholds, candidate or control identity, event simulation, outcomes, costs, funding, universes, point-in-time features, economic screens, family comparison, validation, or promotion language.
---

# QLMG Rankable Backtest Contract

## Check authorization before outcomes

Require explicit authorization for the exact economic screen. Documentation, infrastructure, refactoring, tests, package work, or a general request to improve the backtester does not authorize a run.

Without exact authorization, continue only with outcome-free planning, code, contracts, synthetic fixtures, and mechanical checks. State `economic_run_not_authorized`.

Never inspect data at or after `2026-01-01T00:00:00Z` for strategy outcomes unless a cited formal policy change and explicit human authorization replace the sealed-period rule. Keep live trading and order actions out of scope.

## Load the contracts

Read:

- [DATA_AND_PROTECTED_PERIOD_RULES.md](../../../docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md);
- [RUN_AND_ARTIFACT_CONTRACT.md](../../../docs/agent/RUN_AND_ARTIFACT_CONTRACT.md);
- [KNOWN_FAILURE_PATTERNS.md](../../../docs/agent/KNOWN_FAILURE_PATTERNS.md);
- current repository machine contracts and finalized manifests.

If a repository machine contract conflicts with narrative guidance, fail closed and reconcile it before continuing.

## Freeze the research object

Before outcomes, freeze and record:

- one mechanism and bounded definition set;
- Kraken-only venue and `[2023-01-01, 2026-01-01)` train bounds;
- point-in-time data, lifecycle, and universe authority;
- candidate economic address and parent-neutral raw signal identity;
- real, mechanism-relevant control classes and unique addresses;
- signal-state, non-overlap, exit, boundary, fill, funding, cost, and ambiguity rules;
- parameter budget, hard gates, evidence caps, and decision vocabulary;
- config, code, data, universe, and funding provenance.

Do not reduce control adequacy, change thresholds, or add filters after observing outcomes.

## Enforce mechanical gates

Fail rankability when any of these occur:

- pre-2023, Bybit, or protected rows enter active computation or output;
- current roster is represented as historical truth;
- a signal precedes listing or feature availability;
- maximum-hold preblocking occurs before actual exits;
- candidate or control identities are not frozen before outcomes;
- controls are placeholder, projected, duplicate-address, or mechanism-irrelevant;
- event sampling or capping occurs;
- summary or pooled rows are treated as trades;
- same-bar touch heroics are used in Tier-1 evidence;
- a boundary position is silently force-closed;
- imputed funding activates a signal;
- config, data, universe, code, funding, or output provenance is missing.

Keep last/trade fills, mark margin and liquidation, index anchoring, and signed notional funding distinct.

## Verify without contaminating the holdout

Use synthetic boundary fixtures and pre-2026 mechanical ledgers. Verify protected-row counts, PIT availability, lifecycle, duplicate identities, skip reconciliation, deterministic replay, funding partitions, and output manifests. Run only repository-supported commands recorded in the plan.

Do not claim validation grade from pooled means, parameter search, a hash pass, or a successful mechanical test. Keep evidence level separate from profitability and preserve negative or weak results under the frozen decision vocabulary.

## Preserve lineage and report

Write a new versioned root. Do not mutate or delete old roots. Record supersession explicitly. Report the exact authorization, commands, tests, artifacts, hashes, gates, evidence level, economic-run status, protected-outcome status, and unresolved caps.

## Boundaries

Do not use this skill for:

- a generic unit-test refactor unrelated to research semantics;
- a simple documentation or spelling change;
- review-package zipping or remote handoff alone;
- live capture engineering or execution calibration that contains no strategy selection.
