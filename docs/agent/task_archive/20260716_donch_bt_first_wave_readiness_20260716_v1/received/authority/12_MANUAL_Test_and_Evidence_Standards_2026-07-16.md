---
status: current testing manual
date: 2026-07-16
revision: 1.0
scope: Kraken historical research evidence contracts, test design, reproducibility, and decision states
authority: operating contract plus machine-enforced repository contracts when verified
supersedes: testmanual(1).txt as active manual; retains its sound evidence policy while replacing Bybit and period guidance
provenance: testmanual(1).txt; backtester audit; July 16 research audit; efficient-research and robust-protocol page extractions
known limitations: repository-local command names and current test entry points require direct verification before application
---

# Test and Evidence Standards

## Purpose

This manual prevents mechanical errors, hidden selection, false precision, and claims that exceed the evidence. It governs method; it does not authorize a run.

## Fixed scope

```text
venue: Kraken only
rankable_period: 2023-01-01 through 2025-12-31
protected_period: 2026-01-01 onward
paid_historical_vendor_data: prohibited
live_trading: not_authorized
```

Any rankable runner must fail closed if another venue, pre-2023 rows, or 2026+ protected rows enter feature selection, scoring, return analysis, parameter selection, or candidate ranking.

## Evidence levels

Use the existing eight-level vocabulary:

| Level | Required content | Claim boundary |
|---|---|---|
| `level_0_hypothesis_only` | Falsifiable mechanism and required data. | Idea exists; no empirical support. |
| `level_1_generator_support` | Signal generator produces causal candidates with identity checks. | Generator works mechanically. |
| `level_2_path_or_mae_mfe_support_only` | Path, entry, or MAE/MFE evidence without full trade ledger. | Bounded path or execution hypothesis. |
| `level_3_event_level_trade_ledger` | Event-level trades with causal entry/exit and costs. | Train-only candidate economics, subject to caps. |
| `level_4_event_ledger_plus_real_controls` | Real controls and component isolation. | Candidate may be compared with named controls. |
| `level_5_walkforward_cpcv_parameter_stability` | Purged/embargoed train-only validation, parameter stability, cluster and multiplicity treatment. | Family-specific train-only validation candidate. |
| `level_6_final_holdout_still_sealed` | Frozen candidate and documented untouched holdout. | Holdout remains sealed; not evidence of holdout success. |
| `level_7_execution_depth_or_live_capture_evidence` | Queue/depth/capture or equivalent execution calibration under an approved purpose. | Execution feasibility or calibration; not directional validation by itself. |

Evidence level measures artifact depth, not strategy quality. A weak family with clean level-4 evidence can be more informative than a strong-looking result with contaminated inputs.

## Five status dimensions

Every registry row must store these separately:

```text
hypothesis_status
evidence_level
reproducibility_status
validation_status
deployment_status
```

Examples:

- a hypothesis can be present at level 0;
- a level-4 result can be reproducibility-blocked;
- a reproducible negative result is not validation;
- a frozen level-5 candidate still has no deployment authority;
- level-7 execution evidence cannot turn an unvalidated direction into a live strategy.

## Required hypothesis contract

Before an economic run is proposed, freeze:

```text
hypothesis_id_and_family
economic_mechanism
compelled_actor_or_state
direction_and_horizon
PIT_universe
activation_and_disable_rules
entry_stop_exit_and_timeout
required_data_and_exactness
fees_funding_spread_slippage
main_null_control
component_ablations
falsification_rule
expected_failure_modes
multiplicity_family
protected_data_policy
acceptance_and_stop_rules
```

If the mechanism cannot be distinguished from a renamed closed translation, register it as the same family or reject the task before running.

## Data contract

Each input must declare source, path, hash, schema, venue, resolution, time coverage, PIT meaning, exactness, protected classification, and known gaps.

Required guards:

- immutable raw input identity;
- date and venue checks before feature construction;
- point-in-time instrument eligibility;
- no current-only taxonomy as historical truth;
- exact/mixed/imputed funding partition;
- mark and trade price kept distinct;
- unknown lifecycle dates remain unknown;
- protected files physically separated or filtered before load;
- no silent replacement of missing OI, basis, depth, or typed flow with chart proxies.

## Candidate and event identity

Candidate generation must be parent-neutral where required, causal, and independent of exit outcomes. Event identity must include timestamp, symbol, family, definition, data version, and economic address.

Apply non-overlap using actual executable exit state, not nominal maximum hold. A repaired event tape is a new lineage and must supersede the defective root explicitly.

Store one canonical cross-family episode ID in addition to family-local exact identity. Nearby episodes and exact duplicates are different overlap questions.

## Event-level ledger

Minimum fields:

```text
run_id
family_id
definition_id
candidate_id
economic_address
canonical_episode_id
symbol
signal_ts
decision_ts
entry_ts_and_price
stop_and_risk_denominator
exit_ts_price_and_reason
trade_and_mark_path_refs
gross_return_and_R
fees_funding_spread_slippage
net_return_and_R
funding_exactness
universe_version
data_and_code_hashes
control_linkage
protected_row_count
```

No PF, drawdown, Sharpe, CAGR, `validated`, `beats_controls`, or similar label may be computed from projected means, summary rows, synthetic event R, or anything other than contract-valid event rows.

## Controls

Controls must be real, prospectively comparable, and mechanism-relevant. Common classes:

- component removal;
- immediate entry versus retest;
- matched random or non-event dates;
- parent-only or beta-only response;
- symmetric directional control;
- liquidity, risk-denominator, and regime matching;
- timestamp-delay and execution sensitivity;
- current-survivor versus PIT membership.

Freeze risk denominators and control matching before outcomes. Do not repair extreme control R retrospectively with a floor or winsorization rule chosen after review.

## Costs and execution

- Use venue- and effective-date-specific rules where available.
- Distinguish documented fees from assumed account tiers.
- Keep exact, mixed, imputed, and zero-boundary funding separate.
- Use mark-aware risk and do not treat last-price stops as liquidation models.
- Do not infer passive fills from candle touches.
- Model urgent orders as potentially partial or unfilled when Kraken price protection or available depth matters.
- Keep approximate fills in screening only; they cap claims.
- Do not use aggressive leverage to rescue weak expectancy.

## Validation and multiplicity

The 2026+ holdout remains sealed. Within 2023-2025:

- use purging and embargo around overlapping labels and trades;
- prefer rolling and multiple-path train-only evaluation;
- measure symbol, time, regime, and episode concentration;
- cluster near-identical candidates before expensive validation;
- register every tried family, parameter branch, and follow-up;
- apply family-aware multiplicity and cluster-robust uncertainty;
- report plateaus and stability, not the single best parameter point.

The same three-year tape has been reused adaptively. Multiplicity correction is necessary but does not create independent evidence. Frozen prospective observation is the main next independent source; holdout access requires a separate human decision.

## Efficient research funnel

```text
immutable_raw_data
-> versioned_causal_features
-> frozen_signal_candidates
-> train_only_kill_screen
-> duplicate_and_episode_clustering
-> exact_event_and_execution_simulation
-> real_controls_and_ablations
-> purged_train_validation_and_multiplicity
-> frozen_candidate
-> separately_approved_holdout_or_prospective_observation
```

Early screens may reject obvious failures. They must not promote. Small time or symbol samples may be used only when predeclared, structurally representative, and kill-only.

## Reproducibility contract

Every decision-bearing run must record:

- Git commit and dirty state;
- code and source snapshot hashes;
- configuration and parameter manifest hashes;
- data, universe, funding, catalyst, and taxonomy hashes;
- dependency or environment identity;
- deterministic seeds;
- exact commands and exit codes;
- test names, counts, failures, and logs;
- artifact manifest and final run root;
- superseded roots and reason.

Blank test counts are unknown, not zero. `pass_or_source_not_recorded` must be split into an affirmative pass or an unavailable status.

## Hard rejection and soft caps

Hard rejection applies to logic or evidence failure: lookahead, impossible fills, future-aware filters, wrong venue/period, arithmetic mismatch, no required event ledger, synthetic controls, or a translation that is negative across clean controls and has no distinct mechanism.

Use a soft cap for sparse events, regime or symbol concentration, incomplete exact funding, lifecycle uncertainty, missing depth, incomplete reproducibility, or a coherent mechanism tested only by a weak proxy.

Allowed bounded decisions include:

```text
current_translation_rejected_only
mechanism_preserved_current_translation_weak
sample_limited_mechanism
fragile_context_sleeve
candidate_library_only
prospective_only
blocked_by_protocol_issue
```

## Required report

Every report must state:

1. hypothesis and mechanism;
2. data and universe authority;
3. exact period and protected-row count;
4. event/control identity and overlap;
5. execution and cost assumptions;
6. evidence level and reproducibility state;
7. train-only results with concentration and uncertainty;
8. multiplicity registry update;
9. failures and unavailable evidence;
10. exact decision, prohibited next actions, and permitted next step.

No report may call a strategy live-ready under this manual.

## Source trace

- `testmanual(1).txt`, sections 3-6 and 17-22: no-vendor policy, evidence levels, contracts, funding, validation, and reporting. Its Bybit venue sections are superseded.
- `backtesterreport.md`: current data and package gaps.
- `Efficient Alpha Research for Python Systematic Trading-1.txt`, pages 1-3: staged funnel, kill-only screening, deduplication, and multiple-testing discipline.
- `QLMG Robust Backtest Protocol for Qullamaggie-Inspired Crypto Perpetual Strategies-2.txt`: general event-ledger and reproducibility prior; its multi-venue default is superseded.
