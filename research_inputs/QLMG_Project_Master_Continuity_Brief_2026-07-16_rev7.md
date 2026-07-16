# QLMG / Donch Crypto Perpetual Research Master Continuity and Research Brief

**Date:** 2026-07-16  
**Revision:** rev7  
**Status:** active master continuity / current-state handoff  
**Scope:** Donch / QLMG crypto-perpetual research, Kraken historical backtesting, forward-capture sidecar, shared cost/funding modeling, completed signal-state repair campaign, all clean strategy-family results through H31/H32, the external-review evidence-package phase, and a portable map of project files  
**Live trading:** not authorized  
**Paid historical vendor data:** prohibited  
**Protected strategy-selection cutoff:** `2026-01-01T00:00:00Z`

This document supersedes:

- `QLMG_Project_Master_Continuity_Brief_2026-07-16_rev6.md`;
- `QLMG_Project_Master_Continuity_Brief_2026-07-13_rev5.md`;
- `QLMG_Project_Master_Continuity_Brief_2026-07-12_rev4.md`;
- `QLMG_Project_Continuity_Brief_2026-07-08_rev3.md`;
- `QLMG_Project_Continuity_Brief_2026-07-07_rev2.md`.

It is the current **single human-readable source of truth** for project state, family decisions, repaired evidence lineages, operating constraints, authoritative run roots, current documentation work, and the location and purpose of files supplied to outside reviewers. The external-review evidence package currently being built will be the most complete portable evidence object, but it does not replace finalized run manifests or machine-enforced contracts as execution authority.

Machine-enforced contracts remain authoritative for execution:

- `tools/qlmg_evidence_contracts.py`;
- `tools/qlmg_signal_state_contract.py`;
- versioned manifests and data hashes;
- canonical candidate and control economic-address contracts;
- protected-period, point-in-time, funding, boundary, and leakage audits.

If this brief conflicts with a finalized run manifest or a machine-enforced contract, fail closed and reconcile the discrepancy before continuing.

The backtesting AI agent, live-capture agent, and this ChatGPT project do not automatically share files. Every prompt to an agent must be self-contained and include exact roots, objective, frozen scope, forbidden actions, tests, outputs, and final decision vocabulary.

---

## 1. Revision 7 additions

Revision 7 incorporates all completed work after rev6 and adds a portable project-file map for reviewers who receive the sources outside ChatGPT.

Major changes:

1. The BTC-led delayed-alt-diffusion screen completed under the repaired signal-state contract.
2. H43 produced one highly overlapping moderate-lag / both-up signal region across 4h, 8h, and 12h exits. The best expression, `h43_v1_003`, was conservatively positive and broad but failed severe costs, had weak exact-funded support, and did not establish consistent incremental value from residual-lag ranking. It remains a fragile context sleeve only.
3. The Asia / U.S. cash-open range-resolution H31/H32 screen completed. All 24 definitions were negative under conservative costs, so the tested opening-range translation is closed as `current_translation_rejected_only`.
4. The project has now completed a large part of the clean, bar-compatible Tier-1 hypothesis catalogue without producing a level-5 strategy.
5. Automatic sequencing into additional lower-priority hypotheses is paused.
6. A dedicated external-review evidence-package phase has been commissioned. It must resolve authoritative roots, standardize all family ledgers and statistics, include trader-readable strategy cards, preserve quant and engineering evidence, recompute decision metrics, and package full event/control and MAE/MFE data without opening the holdout or launching new research.
7. The expected documentation run root is:

   ```text
   results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/
   ```

8. No new economic screen should launch until that package is complete, root authority and recomputed metrics pass, and the package has been reviewed internally and by outside analysts.
9. This brief now includes a portable file/folder convention, an inventory of project-source files, a map of current versus superseded compact bundles, and guidance for reviewers who receive a separate folder rather than access to the original ChatGPT project.
10. No strategy has reached level 5, no final holdout has been opened, and no live trading is authorized.

---

## 2. Executive state summary

### 2.1 Project identity and business objective

The active project is **Donch / QLMG Crypto Perpetual Research**.

The old Donch V3 / S1 / state-transition / sidecar strategy program is legacy. It is retained for provenance, infrastructure lessons, and failure analysis, but it is not used for active strategy selection.

The active research program studies systematic crypto-perpetual strategies adapted from Qullamaggie-style ideas and related crypto-market mechanisms:

- liquid-leader continuation;
- prior impulse and compression;
- prior-high proximity and reclaims;
- failed breakouts and failed bounces;
- backside-confirmed blowoff shorts;
- structural breakdown failures and squeeze reclaims;
- catalyst continuation bases;
- delayed forced-flow reversal proxies;
- session and handoff effects;
- synchronous relative strength and lagged BTC-to-alt diffusion;
- both long and short branches.

The business objective is aggressive growth from small capital through genuine repeatable edge. Capacity is secondary. Execution realism, funding, bad wicks, liquidation, lifecycle, delisting, data quality, and slippage are primary.

No current strategy is authorized for live trading.

### 2.2 Current top-level decisions

```text
Old Donch V3/S1:
    legacy_only

Live trading:
    not_authorized

Paid historical vendor data:
    prohibited

Final strategy-selection holdout:
    sealed from 2026-01-01T00:00:00Z onward

Current level-5 candidate:
    none

Strongest preserved research object:
    rfbs_v1_010
    fragile_context_sleeve
    level_4_event_ledger_plus_real_controls
    failed level-5 stability review

Other preserved fragile sleeves / regions:
    lfbs_v1_021
    bcbs_v1_008
    fbsr_v1_001 / 007 / 019 / 021
    rsbb_v1_013 / 014
    h43 moderate_lag + both_up region:
        h43_v1_001 / 002 / 003 exit expressions

Recently closed translations:
    close-confirmed breakout retest v2
    strong-close session handoff
    Asia / U.S. session-open range resolution H31/H32

Current active phase:
    external-review evidence-package construction
    no new economic hypothesis running

Expected documentation root:
    results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/

Next strategic decision:
    external package verification
    outside trader / quant / engineering review
    then a bounded remaining-hypothesis and data-readiness decision
```

### 2.3 Economic reality

No tested strategy has earned validation-grade or stability-ready status.

The strongest current result remains `rfbs_v1_010`:

```text
setup:
    moderate +12% / 3-day / 1.5-ATR failed bounce
    three-bar completed failure confirmation
    broader fragile/countertrend/stress parent policy
    fixed 72-hour short

events: 148
symbols: 61
active months: 27
base mean: +0.336R
conservative mean: +0.298R
severe mean: +0.170R
conservative PF: 1.76
after top-three removal: +0.199R
worst leave-one-symbol: +0.245R
worst leave-one-month: +0.157R
```

Its train-only stability review was encouraging but failed level 5:

```text
powered walk-forward folds:
    4/4 positive under conservative and severe costs

CPCV K=8:
    85.7% conservative-positive paths
    71.4% severe-positive paths

failed gates:
    multiplicity adjustment
    clustered confidence

conservative DSR-equivalent probability: 0.7606
severe DSR-equivalent probability: 0.3396
Reality Check p-value: 0.1387
month-bootstrap conservative 95% CI: [-0.105, +0.620]
month-bootstrap severe 95% CI: [-0.256, +0.485]
```

It remains a preserved research sleeve only. It is not a strategy-selection, portfolio, holdout, or live-trading object.

The newer H43 result is broader in event count but materially weaker:

```text
best expression:
    h43_v1_003
    moderate_lag + both_up + fixed 12h

events: 914
conservative mean: +0.093R
severe mean: -0.026R
conservative top-1% trimmed: +0.006R
exact-funded conservative mean: +0.038R on 25 events
```

It supports BTC-impulse timing under an already positive parent state more than it supports the specific bottom-residual lag ranking.

### 2.4 Immediate next action

The project is in an **evidence consolidation and external review phase**, not another automatic hypothesis screen.

The backtesting agent has been instructed to build one standardized package covering every clean, decision-bearing tested family. Required outputs include:

- authoritative-root and supersession registries;
- trader-readable strategy cards;
- complete frozen definition and formula documentation;
- standardized candidate and control ledgers;
- MAE/MFE and path diagnostics;
- coverage, concentration, period, funding, and control statistics;
- code lineage, source snapshots, unit-test evidence, reproducibility hashes, and hard-gate matrices;
- independently recomputed principal metrics;
- a compact core ZIP and full compressed evidence archive.

Expected package files:

```text
qlmg_external_review_core_20260716_v1.zip
qlmg_external_review_full_20260716_v1.tar.zst
```

No new family should launch until:

1. every family resolves to a current authoritative lineage;
2. no superseded economics are presented as current;
3. recomputed metrics match finalized reports;
4. no protected-period rows or secrets are present;
5. outside review has identified the most important remaining questions;
6. a human decision selects the next phase.

The likely phase after external review is a bounded **remaining-hypothesis and data-readiness review**, not an immediate economic screen.

---

## 3. Source hierarchy and authority

### 3.1 Evidence-policy authority

`testmanual.txt`, version date 2026-06-30, remains the operating evidence-policy source of truth unless explicitly superseded by a frozen newer contract.

It defines:

- no live authorization;
- no paid historical vendor data;
- evidence levels;
- real event-ledger and control requirements;
- candidate-library preservation;
- hard rejection versus soft caps;
- no placeholder controls;
- no current-only taxonomy as historical truth;
- no promotion language without the required evidence.

### 3.2 Current-state authority

This rev7 brief is the human-readable authority for:

- current family routing;
- repaired and superseded run lineages;
- latest family decisions;
- signal-state architecture;
- funding and cost policy;
- active project and documentation phase;
- prompt and workflow standards.

### 3.3 Important project sources

Core sources include:

- `testmanual.txt`;
- `Initial-conversion-plan.txt`;
- `QLMG_Project_Master_Continuity_Brief_2026-07-13_rev5.md`;
- `catdb.md`;
- `QLMG_Hypothesis_Library_2026-07-01.xlsx`;
- `Efficient Alpha Research for Python Systematic Trading.pdf`;
- QLMG rulebook, perpetual-market-structure, robust-backtest, momentum/reversal, catalyst, sector, and no-vendor reports;
- MOV venue-contingency and Kraken migration reports;
- finalized run manifests, decision summaries, event ledgers, controls, and compact bundles.

Research reports generate priors. Finalized machine-enforced run artifacts determine project decisions.

---

## 4. Evidence contract

### 4.1 Evidence levels

```text
level_0_hypothesis_only
level_1_generator_support
level_2_path_or_mae_mfe_support_only
level_3_event_level_trade_ledger
level_4_event_ledger_plus_real_controls
level_5_walkforward_cpcv_parameter_stability
level_6_final_holdout_still_sealed
level_7_execution_depth_or_live_capture_evidence
```

Evidence level describes evidence type, not profitability.

Minimum claim requirements:

```text
Idea worth preserving:
    level_0 or higher

Path or entry hypothesis:
    level_2 or higher

Train-only research candidate:
    level_3 or higher

Candidate claimed to beat controls:
    level_4 with adequate unique-address controls

Candidate for final family-specific validation:
    level_5 on train-only data
    final holdout untouched

Execution-sensitive telemetry candidate:
    level_7 or documented capture substitute

Live-ready:
    not authorized
```

### 4.2 Non-negotiable rules

- No placeholder, synthetic, copied, or projected controls.
- No summary rows treated as event-level trades.
- No future-aware regime labels.
- No full-sample thresholds used as point-in-time inputs.
- No current-only taxonomy backfilled as historical truth.
- No event sampling or event caps in rankable runs.
- No current-live universe backfill.
- No touch fills, queue assumptions, or same-bar heroics in Tier-1 evidence.
- No promotion from aggregate-only output.
- No use of data at or after `2026-01-01T00:00:00Z` for scoring, tuning, controls, or portfolio selection.
- Boundary-crossing entries, holds, outcomes, funding intervals, and controls must be explicitly dropped or censored.
- Artificial closure at a research-window endpoint is prohibited.
- Imputed funding may affect outcome cashflow only; it may never activate signals.
- Severe stress is adversarial, not the sole base economic model.
- No validation, holdout, portfolio, or live work without explicit approval.

### 4.3 Canonical candidate identity

Every generator must emit one genuine economic signal per unresolved setup.

Required candidate economic address fields normally include:

```text
symbol
decision timestamp
entry
stop
risk denominator
exit policy
maximum exit
```

Rules:

- duplicate candidate economic addresses must be zero;
- duplicate symbol-decision candidates must be zero;
- repeated bars from one unresolved sequence cannot become multiple trades;
- canonical selected keys are frozen before outcomes;
- overlapping positions are controlled per symbol and definition using actual executable exits.

### 4.4 Unique control identity

Named control labels are not independent evidence if they share one economic address.

Every control receives a control economic-address hash. Report:

- class-level coverage;
- unique-address coverage;
- matched candidate subset;
- actual unmatched-only candidate complement;
- matched-versus-unmatched bias;
- reasons for zero or inadequate controls.

The recent default adequacy rule is:

```text
coverage >= 70%
unique economic addresses >= 15
```

A different rule must be frozen before outcomes.

### 4.5 Advancement standard

A train-screen lead should normally show:

- positive base and conservative economics;
- preferably positive severe economics;
- sufficient event, symbol, month, and regime coverage;
- positive top-one and top-three removal appropriate to sample size;
- period stability or a predeclared context explanation;
- at least two adequate unique-address contextual/structural controls;
- exact-funding support or an explicit cap;
- parameter-neighbourhood support;
- no mechanical defect.

Level 5 additionally requires meaningful walk-forward/CPCV, clustered confidence, and multiple-testing control.

Sparse positive objects can remain sleeves. They cannot be promoted.

---

## 5. Signal-state contract and repair campaign

### 5.1 Defect discovered

The original LFBS, Backside, RFBS, and delayed-flush runners prematurely suppressed later signals using a nominal 72-hour or seven-day maximum-hold block before actual definition exits were known.

This could remove legitimate re-entries after:

- early stops;
- early EMA exits;
- fixed holds ending before seven days;
- earlier structural exits.

It altered candidate identity, counts, controls, parent-policy nesting, and economics.

### 5.2 Production contract

The production contract is:

```text
signal_state_contract_v1_20260715
```

Required architecture:

```text
parent-neutral mechanical raw signal tape
→ immutable freeze and hashes
→ PIT parent-policy projections
→ definition-local chronological simulation
→ non-overlap using actual executable exit_ts
→ complete skip ledger
→ accepted-trade freeze
→ control freeze
→ outcome analysis
```

Forbidden:

```text
signal generation
→ nominal maximum-hold preblock
→ later actual-exit simulation
```

### 5.3 Machine-enforced requirements

Future rankable runs must record:

```text
signal_state_contract_version
parent_neutral_raw_signal_hash
parent_policy_projection_hash
definition_trade_tape_hash
non_overlap_skip_ledger_hash
preblock_applied = false
```

The evidence contract must fail closed if:

- the raw tape is absent;
- maximum-hold preblocking occurred;
- parent projections do not reconcile;
- eligible rows do not equal accepted rows plus actual-open-position skips plus declared exclusions;
- position state is shared across definitions;
- deterministic replay fails.

### 5.4 Repair campaign

Campaign root:

```text
results/rebaseline/phase_kraken_cross_family_signal_state_repair_campaign_20260715_v1
```

Closure root:

```text
results/rebaseline/phase_kraken_cross_family_repair_campaign_closure_20260715_v1
```

Campaign result:

```text
six affected decision-bearing entries identified
six repaired and closed
unresolved entries: 0
historical roots preserved unchanged
```

Affected repaired lineages:

- LFBS screen;
- LFBS 2023 presample;
- LFBS canonical adjudication;
- Backside screen;
- RFBS screen;
- RFBS downstream materialization.

The delayed-flush repaired root is also valid and uses the same architecture.

---

## 6. Perpetual-specific standards

A defensible crypto-perpetual backtest distinguishes:

- last/trade price for fills;
- mark price for unrealized PnL, margin, and liquidation;
- index/oracle price for mark/funding validation;
- funding as signed cashflow on notional at venue settlement boundaries;
- maker/taker, settlement, delisting, and liquidation charges;
- listing, prelaunch, live, halt, reduce-only, delisting, and settlement states;
- bar ambiguity and execution assumptions;
- venue-specific margin and liquidation mechanics.

Point-in-time requirement:

```text
feature_available_ts <= decision_ts
```

Additional rules:

- no contract-specific history before listing;
- no historical OI gate unless a PIT OI audit passes;
- no imputed-funding signal gate;
- no current decision bar in a historical lookback unless it is explicitly the completed trigger bar;
- funding is applied only after entry at actual settlement boundaries;
- accepted intervals crossing evaluation boundaries are excluded or censored, never artificially closed.

OHLCV-only results are screening evidence. Slower liquid families can reach level 4 using bars, mark, funding, lifecycle, conservative costs, and real controls. Microstructure-dependent families remain capture-capped.

---

## 7. Venue and no-vendor policy

Historical research currently uses Kraken derivatives because of Bybit/EU continuity risk.

Kraken is suitable for:

- liquid and upper-mid-cap perps;
- medium-frequency continuation and reversal;
- candles, trades, mark events, funding, lifecycle, and official analytics.

Kraken is weaker for:

- broad Tier-C opportunity density;
- rapid new-listing research;
- complete historical depth;
- full liquidation sequencing;
- some small-cap and microstructure hypotheses.

Paid historical feeds such as Tardis, Kaiko, Amberdata, CoinAPI, or Coin Metrics will not be purchased.

`waiting_for_vendor_data` is not an allowed terminal status.

Use:

```text
progress_with_current_data
needs_capture_substitute
redesign_to_less_depth_sensitive
candidate_library_only
discard_current_translation_no_vendor_path
preserve_hypothesis_generate_new_variant
blocked_by_protocol_issue
```

---

## 8. Shared cost and funding policy

### 8.1 Default Tier-1 Kraken costs

```text
Base:
    5 bps taker per side
    + 4 bps round-trip slippage

Conservative:
    5 bps taker per side
    + 8 bps round-trip slippage

Severe:
    10 bps per side
    + 12 bps round-trip slippage

Zero-fee:
    diagnostic only

Zero-funding:
    diagnostic only
```

A family-specific contract may differ only if frozen before outcomes.

### 8.2 Shared funding model

Authoritative root:

```text
results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1
```

Model hash:

```text
0054af0ee40740e39739bfade92f342867bb208a4fe7ed15b151a8a0a838d072
```

Selected model:

```text
symbol_robust_location_shrunk_to_liquidity_tier
```

Rules:

- exact funding rows are never altered;
- imputed funding affects outcome cashflows only;
- imputed funding cannot activate signals;
- imputed results retain `funding_imputed_train_screen_cap`;
- exact, mixed, imputed, conservative, severe, and zero-funding results are separated;
- exactness must be decomposed by period because coverage is calendar-confounded.

Coverage remains weakest in 2023–2024 and strongest in parts of 2025 H2.

Funding cashflow has rarely been the main reason a candidate failed. Exactness remains a serious evidence cap because later exact subsets have often been negative.

---

## 9. Research-engine architecture

Production-standard architecture:

1. Build decision inputs by semantic specification.
2. Use canonical hashes with explicit field allowlists.
3. Separate decision-input caches from outcome caches.
4. Build parent-neutral raw signal tapes.
5. Freeze raw and projected tapes before outcomes.
6. Enforce definition-local actual-exit non-overlap.
7. Freeze accepted candidates before controls.
8. Freeze control keys before control outcomes.
9. Write atomic temp-to-final artifacts and manifests.
10. Resume only hash-valid completed shards.
11. Require verified shard completeness before reduction.
12. Use exactness sentinels after architecture changes.
13. Run long jobs detached with heartbeat and Telegram monitoring.
14. Maintain candidate/control economic-address audits.
15. Track whole-run peak RSS, including control stages.
16. Distinguish fixed rolling walk-forward from expanding-window designs in report labels.

Known current contracts include:

```text
signal_state_contract_v1_20260715
unique economic control-address contract
C2 event exposure / parent / shock episode / pathway hierarchy
shared Kraken funding-imputation contract
```

---

## 10. Completed family decisions before the signal-state campaign

### 10.1 TSMOM v6

Authoritative roots include:

```text
results/rebaseline/phase_kraken_full_tsmom_v6_aggregate_20260707_v1
results/rebaseline/phase_kraken_tsmom_v6_targeted_materialization_controls_stress_20260708_v1_20260708_101819
results/rebaseline/phase_kraken_tsmom_v6_survivor_forensic_decomposition_20260708_v1
results/rebaseline/phase_kraken_tsmom_funding_corrected_reopened_forensics_20260712_v1
```

Final result:

```text
definitions reviewed: 28
positive after top-three removal: 0
positive after top-1% trimming: 0
full leave-one-symbol/month/symbol-month gate: 0
positive in exact and imputed partitions: 0
```

Decision:

```text
TSMOM v6:
    defer_current_translation

Broader TSMOM mechanism:
    candidate_library_only
```

Do not reopen v6 with threshold changes.

### 10.2 A1 liquid leader + compression v1

Authoritative roots include:

```text
results/rebaseline/phase_kraken_a1_compression_contract_manifest_20260708_v1
results/rebaseline/phase_kraken_a1_compression_full_180shard_funding_corrected_20260711_v1_20260711_194859
results/rebaseline/phase_kraken_a1_compression_targeted_materialization_controls_stress_20260712_v1
results/rebaseline/phase_kraken_a1_control_context_adjudication_20260712_v1
```

Facts:

```text
full definitions: 1,440
materialized events: 469,846
targeted definitions: 26
base event rows: 9,178
real control rows: 35,430
candidates beating all four robust symmetric controls: 0
stability-review candidates: 0
```

Decision:

```text
A1/compression v1:
    defer_current_translation

Preserve:
    structure-responsive exit observations
    fresh simpler liquid-entry hypotheses
```

Do not subdivide this sample again.

### 10.3 Prior-high / reclaim v2

Authoritative roots:

```text
results/rebaseline/phase_kraken_prior_high_reclaim_v2_canonical_train_scan_20260712_v1
results/rebaseline/phase_kraken_prior_high_reclaim_v2_full_targeted_materialization_20260712_v1
results/rebaseline/phase_kraken_prior_high_v2_control_matching_repair_20260712_v1
```

The aggregate and materialized economics were positive in selected policies, but no candidate achieved robust control victory, parameter stability, and concentration resistance.

Decision:

```text
prior-high/reclaim v2:
    defer_current_translation

Preserve:
    selected entry/exit policy observations
    prior-high proximity as a possible overlay
```

Do not retune the same event tape.

### 10.4 C2 post-catalyst continuation base

Key roots:

```text
results/rebaseline/phase_kraken_c2_audited_v2_1_ingestion_preflight_20260713_v1
results/rebaseline/phase_kraken_c2_shock_episode_budget_repair_20260713_v1
results/rebaseline/phase_kraken_c2_sample_limited_economic_tranche_20260713_v1
```

The audited C2 database is a source-verified seed, not a closed census. Deterministic ingestion and shock-episode weighting repaired the event identity, but the sample-limited economic translation produced no lead.

Decision:

```text
C2 current translation:
    current_translation_weak

C2 mechanism:
    sample_limited_mechanism_preserved

Database:
    source_verified_seed
```

Do not expand the database merely to rescue the current translation.

---

## 11. Repaired short-family lineages

### 11.1 LFBS repaired lineage

Authoritative repaired roots:

```text
results/rebaseline/phase_kraken_lfbs_signal_state_repaired_screen_20260715_v1
results/rebaseline/phase_kraken_lfbs_021_signal_state_repaired_2023_presample_20260715_v1
results/rebaseline/phase_kraken_lfbs_021_signal_state_repaired_canonical_adjudication_20260715_v1
```

Screen repair:

```text
raw signals: 2,486
parent-policy projections: 2,704
accepted definition trades: 7,910
actual-overlap skips: 184
restored/new addresses: 1,931
unchanged addresses: 1,339
removed legacy addresses: 2,807
```

`lfbs_v1_021` canonical repaired tape:

```text
events: 58
symbols: 30
months: 22
base mean: +0.4076R
conservative mean: +0.3481R
severe mean: +0.1482R
conservative PF: 1.539
```

Period split:

```text
2023: +1.2007R conservative mean
2024: -0.2244R
2025 H1: -0.4059R
2025 H2: +0.0708R
2024–2025 combined: -0.1007R
```

Concentration and controls:

```text
after top-three removal: -0.0019R conservative
severe after top-three: -0.1909R
adequate control classes: 1
fully exact 2025-H2 subset: negative
```

Decision:

```text
LFBS broad translation:
    current_translation_weak

lfbs_v1_021:
    fragile_context_sleeve
    historical-preservation object only
```

The prior contaminated LFBS economic decisions are superseded.

### 11.2 Backside-confirmed blowoff repaired screen

Authoritative root:

```text
results/rebaseline/phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1
```

Repair impact:

```text
legacy accepted trades: 1,103
repaired accepted trades: 1,567
restored addresses: 466
removed addresses: 2
```

Broad family result:

```text
median conservative definition mean: -0.315R
positive conservative definitions: 3/24
median conservative PF: 0.498
```

`bcbs_v1_002`:

```text
events: 15
conservative mean: +0.226R
severe mean: +0.162R
after top-three: -0.181R
loses to extension-without-confirmation structural control
status: fragile_positive_train_screen
```

`bcbs_v1_008`:

```text
moderate 40%/5-day extension
three-bar confirmation
fragile parent context
EMA10 exit

events: 20
symbols: 17
months: 9
conservative mean: +0.287R
severe mean: +0.234R
conservative PF: 1.70
after top-three: +0.008R
severe after top-three: -0.039R
```

Period and control evidence:

```text
2023: +0.287R
2024: +0.573R, only 2 trades
2025 H1: -0.269R
2025 H2: +0.471R

beats five adequate control classes
beats extension-without-confirmation by +0.174R
8 fully exact 2025-H2 events: +0.659R conservative mean
```

Decision:

```text
Backside broad translation:
    current_translation_weak

bcbs_v1_008:
    fragile_context_sleeve
```

It is cleaner than LFBS on controls and exact-era support, but too sparse and tail-dependent for advancement.

### 11.3 RFBS repaired lineage

Authoritative roots:

```text
results/rebaseline/phase_kraken_rfbs_signal_state_repaired_screen_20260715_v1
results/rebaseline/phase_kraken_rfbs_signal_state_repaired_materialization_20260715_v1
results/rebaseline/phase_kraken_cross_family_repair_campaign_closure_20260715_v1
results/rebaseline/phase_kraken_rfbs_010_train_only_stability_review_20260715_v1
```

Repair impact:

```text
legacy accepted trades: 1,024
repaired accepted trades: 1,075
restored addresses: 51
removed addresses: 0
```

`rfbs_v1_010`:

```text
moderate failed bounce
three-bar failure confirmation
broader fragile/countertrend/stress parent policy
fixed 72-hour short

events: 148
symbols: 61
months: 27
base mean: +0.336R
conservative mean: +0.298R
severe mean: +0.170R
conservative PF: 1.76
after top-one: +0.254R
after top-three: +0.199R
worst leave-one-symbol: +0.245R
worst leave-one-month: +0.157R
```

Four adequate positive controls:

```text
rally without completed failure: +0.626R uplift
generic failed breakout: +0.260R
non-rally red candle: +0.225R
same-symbol/same-regime random: +0.054R
```

Period and funding caps:

```text
2023: -0.285R
2024: +0.272R
2025 H1: +0.738R
2025 H2: +0.100R

fully exact events: 3, all losses
mixed events: 2, both losses
fully imputed events: 143, positive overall
```

Train-only stability review:

```text
rolling walk-forward: pass
CPCV: pass
stable controls: pass
007/010 neighbourhood: pass
mechanical integrity: pass
multiplicity adjustment: fail
clustered confidence: fail
```

Detailed stability statistics:

```text
powered folds: 4/6
all four powered folds positive conservative and severe
underpowered folds: 3 and 5 events, both negative

CPCV K=8:
    28 paths
    85.7% conservative-positive
    71.4% severe-positive

conservative PSR: 0.9964
severe PSR: 0.9414
conservative DSR-equivalent probability: 0.7606
severe DSR-equivalent probability: 0.3396
CSCV/PBO: 0.1286
Reality Check p-value: 0.1387
```

Clustered confidence:

```text
month-bootstrap conservative CI: [-0.105, +0.620]
month-bootstrap severe CI: [-0.256, +0.485]
symbol-bootstrap severe lower bound: -0.062
symbol-month severe lower bound: -0.049
```

Decision:

```text
rfbs_v1_010:
    fragile_context_sleeve
    level_4_event_ledger_plus_real_controls
    strongest preserved object
    level_5 not awarded
```

Do not create a 2025-H1 filter, exclude 2023, replace `010` with `007`, or retune the same family.

---

## 12. Repaired and newly tested long families

### 12.1 Delayed flush reclaim long

Blocked original root:

```text
results/rebaseline/phase_kraken_delayed_flush_reclaim_long_screen_20260715_v1
```

Valid repaired root:

```text
results/rebaseline/phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1
```

Repair result:

```text
parent-filtered signals: 603
accepted trades: 1,765
actual-overlap skips: 44
newly admitted definition rows: 64
```

Strongest broad definition `dfrl_v1_011`:

```text
events: 177
conservative mean: +0.100R
severe mean: +0.029R
after top-one: -0.120R
after top-three: -0.218R
2023: -0.406R
2024: +0.461R
2025 H1: -0.104R
2025 H2: -0.183R
```

The repaired signals reduced its apparent edge. The family is a 2024 cluster, not a stable effect.

Decision:

```text
delayed flush reclaim v1:
    current_translation_weak

broader deleveraging-reclaim mechanism:
    preserved for future OI/liquidation-confirmed or capture-based translation
```

### 12.2 Close-confirmed breakout retest long v2

Audit-only blocked root:

```text
results/rebaseline/phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v1
```

Valid economic root:

```text
results/rebaseline/phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v2
```

Mechanical facts:

```text
raw signals: 5,247
parent-projected signals: 8,071
accepted definition trades: 20,131
symbols/months: 114/36
```

Leading `cbrl_v2_020`:

```text
60-bar range
9-bar retest window
both-up parent
EMA10 exit

events: 574
base mean: +0.545R
conservative mean: +0.429R
severe mean: +0.039R
```

It failed:

```text
conservative top-1% trimming: -0.015R
severe after top-one: -0.138R
severe after top-three: -0.249R
severe worst leave-one-month: -0.236R
```

The main falsification result:

```text
incremental conservative value versus immediate breakout:
    negative for 24/24 definitions

cbrl_v2_020 immediate-breakout uplift:
    -0.150R

delayed-entry-without-retest uplift:
    -0.175R
```

Decision:

```text
close-confirmed breakout retest v2:
    current_translation_rejected_only

broader immediate-breakout continuation:
    not rejected by this test
```

Do not retune retest depth, window, parent, or exit on this sample.

### 12.3 Failed-breakdown squeeze-reclaim long

Root:

```text
results/rebaseline/phase_kraken_failed_breakdown_squeeze_reclaim_long_screen_20260716_v1
```

Mechanical facts:

```text
raw signals: 10,427
parent-policy projections: 12,854
accepted trades: 38,083
control rows: 97,292
```

Family summary:

```text
median base definition mean: +0.0659R
median conservative: +0.0145R
median severe: -0.1510R
```

Coherent positive region:

```text
both-up parent
fixed 72-hour exit
failed-breakdown reclaim
```

Preserved sleeves:

```text
fbsr_v1_001
fbsr_v1_007
fbsr_v1_019
fbsr_v1_021
```

Best preserved object `fbsr_v1_019`:

```text
60-bar support
9-bar reclaim
both-up
fixed 72 hours

events: 274
symbols: 78
months: 21
conservative mean: +0.267R
severe mean: +0.089R
conservative PF: 1.50
after top-three: +0.166R
worst leave-one-symbol: +0.239R
worst leave-one-month: +0.067R
```

Period split:

```text
2023: +0.598R
2024: -0.160R
2025 H1: +1.105R
2025 H2: -0.668R
```

Its fully exact events were negative. Incremental evidence for waiting for the reclaim was inadequate and directionally unfavourable in undercovered controls.

Decision:

```text
failed-breakdown squeeze-reclaim broad translation:
    mechanism_preserved_current_translation_weak

001 / 007 / 019 / 021:
    fragile_context_sleeves
```

No same-sample retuning or materialization.

---

## 13. Session and cross-sectional families

### 13.1 Strong-close session-handoff continuation

Root:

```text
results/rebaseline/phase_kraken_strong_close_session_handoff_continuation_20260716_v1
```

Mechanical facts:

```text
raw signals: 3,990
accepted definition rows: 17,403
control rows: 30,032
```

Definition-level family economics:

```text
median base mean: -0.036R
median conservative mean: -0.072R
median severe mean: -0.192R
positive severe definitions: 1/24
```

Leading diagnostic `schc_v1_008`:

```text
08:00 UTC
short
both-down parent
fixed 24-hour exit

events: 439
symbols: 90
months: 26
conservative mean: +0.145R
severe mean: +0.018R
after top-three: +0.090R
```

Period split:

```text
2023: +0.016R
2024: +0.950R
2025 H1: -0.057R
2025 H2: -0.061R
```

No control class reached the frozen 70% coverage threshold.

Preserved observations:

```text
strong close location:
    possible feature overlay

handoff timing:
    possible feature overlay

volume-expansion proxy:
    weak incremental evidence
```

Decision:

```text
strong-close session-handoff family:
    current_translation_weak

schc_v1_008:
    diagnostic only
    not a sleeve
```

### 13.2 Relative-strength breakout versus BTC

Root:

```text
results/rebaseline/phase_kraken_relative_strength_breakout_vs_btc_screen_20260716_v1
```

Mechanical facts:

```text
raw signals: 5,154
accepted definition trades: 20,071
control rows: 50,351
```

Only two definitions were positive under all cost modes:

`rsbb_v1_013`:

```text
USD 60-bar breakout
ALT/BTC 20-bar breakout
both-up parent
fixed 72-hour exit

events: 598
conservative mean: +0.275R
severe mean: +0.089R
```

`rsbb_v1_014`:

```text
same signal region
EMA10 exit

events: 536
conservative mean: +0.487R
severe mean: +0.086R
```

Both are right-tail dependent:

```text
013 conservative top-1% trimmed: -0.011R
014 conservative top-1% trimmed: -0.008R
severe after top-one: negative for both
median event: approximately -1R
```

Funding evidence:

```text
013 fully exact 2025-H2 subset: -0.868R
014 fully exact 2025-H2 subset: -1.076R
014 mixed-funded subset: +5.188R
014 imputed 2025-H2 subset: -0.920R
```

Only one control class was adequate. The central ALT/BTC incremental comparison against USD breakout alone had about 49–50% coverage and inconsistent results.

Decision:

```text
relative-strength breakout vs BTC broad translation:
    mechanism_preserved_current_translation_weak

rsbb_v1_013 / 014:
    fragile_context_sleeves

current claim not established:
    simultaneous ALT/BTC breakout consistently improves a USD breakout
```

Do not materialize or retune this signal region.

---

### 13.3 BTC-led delayed alt diffusion long — H43

Root:

```text
results/rebaseline/phase_kraken_btc_led_delayed_alt_diffusion_long_screen_20260716_v1
```

Mechanical facts:

```text
parent-neutral raw signals: 4,361
parent-projected signals: 5,741
accepted definition trades: 16,803
control rows: 52,677
hard-gate failures: 0
```

Only the `moderate_lag + both_up` signal region was positive under base and conservative costs:

| Definition | Exit | Events | Conservative mean | Severe mean |
|---|---:|---:|---:|---:|
| `h43_v1_001` | 4h | 950 | +0.042R | -0.046R |
| `h43_v1_002` | 8h | 927 | +0.083R | -0.021R |
| `h43_v1_003` | 12h | 914 | +0.093R | -0.026R |

These are not three independent sleeves. Their signal sets have roughly 96–99% Jaccard overlap and differ mainly by holding period.

Best expression `h43_v1_003`:

```text
symbols: 92
months: 25
conservative after top-three: +0.054R
conservative top-1% trimmed: +0.006R
worst leave-one-symbol: +0.078R
worst leave-one-month: +0.053R
```

Period pattern:

```text
2023 conservative: -0.035R
2024: +0.054R
2025 H1: +0.212R
2025 H2: +0.584R
```

Controls:

- BTC impulse timing added positive incremental value relative to lag events without a qualifying BTC impulse.
- BTC leadership beat comparable alt-own-impulse controls.
- Residual-lag selection added only approximately `+0.017R` for the 12h expression and was slightly negative for the 4h and 8h expressions.
- Unconditional-alt-after-BTC-impulse and same-parent-random controls were undercovered.

Funding:

```text
h43_v1_003 fully exact:
    25 events
    conservative mean +0.038R
    severe mean -0.027R
```

Decision:

```text
H43 broad translation:
    mixed-to-weak

preserved object:
    one fragile moderate_lag + both_up signal region
    three exit expressions: 4h / 8h / 12h

preserved feature observation:
    abnormal BTC impulse timing in both-up conditions

not established:
    bottom-residual alts consistently outperform comparable alts
```

Do not retune residual thresholds, add a post-2024 filter, or test another holding period on the same sample.

### 13.4 Asia / U.S. cash-open range resolution — H31/H32

Root:

```text
results/rebaseline/phase_kraken_session_open_range_resolution_20260716_v1
```

Mechanical facts:

```text
parent-neutral raw signals: 34,848
parent-projected selected signals: 48,596
definition-event rows: 145,781
control outcomes: 434,430
hard-gate failures: 0
```

The U.S. calendar was XNYS/DST-aware. The Asia contract used 00:00 UTC Monday–Friday. The first full attempt was OOM-killed; symbol-streamed controls completed successfully.

Economics were uniformly weak:

| Mode | Median definition mean | Positive definitions |
|---|---:|---:|
| Base | -0.075R | 1/24 |
| Conservative | -0.099R | 0/24 |
| Severe | -0.180R | 0/24 |

The least-weak definition, `orr_v1_018`, remained negative conservatively:

```text
U.S. cash open
30-minute range
all-regime
fixed 8h

events: 9,195
base mean: +0.006R
conservative mean: -0.022R
severe mean: -0.119R
```

Long, short, Asia, U.S., 30m, 60m, aligned-parent, all-regime, and all three exit dimensions were negative under conservative assumptions.

Controls:

- generic rolling 4h breakout: adequate 24/24, approximately zero/negative uplift;
- same-symbol/same-parent random: adequate 24/24, negligible uplift;
- non-open range breakout: adequate 12/24, small positive diagnostic;
- four-hour shifted clock: larger apparent uplift but only about 24% coverage;
- pre-open directional control: inadequate and unfavourable.

Decision:

```text
H31/H32 tested translation:
    current_translation_rejected_only

broader session seasonality:
    preserved as execution/risk context only

orr_v1_018:
    not a sleeve
```

Do not search additional clocks or opening-range thresholds on this sample.

---

## 14. Active phase: all-tested-hypotheses external review package

Expected run root:

```text
results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/
```

Status:

```text
active documentation / audit / packaging phase
no economic screen authorized
```

Purpose:

> Create one verified, portable evidence package covering every clean, decision-bearing QLMG hypothesis translation so that experienced traders, independent quants, software engineers, and external analysts can review the project without access to the original backtesting repository or ChatGPT project.

This package must not be a simple collection of old compact ZIPs. It must:

1. resolve authoritative versus superseded roots;
2. preserve the latest repaired lineages;
3. standardize strategy definitions and event/control schemas;
4. include full entries, stops, exits, indicators, timeframes, regime rules, costs, funding, and controls;
5. include MAE/MFE and path data where derivable from frozen events without regenerating signals;
6. include code lineage, shared contracts, relevant source snapshots, tests, environment, hashes, and defect/repair history;
7. independently recompute principal metrics and fail closed on mismatches;
8. exclude protected-period data, secrets, credentials, and mutable telemetry as authority;
9. create trader-readable and engineering-review documentation;
10. provide one convenient core ZIP and one full compressed evidence archive.

Expected main outputs:

```text
qlmg_external_review_core_20260716_v1.zip
qlmg_external_review_full_20260716_v1.tar.zst
```

The external-review package will be the best portable review object, but authority remains hierarchical:

```text
machine-enforced contract and finalized run manifest
    > finalized event/control ledgers and hashes
    > external-review package standardized copies
    > this human-readable brief
    > individual research PDFs and old compact bundles
```

When complete, review:

- whether every tested family is included;
- whether the latest valid repaired roots were selected;
- whether blocked and superseded runs are excluded from current economics but indexed for provenance;
- recomputation mismatches;
- missing MAE/MFE or verification windows;
- event/control totals and schemas;
- code and test completeness;
- secret/redaction scan;
- package size and usability;
- unresolved questions for traders, quants, and engineers.

No new economic family should launch during packaging or before human external-review synthesis.

---

## 15. Current family map

| Family / translation | Current status | Main finding | Next action |
|---|---|---|---|
| TSMOM v6 | `defer_current_translation` | No concentration-robust survivor | External review; materially fresh redesign only |
| A1/compression v1 | `defer_current_translation` | No robust incremental edge over controls | Preserve exit lessons only |
| Prior-high/reclaim v2 | `defer_current_translation` | Positive pockets lacked control/stability support | Preserve overlays/policy observations |
| C2 post-catalyst continuation | `current_translation_weak` | No sample-limited economic lead | Preserve mechanism; no immediate expansion |
| LFBS broad repaired | `current_translation_weak` | Later sample negative | Closed |
| LFBS `021` | `fragile_context_sleeve` | Strong 2023, negative 2024–2025, one adequate control | Freeze; no validation |
| Backside broad repaired | `current_translation_weak` | Narrow positive region only | Closed |
| Backside `008` | `fragile_context_sleeve` | Five controls and exact 2025-H2 support; only 20 events | Freeze |
| RFBS `010` | `fragile_context_sleeve`, level 4 | Strongest preserved object; failed multiplicity and clustered confidence | Freeze; no more RFBS work |
| Delayed flush reclaim | `current_translation_weak` | 2024 cluster, tail-dependent | Closed current translation |
| Breakout retest v2 | `current_translation_rejected_only` | Retest waiting did not beat immediate breakout | Closed current translation |
| Failed-breakdown squeeze reclaim | `mechanism_preserved_current_translation_weak` | Both-up fixed-hold region, unstable periods | Preserve four sleeves; no materialization |
| Strong-close handoff | `current_translation_weak` | Close location/handoff may be overlays; no adequate controls | Closed current translation |
| Relative-strength breakout vs BTC | `mechanism_preserved_current_translation_weak` | Two fragile sleeves; incremental relative condition unresolved | Freeze region; no materialization |
| BTC-led delayed alt diffusion H43 | `fragile_context_sleeve` region | BTC impulse timing supported; lag rank weak; severe costs fail | Preserve one moderate-lag/both-up region |
| Session-open range resolution H31/H32 | `current_translation_rejected_only` | Every definition negative under conservative costs | Closed tested translation |
| Sector/theme ignition | `mechanism_preserved` | Needs PIT sector/co-movement object | External review / data-readiness decision |
| Small-cap reversal | `capture_sensitive` | Strategically relevant, depth-sensitive | Forward capture or slower redesign |
| Funding/OI families | `data_readiness_unresolved` | Exact decision-time coverage must be audited before rankable testing | Remaining-hypothesis/data-readiness review |
| Listing/event shorts | `capture_sensitive` | Event replay and lifecycle sensitive | Candidate library / capture |
| External-review evidence package | `active_documentation_phase` | Consolidating all tested evidence and engineering lineage | Complete, verify, distribute, synthesize |

---

## 16. Main research lessons

### 16.1 Candidate identity is part of the hypothesis

Premature maximum-hold preblocking materially altered LFBS, Backside, RFBS, and delayed-flush trade tapes. Candidate generation and actual position state must be separated.

### 16.2 Positive pooled means are not enough

Repeated failures were driven by:

- one favourable year or half-year;
- top-event dependence;
- mixed-funded calendar clusters;
- weak or undercovered controls;
- multiplicity-adjusted uncertainty.

Always inspect definition-level and period-level distributions.

### 16.3 Severe stress is useful but not sufficient by itself

A strategy is not rejected only because severe stress is negative if base and conservative evidence are broad and stable. Conversely, a tiny positive severe mean does not rescue winner dependence or negative structural controls.

### 16.4 Exact funding is an evidence partition

Funding cashflow has usually been small relative to alpha. Exactness still matters because exact-funded subsets have repeatedly been later-period and negative.

Do not use this to mine an exact-period exclusion. Keep the cap explicit.

### 16.5 Positive skew is valid only if the winning tail recurs

Many candidates have median event near `-1R` and positive mean from a small right tail. Such strategies require stronger temporal, control, and clustered-confidence evidence than symmetric-payoff strategies.

### 16.6 Controls must address the claimed mechanism

A candidate beating random entries does not prove its exact setup component adds value.

Examples:

- breakout retest failed against immediate/generic breakout controls;
- relative-strength breakout did not establish the ALT/BTC increment;
- failed-breakdown reclaim did not establish the timing value of waiting for reclaim;
- Backside `008` did establish positive increment against extension without confirmation.

### 16.7 Control risk denominators need prospective comparability

Tiny valid control stop distances can create extreme R values. Preserve official controls, but future contracts should prospectively match candidate/control `risk_to_daily_atr` and enforce common frozen bands.

Do not retroactively floor or winsorize decision controls.

### 16.8 Evidence level and profitability are separate

A weak or rejected translation can have level-4 evidence if it has a clean event ledger and real controls. Do not use custom evidence labels that mix evidence type with strategy quality.

### 16.9 A weak translation is not always a family rejection

Preserve coherent mechanisms when the current rule set fails. Do not repeatedly retune the same sample.

### 16.10 Walk-forward labels must describe the actual design

A report labelled fixed `18-month train / 3-month test / 3-month step` must move the train start. If the train start remains fixed, label it expanding-window.

---

## 17. Forward capture and live-side status

The Kraken forward-capture sidecar remains useful.

Last documented state around 2026-07-08:

```text
local root: /opt/qlmg_live_capture/kraken
snapshot size: approximately 25.955 GiB
files: approximately 175,800
coverage: approximately 2026-07-01 through 2026-07-08 UTC
broad trade/ticker coverage: approximately 300 Kraken perp symbols
heavy-book symbols:
    PF_XBTUSD
    PF_ETHUSD
    PF_SOLUSD
    PF_XRPUSD
    PF_DOGEUSD
    PF_ADAUSD
    PF_AVAXUSD
    PF_LINKUSD
forced-flow rows: approximately 38,200
```

Operational requirements:

- free-space guard;
- automatic pause below emergency threshold;
- bounded chunk rotation;
- upload sealed chunks only;
- copy/check/delete or verified move;
- bounded logs;
- Telegram capture/disk/upload alerts.

Use capture for:

- spread and slippage calibration;
- top-of-book sanity;
- predicted versus realized funding;
- OI and liquidation-field validation;
- prospective confirmation of fragile sleeves;
- future small real-money canaries only after separate approval.

Do not use post-2026 capture to select strategies on the sealed historical program.

---

## 18. Workflow for AI agents

### 18.1 General workflow

1. Maintain a detailed human/ChatGPT plan.
2. Convert it into one focused agent prompt.
3. Run one bounded phase.
4. Review the report and compact bundle.
5. Continue, repair, preserve, or stop under frozen gates.

### 18.2 Prompt standard

Prompts should usually contain:

- one objective;
- one run root;
- exact reference roots;
- frozen definitions and scope;
- forbidden actions;
- hard gates;
- required outputs;
- exact final decision vocabulary.

Preferred length:

```text
approximately 400–900 words
```

Detailed plans may be longer. Avoid monolithic implementation prompts.

### 18.3 Long-running jobs

- detached `tmux` or equivalent;
- Telegram start/progress/failure/completion;
- heartbeat and stall detection;
- whole-run RSS tracking;
- deterministic resume;
- preserve completed artifacts;
- no frequent interactive polling.

### 18.4 Review checklist

For every agent report, verify:

- exact code changes;
- frozen manifest unchanged;
- signal-state contract version;
- raw tape frozen before outcomes;
- controls frozen before outcomes;
- candidate/control duplicate addresses zero;
- PIT parent nesting;
- boundary reconciliation and artificial exits;
- funding joins and signal-gate prohibition;
- protected-period and leakage audits;
- definition-level economics;
- parameter-neighbourhood breadth;
- top-one/top-three/top-1% sensitivity;
- symbol/month/period concentration;
- exact/mixed/imputed funding;
- control coverage and mechanism relevance;
- actual unmatched-only complements;
- evidence level achieved;
- prohibited work launched or not.

---

## 19. Candidate-library requirements

Every alpha run must update a full-schema candidate library.

Required fields include:

```text
family
variant_id
mechanism
direction
required_data_tier
current_data_tier
evidence_level
standalone_status
portfolio_sleeve_status
rare_regime_status
feature_overlay_status
event_count
active_months
active_symbols
dominant_month_share
dominant_symbol_share
mark_available
funding_exact
funding_imputed
mark_proxy_used
funding_proxy_used
data_tier_cap_reason
label_cap_reason
reason_preserved_or_discarded
next_action
```

Lineage fields include:

```text
definition_lane
entry_policy_hash
exit_policy_hash
universe_policy_hash
rank_policy_hash
regime_policy_hash
funding_policy_hash
parameter_vector_hash
selected_event_key_hash
candidate_economic_address_hash
outcome_cache_hash
canonical_contract_version
signal_state_contract_version
active_caps
```

Current important library rows:

```text
rfbs_v1_010:
    evidence_level = level_4_event_ledger_plus_real_controls
    economic_status = fragile_context_sleeve
    next_action = frozen preservation only

lfbs_v1_021:
    evidence_level = level_4_event_ledger_plus_real_controls
    economic_status = fragile_context_sleeve

bcbs_v1_008:
    evidence_level = level_4_event_ledger_plus_real_controls
    economic_status = fragile_context_sleeve

fbsr_v1_001 / 007 / 019 / 021:
    evidence_level = level_4_event_ledger_plus_real_controls
    economic_status = fragile_context_sleeve

rsbb_v1_013 / 014:
    evidence_level = level_4_event_ledger_plus_real_controls
    economic_status = fragile_context_sleeve

strong-close location and handoff timing:
    feature_overlay_candidate
```

Do not create a portfolio from these sleeves until individual evidence improves materially.

---

## 20. Portable project-file and separate-folder map

### 20.1 How an outside reviewer should receive the project

A reviewer may receive a folder rather than access to this ChatGPT project or the backtesting machine. Use this brief as the first human-readable navigation document.

Recommended portable layout:

```text
QLMG_REVIEW_FOLDER/
├── 00_READ_FIRST/
│   ├── QLMG_Project_Master_Continuity_Brief_2026-07-16_rev7.md
│   ├── testmanual.txt
│   └── QLMG_Project_File_Index_2026-07-16_rev7.csv
├── 01_PLANS_AND_PROVENANCE/
├── 02_CORE_QLMG_RESEARCH/
├── 03_RES_RESEARCH/
├── 04_RES2_HYPOTHESES/
├── 05_MOV_VENUE_RESEARCH/
├── 06_CATALYST_AND_SECTOR_DATA/
├── 07_HYPOTHESIS_LIBRARY/
├── 08_COMPACT_BUNDLES/
│   ├── current/
│   └── provenance_or_superseded/
└── 09_EXTERNAL_REVIEW_PACKAGE/
```

The files may also be supplied in one flat folder. In that case, exact filenames and the companion CSV index are the lookup mechanism.

### 20.2 Read-first authority files

| File | Role | Status |
|---|---|---|
| `QLMG_Project_Master_Continuity_Brief_2026-07-16_rev7.md` | Current comprehensive human-readable state, decisions, file map, and workflow | Current authority for human context |
| `testmanual.txt` | Evidence policy, evidence levels, hard rejection vs preservation, no-vendor rules | Active policy authority |
| `QLMG_Project_File_Index_2026-07-16_rev7.csv` | Machine-readable inventory, hashes, categories, canonical status, and duplicate mapping | Current portable index |
| `QLMG_Project_Read_First_SHA256_2026-07-16_rev7.txt` | SHA-256 checksums for the rev7 brief and companion file index | Read-first integrity sidecar |
| `QLMG_Project_Master_Continuity_Brief_2026-07-16_rev6.md` | Previous brief | Superseded; provenance only |
| `QLMG_Project_Master_Continuity_Brief_2026-07-13_rev5(1).md` | Earlier state before repair campaign completion | Superseded; provenance only |
| `QLMG_Project_Master_Continuity_Brief_2026-07-12_rev4(1).md` | Earlier state | Superseded; provenance only |

### 20.3 Planning, methodology, and project-history files

| File | Contents / use |
|---|---|
| `Initial-conversion-plan.txt` | Initial QLMG conversion and strategy research plan. Use for original hypotheses, not current decisions. |
| `Project Analysis and Planning.txt` | Earlier project-state and planning notes. Provenance and context only. |
| `Efficient Alpha Research for Python Systematic Trading.pdf` | Research-funnel, compute, multiplicity, deduplication, validation, and engineering guidance. |
| `QLMG_Hypothesis_Library_2026-07-01.xlsx` | Structured hypothesis catalogue and candidate inventory source. Current decisions must be reconciled against later run artifacts. |

### 20.4 Core QLMG research reports

| File | Main subject |
|---|---|
| `QLMG Rulebook for Kristjan Qullamaggie Kullamägi Public Trading Setups.pdf` | First-party Qullamaggie breakout, episodic-pivot, parabolic, stop, exit, and risk rules. |
| `QLMG Robust Backtest Protocol for Qullamaggie-Inspired Crypto Perpetual Strategies.pdf` | Full perp-specific backtest and validation architecture. |
| `QLMG Crypto Perpetual Futures vs Stocks for Systematic Strategy Testing.pdf` | Why perpetuals require mark, index, funding, lifecycle, venue-specific liquidation, and fee modeling. |
| `QLMG Cryptocurrency Momentum, Reversal, Perpetuals, Open Interest, and Liquidation Cascades.pdf` | Momentum/reversal split by liquidity and derivatives-state research. |
| `QLMG Cryptocurrency Momentum, Reversal, Funding, Open Interest, and Liquidation Cascades.pdf` | Liquid-crypto continuation, funding/OI, and liquidation evidence. |
| `QLMG Historical Crypto Perpetual Setups From 2020 Through Mid 2026.pdf` | Historical setup illustrations and priors; not a decisive backtest. |
| `QLMG Taxonomy of Crypto Catalysts Analogous to Stock Episodic Pivots.pdf` | Catalyst mechanisms and historical examples. |
| `QLMG Simple Alpha Hypotheses for Bybit USDT Perpetuals.pdf` | Original simple strategy ideas, regime framing, ORB, weak-asset fades, and venue-specific hypotheses. |

### 20.5 RES research files

| File | Main subject |
|---|---|
| `RES Liquid Universe Continuation Strategies for Crypto Perpetual Futures.pdf` | Evidence ranking for TSMOM, liquid continuation, prior-high, retest, and sector overlays. |
| `RES Point-in-Time Regime Framework for Crypto Perpetual Futures.pdf` | PIT eligibility, parent weather, breadth, derivatives, liquidity, session, and lifecycle layers. |
| `RES Designing Less Depth-Sensitive Crypto Perp Tests.pdf` | Which families are honestly bar-testable and conservative Tier-1 execution assumptions. |
| `RES Research Base for Liquid-Sector Episodic Pivots and Post-Catalyst Continuation.pdf` | Sector ignition and catalyst continuation mechanisms, taxonomy, and controls. |

### 20.6 RES2 hypothesis and practitioner files

| File | Main subject |
|---|---|
| `RES2 Crypto Perpetual Futures Alpha Hypothesis Catalogue.pdf` | Large canonical hypothesis catalogue with IDs, requirements, controls, and falsification rules. |
| `RES2 Public-Data Crypto Perpetual Alpha Hypotheses.pdf` | Public-data and forward-capture hypotheses, especially derivatives and microstructure states. |
| `RES2 Regime Map for Crypto Perpetual Alpha.pdf` | Sleeve activation by parent, crowding, purge, and liquidity regime. |
| `RES2 Crypto Perpetual Practitioner Alpha Map.pdf` | Practitioner claims converted into testable mechanisms and controls. |
| `RES2 No-Vendor Roadmap for Testing Crypto Perpetual Alpha Hypotheses.pdf` | Data-tier and no-vendor routing by family. |

### 20.7 Venue and migration files

| File | Main subject |
|---|---|
| `MOV Contingency Venues for EU QLMG if Bybit Global Perps Become Unavailable.pdf` | Regulated and onchain contingency venue shortlist. |
| `MOV EU Venue Access Report for Crypto Perpetual Futures.pdf` | EU/EEA product-access and regulatory comparison. |
| `MOV Kraken Derivatives Versus Bybit for QLMG Perpetual Strategies.pdf` | Breadth, liquidity, product, and family-fit comparison. |
| `MOV Kraken derivatives data availability and acquisition report for QLMG.pdf` | Kraken public historical data coverage and gaps. |
| `MOV Kraken Live Trading Migration Blueprint for a Bybit USDT Perpetual Bot.pdf` | Kraken connector, account, collateral, order, margin, and liquidation migration design. |

These venue reports may contain time-sensitive statements as of their publication date. Re-verify before operational use.

### 20.8 Catalyst and sector files

| File | Main subject |
|---|---|
| `Post-Catalyst Continuation Base Catalyst Database.pdf` | Initial 2020–2025 catalyst database and schema. |
| `Point In Time Sector Seeds for Crypto Perpetual Theme Ignition.pdf` | PIT sector-seed methodology, taxonomy limits, and starter mappings. |
| `catdb(1).md` | Audited C2 Catalyst Database v2.1 consolidated register. This exact filename is present; do not assume `catdb.md` exists. |

### 20.9 Compact-bundle guidance

Individual compact bundles are convenient review extracts. They are not interchangeable with full run roots, and older bundles can be superseded.

Current/latest review bundles in the supplied project folder:

| Filename | Current lineage / purpose |
|---|---|
| `compact_review_bundle lfbs.zip` | Repaired canonical LFBS `021` adjudication extract |
| `compact_review_bundle backside blowoff.zip` | Repaired Backside screen |
| `compact_review_bundle screen.zip` | Repaired RFBS screen |
| `compact_review_bundle materialization.zip` | Repaired RFBS materialization/control adjudication |
| `compact_review_bundle repair.zip` | Cross-family repair closure and corrected RFBS formal-candidate decision |
| `compact_review_bundle stability.zip` | RFBS `010` train-only stability review |
| `compact_review_bundle delay.zip` | Repaired delayed-flush reclaim screen |
| `compact_review_bundle breakout.zip` | Close-confirmed breakout-retest v2 |
| `compact_review_bundle breakdown.zip` | Failed-breakdown squeeze-reclaim |
| `compact_review_bundle strong close.zip` | Strong-close session handoff |
| `compact_review_bundle relative st.zip` | Relative-strength breakout versus BTC |
| `compact_review_bundle btcled.zip` | BTC-led delayed-alt diffusion H43 |
| `compact_review_bundle session open.zip` | Asia / U.S. cash-open range resolution H31/H32 |
| `compact_review_bundle repairt.zip` | Cross-family repair campaign registry/contract phase |
| `compact_review_bundle closed mech.zip` | Source-only preblock audit that paused the first breakout-retest attempt |

Superseded or diagnostic bundles retained only for provenance:

| Filename | Why not current |
|---|---|
| `compact_review_bundle bo.zip` | Original pre-repair Backside screen |
| `compact_review_bundle riskoff.zip` | Original pre-repair RFBS screen |
| `compact_review_bundle rfbs.zip` | Original pre-repair RFBS materialization |
| `compact_review_bundle fl(1).zip` | Blocked delayed-flush diagnostic before signal-state repair |

Some files also exist with `%20` in their filenames. These are byte-identical duplicate copies of the corresponding space-named ZIP and add no evidence. `QLMG_Project_Master_Continuity_Brief_2026-07-16_rev6(1).md` is also byte-identical to the non-`(1)` rev6 file.

### 20.10 Files not currently represented by individual compact bundles

The portable source folder does not contain individual current compact bundles for every older family, including some TSMOM, A1/compression, prior-high/reclaim, and C2 roots. Their authoritative results live on the backtesting machine and are indexed by run root in the next section.

The external-review evidence package is being created specifically to close this portability gap. When available, place it under:

```text
09_EXTERNAL_REVIEW_PACKAGE/
```

and treat its authoritative-root registry and recomputation report as the preferred portable entry point for all tested families.

---

## 21. Major authoritative run-root index

These are repository-relative paths on the backtesting machine. A separate source folder may contain compact bundles rather than these full trees.

### 21.1 Evidence and shared infrastructure

```text
results/rebaseline/phase_qlmg_mechanical_qa_evidence_contract_20260630_v1_20260630_074328
results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1
results/rebaseline/phase_kraken_cross_family_signal_state_repair_campaign_20260715_v1
results/rebaseline/phase_kraken_cross_family_repair_campaign_closure_20260715_v1
```

### 21.2 TSMOM

```text
results/rebaseline/phase_kraken_full_tsmom_v6_aggregate_20260707_v1
results/rebaseline/phase_kraken_tsmom_v6_targeted_materialization_controls_stress_20260708_v1_20260708_101819
results/rebaseline/phase_kraken_tsmom_v6_survivor_forensic_decomposition_20260708_v1
results/rebaseline/phase_kraken_tsmom_funding_corrected_reopened_forensics_20260712_v1
```

### 21.3 A1/compression

```text
results/rebaseline/phase_kraken_a1_compression_contract_manifest_20260708_v1
results/rebaseline/phase_kraken_a1_compression_full_180shard_funding_corrected_20260711_v1_20260711_194859
results/rebaseline/phase_kraken_a1_compression_targeted_materialization_controls_stress_20260712_v1
results/rebaseline/phase_kraken_a1_control_context_adjudication_20260712_v1
```

### 21.4 Prior-high/reclaim

```text
results/rebaseline/phase_kraken_prior_high_reclaim_v2_canonical_train_scan_20260712_v1
results/rebaseline/phase_kraken_prior_high_reclaim_v2_full_targeted_materialization_20260712_v1
results/rebaseline/phase_kraken_prior_high_v2_control_matching_repair_20260712_v1
```

### 21.5 C2 catalyst

```text
results/rebaseline/phase_kraken_c2_audited_v2_1_ingestion_preflight_20260713_v1
results/rebaseline/phase_kraken_c2_shock_episode_budget_repair_20260713_v1
results/rebaseline/phase_kraken_c2_sample_limited_economic_tranche_20260713_v1
```

### 21.6 Repaired LFBS

```text
results/rebaseline/phase_kraken_lfbs_signal_state_repaired_screen_20260715_v1
results/rebaseline/phase_kraken_lfbs_021_signal_state_repaired_2023_presample_20260715_v1
results/rebaseline/phase_kraken_lfbs_021_signal_state_repaired_canonical_adjudication_20260715_v1
```

### 21.7 Repaired Backside

```text
results/rebaseline/phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1
```

### 21.8 Repaired RFBS and stability review

```text
results/rebaseline/phase_kraken_rfbs_signal_state_repaired_screen_20260715_v1
results/rebaseline/phase_kraken_rfbs_signal_state_repaired_materialization_20260715_v1
results/rebaseline/phase_kraken_cross_family_repair_campaign_closure_20260715_v1
results/rebaseline/phase_kraken_rfbs_010_train_only_stability_review_20260715_v1
```

### 21.9 Delayed flush reclaim

```text
results/rebaseline/phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1
```

### 21.10 Close-confirmed breakout retest

```text
results/rebaseline/phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v2
```

### 21.11 Failed-breakdown squeeze reclaim

```text
results/rebaseline/phase_kraken_failed_breakdown_squeeze_reclaim_long_screen_20260716_v1
```

### 21.12 Strong-close session handoff

```text
results/rebaseline/phase_kraken_strong_close_session_handoff_continuation_20260716_v1
```

### 21.13 Relative-strength breakout versus BTC

```text
results/rebaseline/phase_kraken_relative_strength_breakout_vs_btc_screen_20260716_v1
```

### 21.14 BTC-led delayed alt diffusion

```text
results/rebaseline/phase_kraken_btc_led_delayed_alt_diffusion_long_screen_20260716_v1
```

### 21.15 Asia / U.S. cash-open range resolution

```text
results/rebaseline/phase_kraken_session_open_range_resolution_20260716_v1
```

### 21.16 External-review evidence package — expected active root

```text
results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1
```

### 21.17 Superseded or diagnostic-only roots

Preserve, but do not use for active ranking:

```text
results/rebaseline/phase_kraken_liquid_failed_breakout_short_screen_20260713_v1
results/rebaseline/phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1
results/rebaseline/phase_kraken_lfbs_021_canonical_episode_adjudication_20260713_v1
results/rebaseline/phase_kraken_backside_blowoff_short_screen_20260713_v1
results/rebaseline/phase_kraken_riskoff_failed_bounce_short_screen_20260714_v1
results/rebaseline/phase_kraken_rfbs_control_overlap_materialization_20260714_v1
results/rebaseline/phase_kraken_delayed_flush_reclaim_long_screen_20260715_v1
results/rebaseline/phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v1
```

These roots remain provenance. Their economics were superseded, mechanically blocked, or never launched.

---

## 22. Things not to do

- Do not treat old Donch results as active evidence.
- Do not launch another hypothesis while the external-review package is incomplete.
- Do not use individual old compact bundles without checking the supersession map.
- Do not reopen TSMOM v6 by threshold changes.
- Do not subdivide A1/compression v1 again.
- Do not retune prior-high v2 on its existing event tape.
- Do not retune C2 on the same shock episodes.
- Do not create a 2023-style LFBS filter.
- Do not retune Backside `008` or RFBS `010` around observed weak periods.
- Do not switch RFBS `010` to `007` after the stability review.
- Do not manufacture a portfolio from fragile sleeves.
- Do not retune breakout-retest after its structural-control failure.
- Do not convert undercovered reclaim, lag, immediate-entry, shifted-clock, or unconditional controls into same-sample strategies.
- Do not retune failed-breakdown, strong-close, RSBB, H43, or session-open contracts around favourable 2024/2025 clusters.
- Do not add a post-2024 H43 activation filter or a new residual-rank threshold.
- Do not search more session-open clocks or OR lengths after H31/H32 failed.
- Do not lower control adequacy after seeing results.
- Do not count duplicate control addresses independently.
- Do not use maximum-hold preblocking before actual exits.
- Do not close trades at sample endpoints.
- Do not use imputed funding as a signal.
- Do not use post-2026 capture for historical strategy selection.
- Do not open the holdout because a pooled mean, powered-fold median, or CPCV median is positive.
- Do not recommend live trading.
- Do not begin broad data archaeology unless it could change a frozen decision.
- Do not treat the external-review package as permission to re-optimize historical winners.

---

## 23. Things to prioritize

1. Complete the all-tested-hypotheses external-review evidence package.
2. Verify authoritative roots, supersession, package hashes, metric recomputation, protected-period exclusion, and secret redaction.
3. Ensure every tested family has trader-readable, quant-method, and engineering-lineage documentation.
4. Ensure the full package contains decision-bearing event/control data, MAE/MFE, paths, coverage, costs, funding partitions, and code/test evidence.
5. Distribute the package to outside traders, quant analysts, and engineers.
6. Synthesize disagreements and unresolved questions rather than taking a simple vote.
7. Perform a bounded remaining-hypothesis and data-readiness review after external analysis.
8. Enforce `signal_state_contract_v1_20260715` in every future event runner.
9. Prefer materially independent hypotheses over repeated threshold refinements.
10. Report definition-level distributions, not only row-weighted pooled summaries.
11. Separate exact, mixed, and imputed funding by period.
12. Require controls that address the exact claimed mechanism.
13. Preserve fragile sleeves without treating them as executable strategies.
14. Maintain the full candidate library, portable file index, and authoritative run-root index.
15. Keep the final holdout sealed.
16. Use forward capture for execution and microstructure questions rather than pretending bar data answers them.

---

## 24. Starting instruction for a new ChatGPT project chat

```text
We are continuing the Donch / QLMG Crypto Perpetual Research project in a new chat.

First, find and read:

`QLMG_Project_Master_Continuity_Brief_2026-07-16_rev7.md`

Also read:

- `testmanual.txt`
- `QLMG_Project_File_Index_2026-07-16_rev7.csv`

Treat rev7 as the current single human-readable truth-source. Treat machine-enforced contracts, finalized manifests, and event/control hashes as execution authority.

Important current state:

- Old Donch V3/S1 is legacy.
- No live trading is authorized.
- Paid historical vendor data is prohibited.
- The final strategy-selection holdout begins at `2026-01-01T00:00:00Z` and remains sealed.
- The cross-family signal-state repair campaign is complete.
- Rankable event families must comply with `signal_state_contract_v1_20260715`.
- No strategy has reached level 5.
- `rfbs_v1_010` is the strongest preserved object but remains only a level-4 fragile sleeve after failing multiplicity and clustered-confidence gates.
- Other preserved objects include `lfbs_v1_021`, `bcbs_v1_008`, selected failed-breakdown sleeves, `rsbb_v1_013/014`, and one highly overlapping H43 moderate-lag/both-up region.
- Close-confirmed breakout retest, strong-close handoff, and H31/H32 session-open range resolution are closed current translations.
- The current phase is documentation and external review, not another strategy screen.
- The backtesting agent is expected to create:
  `results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1/`
- Expected portable outputs are:
  `qlmg_external_review_core_20260716_v1.zip`
  `qlmg_external_review_full_20260716_v1.tar.zst`

When I provide the package report and files:

1. Verify authoritative-root resolution and supersession.
2. Check that every clean tested family is included and protected-period data are absent.
3. Recompute key statistics from standardized ledgers and inspect mismatches.
4. Review entries, stops, exits, indicators, timeframes, regimes, controls, funding, MAE/MFE, paths, concentration, periods, and evidence levels.
5. Review code lineage, contracts, tests, environment, hashes, defect repairs, memory/runtime, and deterministic replay evidence.
6. Separate trader, quant, and engineering concerns.
7. Do not launch or recommend a new economic family until the external review is synthesized.
8. After review, propose a bounded remaining-hypothesis/data-readiness phase or a reason to pause historical screening.
```

---

## 25. Provenance and uncertainty

This brief combines:

- `QLMG_Project_Master_Continuity_Brief_2026-07-16_rev6.md`;
- `testmanual.txt`;
- the audited C2 catalyst database report;
- finalized reports and compact bundles for the signal-state repair campaign;
- repaired LFBS, Backside, RFBS, and delayed-flush lineages;
- RFBS train-only stability review;
- close-confirmed breakout-retest v2;
- failed-breakdown squeeze-reclaim;
- strong-close session-handoff continuation;
- relative-strength breakout versus BTC;
- BTC-led delayed-alt-diffusion H43;
- Asia / U.S. cash-open range resolution H31/H32;
- the commissioned external-review evidence-package contract;
- the current project-source file inventory and byte-identical duplicate audit.

The production repositories and full run trees remain on separate machines. Finalized manifests, full event/control ledgers, hashes, and compact bundles are the reviewable evidence objects. This brief cannot independently reproduce every run; it records the reconciled human-readable state.

The external-review evidence package is not yet complete at the time of this revision. Section 14 records its contract and expected role, not a completed result.

Venue-access and exchange-product statements in MOV files are time-sensitive and must be re-verified before operational use.
