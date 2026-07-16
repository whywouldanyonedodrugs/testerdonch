# QLMG A1 + Compression Sweep Plan

Date: 2026-07-08  
Scope: A1 Liquid Leader Continuation Breakout with H06/H12/H13 compression / volatility-contraction subfamily on Kraken-style data  
Status: planning and prompt sequence. This is not validation, final-holdout evidence, live-readiness, or permission to trade.

## 1. Decision frame

The current TSMOM v6 line should be closed as a non-advancing current translation, but not treated as a family-level rejection. Its infrastructure work is valuable and should be reused. The next active research family should be a coupled liquid-continuation morphology program:

- primary family: A1 / H01 liquid leader continuation breakout;
- coupled subfamily: H06/H12/H13 compression, volatility contraction, and flat-range escape;
- prior-high/reclaim: filter/context only in this cycle, not standalone;
- TSMOM: candidate-library only for later redesign, not part of this sweep.

The reason to couple A1 and compression is that the economic setup is one morphology: a liquid leader makes a prior impulse, builds a controlled base or volatility contraction, then breaks out on a close-confirmed trigger under favorable parent/funding/liquidity context. Compression alone is allowed as a sublane, but it should be compared against A1-only and A1-plus-compression, not treated as a disconnected family.

## 2. Lessons from the TSMOM cycle

The TSMOM cycle produced reusable engineering lessons:

1. Build semantic caches by unique spec, not by candidate row.
2. Separate decision-input caches from outcome caches.
3. Freeze selected event keys before reading outcomes.
4. Use exactness sentinels after every major architecture change.
5. Materialized survivors must face leave-one-symbol, leave-one-month, leave-one-symbol-month, top-event, funding, and slippage stress before stability review.
6. Do not trust aggregate-only nonfutile rows.
7. Do not let a long job run interactively; launch detached with Telegram and heartbeat.
8. Every long stage must be resumable by deterministic shard manifests and semantic cache hashes.
9. Do not promote candidates that depend on proxy funding, one symbol-month, or zero-slippage assumptions.

These gates become mandatory for A1/compression.

## 3. Data policy and capture inventory implications

### 3.1 Historical train data

Historical rankable train screens should use current Kraken K0/K1-style data:

- trade candles / OHLCV;
- mark where available;
- funding or funding proxy where available;
- lifecycle and instrument metadata;
- PIT universe reconstruction;
- conservative all-taker fees and slippage stress.

Historical OI is not confirmed complete locally. Therefore:

- OI vetoes must not be a hard requirement for rankable historical A1/compression rows;
- OI-dependent definitions are either omitted, marked `oi_unavailable_cap`, or moved to a forward-capture sidecar;
- funding and parent/liquidity gates can be rankable if PIT-safe.

### 3.2 Forward capture data

The live capture inventory is good enough for execution calibration and sidecar research, but not a substitute for multi-year historical vendor data.

Useful captured data:

- raw ticker-lite for broad symbol state;
- raw public trades for broad symbols;
- heavy-symbol book and full ticker for BTC/ETH/SOL/XRP/DOGE/ADA/AVAX/LINK;
- normalized ticker_1m with mark/index/bid/ask/openInterest/fundingRate/fundingRatePrediction;
- normalized book_top for spread sanity;
- normalized forced-flow trades with liquidation/assignment/unwind-like public trade types.

Use capture data for:

- slippage/spread stress calibration by liquidity tier;
- sanity checks for fundingRatePrediction/OI fields;
- future sidecar capture-calibration reports;
- not as a blocker if missing for historical years.

Do not require paid vendor data. If exact depth/liquidation data is missing historically, keep the historical scan conservative and cap the label.

## 4. Family contract

### 4.1 Common decision semantics

All rankable A1/compression definitions must be close-confirmed and next-bar executable.

Allowed decision bars:

- 4h close;
- daily close;
- optional 8h close if already supported by cache/calendar.

Execution:

- next available 5m or next-bar open after decision close;
- all-taker base assumption;
- no touch-fill or same-bar target-first assumptions;
- current decision bar excluded from prior high, range top, compression window, and impulse/base reference calculations unless explicitly used only as the close-confirmed breakout bar.

Shorts:

- include only a small diagnostic short lane if the engine already supports it cleanly;
- short continuation breakdowns require parent stress/downtrend regime and close-confirmed breakdown;
- do not let short variants dominate the first tranche.

### 4.2 Universe and ranking

Universe policies:

1. `tier_ab_liquid_strict`: liquid majors and upper-liquid alts.
2. `tier_ab_liquid_broad`: broader but still tradable liquid alts.
3. `majors_only`: BTC/ETH/SOL/XRP/DOGE/ADA/AVAX/LINK-style reference set if PIT-live.
4. `kraken_listed_liquid_tail_capped`: capped diagnostic, lower priority.

Ranking options:

- 20d / 60d / 90d relative strength;
- 20d / 60d risk-adjusted return;
- prior impulse rank;
- liquidity rank as tie-break, not alpha by itself;
- prior-high proximity as filter/context, not standalone entry system.

Top-N options:

- Top 3, Top 5, Top 10, Top 20.

Tie handling:

- deterministic seeded hash or PIT liquidity tie-break;
- never alphabetical, first-N, or current-live order.

### 4.3 Regime/context design

Do not expect the setup to work all the time. Each entry lane should be tested under context variants:

Parent regime gates:

1. none / baseline diagnostic;
2. BTC above rising 50d or 60d trend;
3. BTC and ETH both above rising medium trend;
4. rotation/mixed regime allowed if liquid breadth confirms;
5. fragile/stress regime disables long continuation or reduces to diagnostic.

Funding gates:

1. none baseline;
2. exclude upper 20% funding percentile;
3. exclude upper 10% funding percentile;
4. exclude extreme persistent positive funding;
5. exact/proxy funding flag must be carried into event rows.

Breadth/liquidity context:

- PIT eligible breadth above moving-average threshold;
- liquid-universe breadth improving;
- stale-bar / bad-wick / data gap exclusions;
- broad spread/depth only if historical public data supports it, otherwise use conservative cost stress and cap.

### 4.4 Entry lanes

The manifest should use structured lanes rather than a broad unlabelled grid.

#### Lane A: A1 impulse-base breakout

Required pieces:

- prior impulse over 7/14/30/60 days;
- impulse threshold 25/50/75/100% for long alts, lower thresholds for majors if needed;
- base window 5/10/20/30/45 days;
- pullback max 15/25/35/50%;
- base tightness by ATR/range contraction;
- path smoothness: fewer single-bar jumps, controlled close progression, or impulse continuity score;
- breakout: close above base/range high or close above N-day high;
- confirmation: volume/range expansion if reliable, otherwise omitted/capped.

#### Lane B: A1 + compression

A1 impulse/base must pass plus one compression condition:

- realized-vol percentile below threshold;
- box width / ATR below threshold;
- final contraction range narrower than prior range;
- flat/declining volatility before close-confirmed breakout.

#### Lane C: H12 compression-to-expansion breakout

Compression condition can stand without the full A1 impulse:

- realized vol bottom 10/20/30 percentile over rolling 90/180 days;
- close outside compression range;
- parent and liquidity gate required for longs;
- compare to generic Donchian breakout controls.

#### Lane D: H13 flat-range escape

- flat range slope over 20/40/60 bars or days depending timeframe;
- narrow box relative to ATR;
- close outside box;
- failed close back inside box is a defined exit.

#### Lane E: H06 VCP-like contraction

- two or three contraction legs if extractable from bars;
- decreasing pullback ranges;
- final tightness filter;
- close-confirmed expansion.

If VCP is too discretionary or slow to implement robustly, it may be represented by range/ATR contraction rather than pattern-recognition heuristics.

### 4.5 Exit policy

Exit must be swept coherently, not as metadata.

Baseline exit classes:

1. fixed hold: 1d, 3d, 5d, 10d, 15d where timeframe permits;
2. ATR initial stop: 1.0x, 1.5x, 2.0x ATR;
3. structure stop: breakout bar low, base low, range low / range high for shorts;
4. failed close back inside base/range;
5. EMA/SMA trail: 10 EMA, 20 EMA, 10 SMA, 20 SMA;
6. ATR trail: 2x or 3x ATR trail;
7. time stop if no extension after N bars;
8. optional R-multiple partial only if implemented exactly; otherwise omit from rankable sweep.

MAE/MFE support:

- For every selected event, compute MAE/MFE through the maximum holding window before judging exit families.
- Report MAE/MFE distributions by lane, regime, symbol, and month.
- Use MAE/MFE to interpret exit suitability, not to post-hoc tune a single winner without multiplicity accounting.

### 4.6 Definition budget

Use coverage design, not a full Cartesian grid.

Recommended first non-smoke manifest:

- entry specs: 160-240 unique entry specs;
- exit policies: 6-10 unique exit policies;
- resulting candidate definitions: approximately 800-1800 if the cache architecture can handle exit overlays efficiently;
- if cache/exactness benchmarks project runtime above 48h, split into lanes/tranches while preserving the full PIT universe.

Target lane coverage:

- A1 impulse-base: 25%;
- A1 + compression: 30%;
- H12 RV compression breakout: 20%;
- H13 flat-range escape: 15%;
- H06 VCP-like contraction: 10%;
- shorts diagnostic: max 10-15% of total, if implemented safely.

Every definition must include a `definition_lane`, `subfamily_id`, `entry_policy_hash`, `exit_policy_hash`, `universe_policy_hash`, `rank_policy_hash`, `regime_policy_hash`, `funding_policy_hash`, and `parameter_vector_hash`.

## 5. Cache/optimization architecture

The A1/compression sweep should generalize the TSMOM table architecture.

Required cache classes:

1. decision calendar;
2. PIT liquid universe membership;
3. leader rank features;
4. Top-N membership;
5. impulse/base/path-smoothness features;
6. compression/range/RV features;
7. parent-regime gate;
8. funding gate;
9. breakout signal mask;
10. selected event key table;
11. interval outcome/path cache after selected keys freeze;
12. grouped aggregate by definition/symbol/month.

Long stages must:

- write deterministic partitions during execution;
- atomically write temp -> final shards;
- validate semantic cache hashes;
- support resume without replaying completed shards;
- fail closed on stale manifests;
- launch detached under tmux for long runs;
- send Telegram alerts on start, stage transition, failure, disk/RSS threshold, and completion;
- write a heartbeat file;
- avoid interactive polling except every 20-60 minutes.

## 6. Evidence gates

Before non-smoke aggregate:

- contract manifest exists;
- engine routing audit passes;
- PIT universe/rank exactness passes;
- feature timestamps obey `feature_available_ts <= decision_ts`;
- parent/funding gates are bound, not metadata-only;
- exit policies are executable;
- representative scalar/cache exactness mismatch count is zero;
- protected holdout interval audit passes;
- no event sampling/caps;
- no current-live/static/alphabetical selection;
- peak RSS below 6 GiB in benchmark;
- projected full cold runtime acceptable or explicitly approved.

After aggregate:

- cluster near-duplicate definitions;
- materialize representatives only;
- require real controls;
- run stress and forensics before stability review.

Mandatory forensics for survivors:

- leave-one-symbol;
- leave-one-month;
- leave-one-symbol-month;
- top-1/top-3 event exclusion;
- exact/proxy funding split;
- funding stress 150% and 200%;
- fee/slippage stress +2/+5/+8/+12 bps;
- no-slippage-field cap if still zero;
- nearest-neighbor control availability review;
- MAE/MFE exit suitability.

## 7. Prompt sequence

Send one prompt at a time. Review each report before sending the next.

---

## Prompt 1 - Contract, manifest, and tranche design

```text
Proceed with A1 + Compression Liquid Continuation Contract and Manifest Design v1.

This is analysis and manifest-design only.
Do not modify production code unless creating inert contract artifacts requires a profile stub; if code changes are needed, explain before changing.
Do not launch aggregate, materialization, validation, holdout, prior-high standalone, TSMOM, broad sweep, or live-prep.
Do not use event sampling or event caps.

Reference infrastructure report root:
`results/rebaseline/phase_kraken_a1_h06_h12_h13_sweep_infrastructure_report_20260708_v1`

Reference TSMOM cache architecture roots:
`results/rebaseline/phase_kraken_tsmom_calendar_universe_cache_builder_20260707_v1_20260707_123020`
`results/rebaseline/phase_kraken_tsmom_rank_topn_cache_builder_20260707_v1_20260707_153811`
`results/rebaseline/phase_kraken_tsmom_mask_selected_key_builder_20260707_v1_20260707_201216`
`results/rebaseline/phase_kraken_tsmom_outcome_grouped_aggregate_20260707_v1`

Create run root:
`results/rebaseline/phase_kraken_a1_compression_contract_manifest_20260708_v1/`

Goal:
Design a sweep-ready contract and curated manifest for A1 Liquid Leader Continuation Breakout with H06/H12/H13 compression as a coupled subfamily.

Scope:
- Primary family: A1/H01 liquid leader continuation breakout.
- Coupled subfamily: H06/H12/H13 compression / volatility contraction / flat-range escape.
- Prior-high/reclaim may appear only as a filter/context feature, not as a standalone entry system.
- OI veto must be omitted, capped, or sidecar-only unless local PIT OI is proven available.

Required design outputs:
1. Define family lanes:
   - A1 impulse-base breakout;
   - A1 + compression;
   - H12 RV compression breakout;
   - H13 flat-range escape;
   - H06 VCP-like contraction if mechanically implementable;
   - optional short diagnostic lane, capped.
2. Define entry semantics:
   - decision timeframe;
   - next-bar execution;
   - current-bar exclusion rules;
   - impulse windows/thresholds;
   - base/consolidation windows;
   - pullback depth;
   - path smoothness;
   - compression/RV/range features;
   - close-confirmed breakout trigger.
3. Define exit semantics:
   - fixed hold;
   - ATR stop/trail;
   - structure/base/range failure;
   - failed close back inside range;
   - 10/20 EMA/SMA trail;
   - time stop;
   - partial exits only if exact, otherwise excluded.
4. Define universe/ranking:
   - PIT liquid-universe policies;
   - leader rank metrics;
   - Top-N values;
   - non-alphabetic tie policy;
   - prior-high proximity filter-only options.
5. Define regimes/context:
   - parent BTC/ETH gates;
   - breadth/liquidity context;
   - funding crowding vetoes;
   - regime-conditioned reporting.
6. Define controls and forensics:
   - same-symbol controls;
   - same-regime controls;
   - generic breakout controls;
   - Donchian/simple breakout controls;
   - nearest-neighbor if features exist;
   - leave-one and top-event kill gates.
7. Create a curated first-tranche manifest target:
   - 160-240 entry specs;
   - 6-10 exit policies;
   - expected 800-1800 definition rows if cache/exactness supports it;
   - split into subtranches if projected runtime is too high.

Required files:
- `contract/a1_liquid_leader_continuation_contract.md`
- `contract/compression_subfamily_contract.md`
- `contract/entry_semantics.md`
- `contract/exit_stop_semantics.md`
- `contract/regime_context_semantics.md`
- `contract/control_semantics.md`
- `contract/forensic_kill_gates.md`
- `redesign/a1_h06_h12_h13_parameter_space_v1.yaml`
- `redesign/a1_h06_h12_h13_invalid_parameter_combinations_v1.yaml`
- `redesign/a1_h06_h12_h13_curated_sweep_definitions_v1.csv`
- `budget/definition_lane_budget.csv`
- `infrastructure/cache_reuse_generalization_plan.md`
- `prompts/next_phase_prompt.md`
- `decision_summary.json`
- `compact_review_bundle/`

Final response:
- run root;
- status;
- code modified yes/no;
- definition count proposed;
- lane counts;
- exit policy count;
- whether prior-high is filter-only;
- whether OI is omitted/capped/sidecar;
- next prompt path;
- compact bundle path.
```

---

## Prompt 2 - Binding implementation and cache dry-run profile

```text
Proceed with A1 + Compression Binding and Cache Dry-Run Implementation v1.

This is code implementation plus dry-run/exactness only.
Do not launch a non-smoke aggregate sweep.
Do not materialize candidates.
Do not run validation, holdout, TSMOM, prior-high standalone, or live-prep.
Do not use event sampling or event caps.

Reference contract root:
`<CONTRACT_ROOT_FROM_PROMPT_1>`

Create run root:
`results/rebaseline/phase_kraken_a1_compression_binding_cache_dry_run_20260708_v1/`

Goal:
Implement the coupled A1/H06/H12/H13 manifest binding and cache-builder profile, then prove on real data that event generation is PIT-safe, resumable, and exact against scalar representative checks.

Required implementation:
1. Add explicit profile name:
   `a1_compression_binding_cache_dry_run_20260708_v1`
2. Load only the curated manifest from Prompt 1.
3. Add explicit engine routing; do not use legacy text archetype inference.
4. Build or generalize semantic caches for:
   - decision calendar;
   - PIT liquid universe membership;
   - leader rank features;
   - Top-N membership;
   - impulse/base/path-smoothness features;
   - compression/RV/range features;
   - parent gate;
   - funding gate;
   - breakout signal masks;
   - selected event keys.
5. Do not build outcomes yet unless the profile explicitly labels a small exactness sentinel.
6. Apply parent/funding gates to A1/compression rows, not only TSMOM.
7. OI must be omitted, capped, or sidecar-only according to the contract.
8. All decision-input rows must satisfy `feature_available_ts <= decision_ts`.
9. All Top-N and universe panels must be PIT-dynamic, not static/current-live/alphabetical.
10. Selected keys must be frozen and content-hashed.
11. Long writes must be sharded/manifested and resumable.

Required dry-run benchmarks:
- representative 24-row scalar/cache exactness;
- medium 128-256 row scalar/cache exactness;
- one canary window with exact funding;
- one canary window with proxy/missing funding if present;
- no aggregate launch.

Required files:
- `binding/manifest_to_engine_trace.csv`
- `binding/definition_hash_audit.csv`
- `cache/decision_calendar_manifest.csv`
- `cache/liquid_universe_manifest.csv`
- `cache/leader_rank_manifest.csv`
- `cache/topn_membership_manifest.csv`
- `cache/impulse_base_feature_manifest.csv`
- `cache/compression_feature_manifest.csv`
- `cache/breakout_signal_manifest.csv`
- `cache/selected_event_key_manifest.csv`
- `audit/decision_input_leak_audit.csv`
- `audit/topn_dynamic_panel_audit.csv`
- `audit/parent_gate_binding_audit.csv`
- `audit/funding_gate_binding_audit.csv`
- `audit/oi_availability_cap_audit.csv`
- `audit/event_count_theoretical_vs_actual.csv`
- `benchmark/scalar_cache_exactness.csv`
- `performance/cache_builder_timing.csv`
- `performance/resume_safety_report.md`
- `decision_summary.json`
- `compact_review_bundle/`

Pass criteria:
- exactness mismatches = 0;
- leak violations = 0;
- static Top-N failures = 0;
- event sampling/caps = false;
- manifest validation failures = 0;
- peak RSS < 6 GiB;
- resume safety proven on at least one interrupted/reused shard path or simulated stale-manifest test.

Final response:
- run root;
- status;
- code modified yes/no;
- aggregate launched yes/no, must be no;
- manifest definitions loaded;
- selected event count;
- exactness pass yes/no;
- mismatch count;
- peak RSS;
- runtime;
- next recommended phase;
- compact bundle path.
```

---

## Prompt 3 - Mechanical canary and outcome/exit benchmark

```text
Proceed with A1 + Compression Mechanical Canary and Exit/Outcome Benchmark v1.

This is a mechanical canary and benchmark phase.
Do not launch the full aggregate sweep.
Do not run validation, holdout, broad materialization, prior-high standalone, TSMOM, or live-prep.
Do not use event sampling or event caps except for explicitly labeled scalar exactness sentinels.

Reference binding/cache root:
`<BINDING_CACHE_ROOT_FROM_PROMPT_2>`

Create run root:
`results/rebaseline/phase_kraken_a1_compression_mechanical_canary_20260708_v1/`

Goal:
Prove that A1/compression selected event keys, executable exits, MAE/MFE, outcome paths, controls scaffolding, funding/fee/slippage attribution, and protected-boundary rules work before any long sweep.

Required tasks:
1. Build interval/path outcome cache after selected-key freeze for canary subsets only.
2. Exercise all exit families from the manifest:
   - fixed hold;
   - ATR stop;
   - ATR trail;
   - structure/base/range failure;
   - failed close back inside range;
   - 10/20 EMA/SMA trail;
   - time stop.
3. Compute MAE/MFE through the max hold window for every canary event.
4. Run aggregate-vs-materialized parity on canary definitions.
5. Generate real controls for canary definitions where possible.
6. Report nearest-neighbor missing feature reasons if unavailable.
7. Run fee/funding/slippage stress grid on canary outputs.
8. Verify protected interval rule: decision, entry, exit, funding settlement, and outcome intervals must not cross `>= 2026-01-01` unless dropped/censored by declared rule.
9. Include at least:
   - one exact-funding window;
   - one proxy/missing-funding window;
   - one parent-expansion regime;
   - one rotation/mixed or stress regime;
   - A1-only, compression-only, and A1+compression lanes.

Required outputs:
- `mechanical/window_manifest.csv`
- `mechanical/canary_definition_manifest.csv`
- `cache/canary_interval_outcome_manifest.csv`
- `materialized/canary_event_ledgers/`
- `controls/canary_control_ledger.parquet`
- `audit/aggregate_vs_materialized_parity.csv`
- `audit/exit_module_binding_audit.csv`
- `audit/mae_mfe_path_audit.csv`
- `audit/protected_interval_audit.csv`
- `audit/control_semantics_audit.csv`
- `stress/canary_stress_summary.csv`
- `benchmark/canary_exactness_sentinel.csv`
- `performance/canary_runtime_projection.md`
- `decision_summary.json`
- `compact_review_bundle/`

Pass criteria:
- aggregate/materialized parity mismatch = 0;
- protected interval violations = 0;
- control rows nonzero or fail-closed with precise reasons;
- all exit modules executable;
- MAE/MFE available for canary events;
- no validation/holdout;
- projected full aggregate runtime acceptable or requires operator review.

Final response:
- run root;
- status;
- code modified yes/no;
- full aggregate launched yes/no, must be no;
- exit modules tested;
- canary event count;
- controls pass/fail;
- parity pass/fail;
- projected full runtime;
- blockers;
- next recommended phase;
- compact bundle path.
```

---

## Prompt 4 - Full aggregate train screen, detached

```text
Proceed with A1 + Compression Full Train Aggregate Screen v1.

This is a full train-only aggregate screen.
Launch detached under tmux or equivalent if projected runtime is above 30 minutes.
Enable Telegram if configured.
After confirming PID, run root, log path, heartbeat, and alert events, stop active polling and return control to the operator.

Do not run validation, final holdout, materialization, prior-high standalone, TSMOM, broad two-family work, or live-prep.
Do not use event sampling, event caps, `--max-symbols`, or portfolio caps.

Reference mechanical canary root:
`<MECHANICAL_CANARY_ROOT_FROM_PROMPT_3>`

Create fresh run root:
`results/rebaseline/phase_kraken_a1_compression_full_train_aggregate_20260708_v1_<UTC_SUFFIX>/`

Goal:
Run the curated A1/H06/H12/H13 coupled manifest over the full eligible train window and PIT universe, using the validated cache/outcome architecture.

Required scope:
- all curated manifest definitions from Prompt 1;
- full PIT eligible universe for each universe policy;
- all lanes: A1-only, A1+compression, H12, H13, H06 if implemented, and short diagnostics if allowed;
- all predeclared exit policies;
- all regime/funding/universe/rank variants;
- no final holdout.

Required gates:
- scalar fallback false except labeled exactness sentinel;
- exactness sentinel pass;
- decision-input leak violations = 0;
- static Top-N failures = 0;
- protected interval violations = 0;
- event sampling/caps false;
- selected-key freeze before outcome read;
- manifest validation failures = 0;
- peak RSS below soft limit with margin;
- checkpoint/resume enabled.

Detached monitoring:
- Telegram on start/stage/failure/completion;
- heartbeat file updated at least every 10 minutes;
- RSS/disk alert thresholds;
- no interactive polling more often than every 20-60 minutes.

Required outputs:
- `aggregate/grouped_aggregate_summary.parquet`
- `aggregate/lane_summary.csv`
- `aggregate/exit_policy_summary.csv`
- `aggregate/regime_context_summary.csv`
- `aggregate/symbol_month_summary.parquet`
- `audit/exactness_sentinel.csv`
- `audit/protected_interval_audit.csv`
- `audit/no_scalar_fallback_audit.csv`
- `audit/topn_dynamic_panel_audit.csv`
- `performance/stage_timing_breakdown.csv`
- `performance/cache_hit_miss_report.csv`
- `decision/nonfutile_aggregate_candidates.csv`
- `decision_summary.json`
- `compact_review_bundle/`

Final response:
- run root;
- status;
- total runtime;
- definitions evaluated;
- selected event count;
- nonfutile aggregate count;
- exactness pass/mismatch count;
- event sampling/caps;
- peak RSS;
- top nonfutile lanes;
- next recommended phase;
- compact bundle path.
```

---

## Prompt 5 - Survivor clustering and targeted materialization profile

```text
Proceed with A1 + Compression Survivor Clustering and Targeted Materialization Profile v1.

This is survivor selection, duplicate clustering, and launch-profile preparation.
Do not launch materialization yet.
Do not run validation, final holdout, prior-high standalone, TSMOM, broad sweep, or live-prep.
Do not reselect outside the aggregate output.

Reference full aggregate root:
`<FULL_AGGREGATE_ROOT_FROM_PROMPT_4>`

Create run root:
`results/rebaseline/phase_kraken_a1_compression_survivor_materialization_plan_20260708_v1/`

Goal:
Cluster aggregate-nonfutile definitions, select a small diversified representative set for targeted materialization, and implement/preflight a dedicated materialization-controls-stress profile.

Required tasks:
1. Load aggregate-nonfutile rows only from the full aggregate root.
2. Cluster near-duplicates by:
   - selected event overlap;
   - symbol-month return correlation;
   - parameter vector similarity;
   - same lane/exit/regime family.
3. Select representatives by:
   - lane diversity;
   - exit-policy diversity;
   - regime-context diversity;
   - net R and cost/funding robustness;
   - not simply top scaled net R.
4. Freeze deferred duplicates.
5. Build a shortlist, target 3-8 representatives unless there are fewer credible clusters.
6. Implement or verify profile:
   `a1_compression_targeted_materialization_controls_stress_20260708_v1`
7. Run lineage dry-run only; do not materialize yet.
8. Verify exact full aggregate, selected-key, outcome-cache, parameter-hash, and manifest lineage.

Required outputs:
- `selection/survivor_screening_rules.md`
- `selection/survivor_shortlist.csv`
- `selection/deferred_duplicates.csv`
- `selection/duplicate_cluster_report.csv`
- `profile/profile_contract.md`
- `preflight/shortlist_lineage_audit.csv`
- `preflight/profile_dry_run_report.md`
- `prelaunch/next_materialization_launch_prompt.md`
- `decision_summary.json`
- `compact_review_bundle/`

Final response:
- run root;
- status;
- materialization launched yes/no, must be no;
- aggregate candidates screened;
- duplicate clusters;
- shortlisted representatives;
- lineage audit pass/fail;
- supported launch profile;
- next prompt path;
- compact bundle path.
```

---

## Prompt 6 - Targeted materialization, controls, stress, and forensics

```text
Proceed with A1 + Compression Targeted Materialization / Controls / Stress / Forensics v1.

Use the dedicated profile created in Prompt 5.
This is train-only targeted materialization and forensic review.
Do not run validation, final holdout, prior-high standalone, TSMOM, broad sweep, or live-prep.
Do not re-rank, reselect, expand, or replace candidates.
Do not materialize deferred duplicates unless explicitly selected in the Prompt 5 shortlist.

Reference survivor plan/profile root:
`<SURVIVOR_PLAN_ROOT_FROM_PROMPT_5>`

Reference full aggregate root:
`<FULL_AGGREGATE_ROOT_FROM_PROMPT_4>`

Create fresh run root:
`results/rebaseline/phase_kraken_a1_compression_targeted_materialization_controls_stress_20260708_v1_<UTC_SUFFIX>/`

Required tasks:
1. Re-run shortlist lineage gate and fail closed on mismatch.
2. Build event-level ledgers for shortlisted representatives only.
3. Generate real matched controls:
   - same-symbol;
   - same-regime;
   - generic breakout;
   - Donchian/simple breakout;
   - nearest-neighbor if features exist, otherwise precise fail-closed reason.
4. Run stress:
   - fee stress;
   - slippage stress using historical cap plus forward-capture calibration if available;
   - funding 150% and 200%;
   - exact/proxy funding split;
   - raw/scaled if any scaling is used;
   - no-slippage-field cap if applicable.
5. Run MAE/MFE and exit suitability review.
6. Run forensics:
   - leave-one-symbol;
   - leave-one-month;
   - leave-one-symbol-month;
   - top-1/top-3 event exclusion;
   - regime-slice dependency;
   - lane/exit dependency.
7. Assign train-only evidence levels and candidate-library status.
8. Do not use promotion labels beyond current evidence contract.

Required outputs:
- `materialized/event_ledgers/`
- `controls/control_ledger/control_ledger.parquet`
- `controls/control_semantics_audit.csv`
- `stress/targeted_stress_summary.csv`
- `stress/funding_fee_slippage_attribution.csv`
- `forensics/leave_one_out_symbol.csv`
- `forensics/leave_one_out_month.csv`
- `forensics/leave_one_out_symbol_month.csv`
- `forensics/top_event_dependency.csv`
- `forensics/mae_mfe_exit_review.csv`
- `decision/candidate_level_decision_table.csv`
- `candidate_library/a1_compression_candidate_library_update.csv`
- `decision_summary.json`
- `compact_review_bundle/`

Final response:
- run root;
- status;
- materialized definitions;
- event rows per definition;
- control rows per definition;
- stress pass/fail;
- forensics pass/fail;
- candidates allowed to proceed to train-only stability review, if any;
- evidence levels;
- compact bundle path.
```

## 8. Expected decision logic after Prompt 6

Do not proceed to validation automatically. After targeted materialization and forensics, apply:

- if no candidates survive concentration/funding/slippage/control gates: close current A1/compression translation and route to another family or redesign;
- if one or more candidates survive: run a separate train-only stability/CPCV planning phase, still not final holdout;
- if survivors depend on proxy funding or one symbol-month: diagnostic only;
- if controls are missing stronger nearest-neighbor features: keep evidence capped.
