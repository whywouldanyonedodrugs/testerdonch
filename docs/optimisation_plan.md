Proceed with Kraken Aggregate Cache Acceleration Foundation v1.

This is an infrastructure acceleration phase. It must not launch a new alpha sweep, must not relabel current or prior results, and must not modify or resume the currently active two-family run under changed code.

Current active run for read-only profiling only:
`results/rebaseline/phase_kraken_uncapped_tier1_two_family_sweep_repaired_memorysafe_20260706_v1_20260706_133455`

Known active-run state from operator:
- status active;
- current stage `tsmom-v4-v5-aggregate-screen`;
- runtime about 2h22m at report time;
- processed `195 / 777` candidate-symbol rows;
- current rate about `0.023 rows/sec`;
- ETA about 7h;
- RSS about 0.70 GiB;
- no protocol failure found;
- memory-safe cache lifecycle working;
- throughput still slow.

Do not stop the active run unless explicitly instructed. Do not patch the active root in place. If code changes are made, do not resume the active run under the new code unless a compatibility hash/migration report says it is safe and the operator approves.

Create new run root:
`results/rebaseline/phase_kraken_aggregate_cache_acceleration_foundation_20260706_v1/`

Required files/modules:
- Prefer adding reusable cache helpers in a separate module, e.g. `tools/kraken_aggregate_cache_layer.py`, to reduce risk to the active runner.
- Patch `tools/run_kraken_family_engine_aggregate_first_sweep.py` only behind a new profile/flag.
- Update `unit_tests/test_kraken_family_engine_aggregate_first_sweep.py` or add a separate cache test module.

Required phase profile:
`aggregate_cache_acceleration_foundation_20260706_v1`

Do not launch non-smoke aggregate sweeps from this phase.

Required stages:
1. `preflight-and-active-run-snapshot`
2. `cache-contracts-and-leak-boundary`
3. `candidate-registry-and-shard-manifest`
4. `heartbeat-and-hotspot-timers`
5. `decision-calendar-cache`
6. `universe-membership-cache`
7. `tsmom-decision-input-caches`
8. `tsmom-interval-outcome-cache`
9. `prior-high-feature-and-outcome-cache-contract`
10. `cache-vs-scalar-exactness-benchmark`
11. `cache-performance-benchmark`
12. `parallelism-readiness-benchmark`
13. `accelerated-two-family-launch-readiness`
14. `decision-report`
15. `compact-review-bundle`
16. `all`

Stage 1 — preflight-and-active-run-snapshot

Read-only inspect the active run if it exists.

Record:
- active tmux/session state;
- active run root;
- progress files;
- current candidate-symbol throughput;
- RSS;
- current stage;
- whether any protocol failure is present.

Do not modify active run artifacts.

Write:
- `preflight/active_run_snapshot.md`
- `preflight/input_manifest.json`
- `preflight/code_hash_manifest.csv`
- `preflight/git_status_snapshot.txt`

Stage 2 — cache-contracts-and-leak-boundary

Define two cache classes.

A. Decision-input caches:
- decision calendar;
- universe membership;
- rank features;
- top-N membership;
- TSMOM signal matrix;
- parent gate;
- funding gate;
- vol scale;
- prior-high features.

Invariant:
`feature_available_ts <= decision_ts`

B. Outcome/execution caches:
- TSMOM interval returns;
- prior-high execution results;
- stop/target/VWAP/structure outcomes;
- funding paid during hold.

Outcome caches may use bars after `decision_ts`, but must never be used for:
- universe eligibility;
- ranking;
- signal generation;
- parent/funding gate selection;
- candidate selection;
- tuning;
- control construction before event keys are frozen.

Safe aggregate flow:
1. Build `selected_event_keys` from decision-input caches only.
2. Freeze and hash `selected_event_keys`.
3. Join outcome/execution cache.
4. Compute raw and scaled R.
5. Aggregate.

Write:
- `contracts/cache_class_contract.md`
- `contracts/decision_input_cache_schema.yaml`
- `contracts/outcome_cache_schema.yaml`
- `contracts/leak_boundary_policy.md`

Hard gate:
Any code path that reads outcome fields before selected event keys are frozen fails the phase.

Stage 3 — candidate-registry-and-shard-manifest

Implement deterministic candidate registry before aggregate evaluation.

Required fields:
- `run_id`
- `candidate_definition_id`
- `candidate_symbol_id`
- `family`
- `symbol_id`
- `symbol`
- `parameter_vector_hash`
- `definition_hash`
- `universe_policy_hash`
- `rank_policy_hash`
- `signal_policy_hash`
- `parent_gate_policy_hash`
- `funding_gate_policy_hash`
- `vol_scale_policy_hash`
- `interval_policy_hash`
- `feature_policy_hash`
- `event_semantics_version`
- `code_hash`
- `config_hash`
- `input_manifest_hash`
- `train_window_hash`
- `candidate_identity_hash`
- `status`

Candidate registry is the authoritative list. Do not discover candidate-symbol rows opportunistically during aggregate evaluation.

Implement deterministic shard manifest protocol:
- write candidate summary / symbol-month / fold summary / event-address rows under temp path;
- compute content hashes from canonical sorted rows, not unstable parquet metadata;
- write `shard_manifest.json`;
- atomically move temp path to final path;
- update global shard manifest;
- resume only from valid manifests and matching hashes.

Write:
- `registry/candidate_registry_schema.md`
- `registry/candidate_registry_sample.csv`
- `shards/shard_manifest_schema.json`
- `shards/resume_protocol.md`

Tests:
- stale shard rejected;
- path-only shard not accepted;
- hash mismatch rejected;
- same candidate cannot be completed twice;
- candidate identity changes if code/config/source/cache hashes change.

Stage 4 — heartbeat-and-hotspot-timers

Add per-candidate heartbeat and detailed timers to aggregate stage.

Heartbeat fields:
- run_id
- stage
- family
- current candidate identity hash
- current definition id
- current symbol
- processed candidate-symbol rows
- total candidate-symbol rows
- candidate runtime seconds
- rolling rows/sec
- RSS GiB
- last completed candidate identity hash
- updated_at

Timers:
- candidate_total_seconds
- cache_read_seconds
- universe_join_seconds
- rank_join_seconds
- signal_join_seconds
- gate_join_seconds
- interval_join_seconds
- event_mask_seconds
- summary_groupby_seconds
- shard_write_seconds
- selected_event_count
- rss_before
- rss_after

Write:
- `performance/aggregate_heartbeat_schema.json`
- `performance/hotspot_timer_schema.csv`
- patch aggregate stage so future runs update heartbeat after each candidate-symbol row.

Stage 5 — decision-calendar-cache

Build canonical decision calendars by family/cadence/timeframe.

Fields:
- family
- cadence
- timeframe
- decision_ts
- decision_bar_open_ts
- decision_bar_close_ts
- decision_available_ts
- entry_search_start_ts
- train_window_id
- decision_calendar_hash

Rules:
- `decision_ts < 2026-01-01`
- for train events whose exit would cross train cutoff, apply explicit censor/drop rule and record `interval_censored_train_boundary`.

Write:
- `cache/decision_calendar/`
- `cache_manifests/decision_calendar_manifest.csv`
- `audit/decision_calendar_train_boundary_audit.csv`

Stage 6 — universe-membership-cache

Implement universe membership cache for PIT policies used by current TSMOM and prior-high definitions.

Schema:
- decision_ts
- universe_policy_hash
- family
- symbol
- symbol_id
- eligible
- reason_code
- lifecycle_status
- listed_asof
- delisted_asof
- data_available
- data_validity_state
- liquidity_score
- liquidity_rank
- liquidity_source_start_ts
- liquidity_source_end_ts
- liquidity_feature_available_ts
- feature_available_ts
- input_manifest_hash

Rules:
- one row per `(decision_ts, universe_policy_hash, symbol_id)`;
- no static list output as rankable universe;
- `feature_available_ts <= decision_ts`;
- `liquidity_source_end_ts <= decision_ts`;
- no alphabetical/first-N/current-live selection.

Also write:
- `universe_decision_audit` with eligible symbol count, selected symbols hash, excluded counts, max feature timestamp, PIT violation count.

Outputs:
- `cache/universe_membership/`
- `cache/universe_decision_audit/`
- `cache_manifests/universe_membership_manifest.csv`
- `audit/universe_cache_pit_audit.csv`

Stage 7 — TSMOM decision-input caches

Implement cache tables for TSMOM decision inputs.

A. TSMOM rank feature:
- symbol_id
- decision_ts
- cadence
- lookback_days
- rank_metric
- rank_feature_hash
- trailing_return
- risk_adjusted_return
- relative_strength_score
- volatility_estimate
- feature_source_start_ts
- feature_source_end_ts
- feature_available_ts
- valid_asof
- invalid_reason

B. Top-N membership:
- decision_ts
- rank_policy_hash
- universe_policy_hash
- rank_feature_hash
- symbol_id
- eligible_universe
- rank_feature_valid
- score
- rank
- top_n
- top_n_pass
- tie_break_value
- tie_break_policy_hash
- rank_population_count
- max_decision_input_available_ts
- invalid_reason

C. TSMOM signal matrix:
- symbol_id
- decision_ts
- signal_policy_hash
- lookback_days
- side_policy
- momentum_value
- signal_pass_long
- signal_pass_short
- signal_pass_long_flat
- feature_source_start_ts
- feature_source_end_ts
- feature_available_ts
- valid_asof
- invalid_reason

D. Parent gate matrix.

E. Funding gate matrix.

F. Vol-scale matrix.

Rules:
- all are decision-input caches;
- all must satisfy feature availability <= decision timestamp;
- top-N must be recomputed per decision timestamp;
- tie-break cannot be alphabetical/first-N;
- rank policy hash must include population rule.

Outputs:
- `cache/tsmom_rank_feature/`
- `cache/tsmom_topn_membership/`
- `cache/tsmom_signal_feature/`
- `cache/parent_regime_gate/`
- `cache/funding_gate/`
- `cache/vol_scale/`
- `cache_manifests/tsmom_decision_input_cache_manifest.csv`
- `audit/tsmom_decision_input_leak_audit.csv`
- `audit/topn_correctness_audit.csv`
- `audit/dynamic_panel_regression_audit.csv`

Stage 8 — TSMOM interval-outcome-cache

Implement TSMOM interval return outcome cache.

Schema:
- symbol_id
- symbol
- decision_ts
- interval_policy_hash
- cadence
- hold_interval
- side_basis
- entry_ts
- entry_price
- exit_ts
- exit_price
- exit_reason
- interval_valid
- invalid_reason
- raw_gross_R
- raw_fee_R
- raw_funding_R
- raw_slippage_R
- raw_net_R
- long_raw_gross_R
- long_raw_fee_R
- long_raw_funding_R
- long_raw_slippage_R
- long_raw_net_R
- short_raw_gross_R
- short_raw_fee_R
- short_raw_funding_R
- short_raw_slippage_R
- short_raw_net_R
- funding_boundary_count
- exact_funding_count
- proxy_funding_count
- funding_unavailable_count
- entry_bar_source_ts
- exit_bar_source_ts
- max_execution_source_ts
- max_funding_source_ts
- train_boundary_censored
- lifecycle_exit_flag
- data_gap_flag
- input_manifest_hash

Rules:
- this is outcome cache, not decision input;
- do not require max_execution_source_ts <= decision_ts;
- never allow selection/ranking/gating to read outcome fields before selected event keys are frozen;
- vol scaling applied in aggregate layer, not baked into raw interval returns;
- scaled values stored separately.

Outputs:
- `cache/tsmom_interval_return/`
- `cache_manifests/tsmom_interval_return_manifest.csv`
- `audit/outcome_cache_selection_leak_audit.csv`

Stage 9 — prior-high feature-and-outcome-cache-contract

Implement contracts and maybe minimal cache generation for prior-high.

Feature cache:
- symbol_id
- decision_ts
- feature_policy_hash
- signal_timeframe
- prior_high_lookback
- prior_high_value
- prior_high_source_start_ts
- prior_high_source_end_ts
- current_decision_close
- current_decision_close_ts
- current_decision_available_ts
- distance_to_high
- reclaim_pass
- reclaim_transition
- retest_window_state
- ath_value
- long_window_high_value
- atr_timeframe
- atr_window_bars
- atr_value
- atr_available_ts
- vwap_mode
- vwap_value
- vwap_available_ts
- structure_stop_value
- feature_available_ts
- valid_asof
- invalid_reason

Execution result cache:
- symbol_id
- decision_ts
- execution_policy_hash
- entry_ts
- entry_price
- stop_ts
- stop_price
- target_ts
- target_price
- exit_ts
- exit_price
- exit_reason
- raw_gross_R
- raw_fee_R
- raw_funding_R
- raw_slippage_R
- raw_net_R
- same_bar_conflict_flag
- touch_fill_used
- passive_fill_used
- execution_valid
- invalid_reason

Rules:
- prior-high feature must satisfy `feature_available_ts <= decision_ts`;
- current decision bar may be reclaim close, but prior-high lookback excludes it;
- execution may use later bars only after event selection;
- touch/passive/target-first assumptions forbidden.

Outputs:
- `contracts/prior_high_feature_cache_contract.md`
- `contracts/prior_high_execution_result_cache_contract.md`
- if implemented now, write cache manifests and audits;
- if not implemented, write `prior_high_cache_deferred_report.md` and explain what remains.

Stage 10 — cache-vs-scalar-exactness-benchmark

Compare cache path against existing scalar path.

Representative tests:
- TSMOM top-major daily long definition.
- TSMOM tier A/B top-N definition.
- TSMOM funding-aware definition.
- Prior-high breakout definition, if cache implemented.
- Prior-high ATR/VWAP/structure definition, if cache implemented.

Compare:
- selected event keys;
- event counts;
- raw gross/fee/funding/slippage/net R;
- scaled R;
- symbol-month summaries;
- active caps;
- invalid reason counts.

Write:
- `benchmark/cache_vs_scalar_exactness.csv`
- `benchmark/cache_vs_scalar_mismatch_report.md`

Hard gate:
Any mismatch not explained by an explicitly accepted cache-v1 limitation blocks accelerated launch readiness.

Stage 11 — cache-performance-benchmark

Benchmark current scalar vs cache path on a bounded, representative workload.

Use:
- TSMOM v4/v5 definitions subset;
- prior-high v2 subset if prior-high cache implemented;
- full PIT universe for the benchmark subset if runtime is acceptable;
- otherwise use mechanical benchmark labels only.

Report:
- scalar runtime estimate;
- cache build time;
- cache evaluation time;
- cache read time;
- total runtime;
- expected amortization break-even;
- estimated speedup;
- storage usage;
- cache hit rate;
- largest cache artifacts.

Write:
- `benchmark/cache_performance_benchmark.csv`
- `benchmark/cache_runtime_projection.md`
- `benchmark/cache_storage_report.md`

Stage 12 — parallelism-readiness-benchmark

Do not turn on parallelism for production yet.

Benchmark only:
- 1 worker
- 2 workers
- optionally 4 workers if disk/RSS safe

Rules:
- immutable cache reads only;
- deterministic shard writes;
- no shared mutable in-memory cache;
- worker-local memory guard;
- parent validates shard hashes.

Write:
- `parallelism/parallelism_readiness_report.md`
- `parallelism/worker_resource_profile.csv`

Stage 13 — accelerated-two-family-launch-readiness

Do not launch accelerated two-family sweep in this phase.

Write:
- `prelaunch/accelerated_two_family_launch_readiness.json`
- `prelaunch/next_accelerated_two_family_command.md` only if all gates pass.

Launch readiness requires:
- decision-input leak audits pass;
- outcome leak audit pass;
- candidate registry/shard resume pass;
- TSMOM cache exactness pass;
- prior-high cache exactness pass or prior-high clearly still scalar but safe;
- performance benchmark shows material improvement or clear scoped benefit;
- no protected timestamp violations;
- no stale cache manifests;
- no static universe regression;
- no event sampling/event caps.

Decision options:
- `cache_acceleration_ready_for_tsmom_only_launch`
- `cache_acceleration_ready_for_two_family_launch`
- `cache_acceleration_partial_use_universe_and_tsmom_only`
- `repair_cache_exactness_next`
- `repair_cache_leakage_next`
- `repair_cache_performance_next`
- `blocked_by_protocol_issue`

Testing:
Add unit tests for:
- decision cache availability timestamps;
- outcome cache not usable before selected-event freeze;
- universe membership one row per decision/policy/symbol;
- top-N dynamic ranking and deterministic non-alphabetical tie-break;
- interval outcome cache cannot leak into selection;
- candidate registry identity uniqueness;
- shard manifest resume validation;
- stale cache manifest rejection;
- cache-vs-scalar equality on fixtures;
- protected cutoff/censor handling;
- no event caps/sampling introduced;
- active run not modified.

Run:
`./.venv/bin/python -m py_compile tools/run_kraken_family_engine_aggregate_first_sweep.py tools/qlmg_evidence_contracts.py tools/qlmg_real_controls.py tools/qlmg_screening_core.py`

`./.venv/bin/python -m unittest unit_tests.test_kraken_family_engine_aggregate_first_sweep`

Smoke:
Use a dedicated smoke for `aggregate_cache_acceleration_foundation_20260706_v1`.

Final response:
- run root;
- status;
- active run modified yes/no;
- active run stopped yes/no;
- cache contracts written yes/no;
- candidate registry implemented yes/no;
- shard resume implemented yes/no;
- heartbeat/timers implemented yes/no;
- universe cache implemented yes/no;
- TSMOM decision caches implemented yes/no;
- TSMOM interval outcome cache implemented yes/no;
- prior-high cache implemented/deferred;
- decision-input leak audits pass yes/no;
- outcome leak audit pass yes/no;
- cache-vs-scalar exactness pass yes/no;
- performance speedup;
- storage added;
- parallelism readiness;
- accelerated launch decision;
- next command path if allowed;
- compact bundle path.