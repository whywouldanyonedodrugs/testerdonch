# Stage 2E — C01 Level-3 Runner Implementation and Economic Execution

```text
task_id: donch_bt_stage_2e_c01_level3_economic_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: yes — exactly one C01 Level-3 run under the frozen contract below
protected_outcome_access: no
Level4_controls_authorized: no
```

## Exact approval

Implement, independently review, and execute the C01 Level-3 economic runner for:

```text
family: C01_debetaed_residual_shock_path_bifurcation
venue: Kraken only
rankable interval: [2023-01-01, 2026-01-01)
final contract SHA-256:
  c655e94c35412354356bb7f89c07ca17b71c2ae6537a2a1c42aa3dce928ba77d
Stage 2D commit:
  bb9aa9f9cc1b5dcf1c749349ec33f5c6c54ed131
onset event tape SHA-256:
  e4587653aec82fb66ab6775284501ca768b6689a6b14cdb17a90799f32cea6b7
```

This approval permits outcome computation only for the 16 frozen Level-3 definitions. Do not run Level-4 controls, modify definitions, inspect 2026+ outcomes, or use capture data.

## Start and authority checks

Verify the repository, `AGENTS.md` chain, clean synchronized Git state, machine contracts, and all input paths/hashes. Record the actual starting commit. A clean descendant of `bb9aa9...` containing only expected archive/governance records is acceptable; stop for any material code, data, or authority change.

Required immutable inputs:

```text
docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md
docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_LEVEL3_DEFINITION_REGISTER.csv
docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_LEVEL3_DECISION_RULES.json
docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_LEVEL4_CONTROL_CONTRACT.md
docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1/C01_ONSET_EVENT_TAPE.parquet
```

Also verify the frozen generator, feature, cohort, reference-panel, and economic-draft hashes recorded in the approval packet.

## Binding interpretation

Use the Stage 2D machine definition register and decision-rules JSON as the precise interpretation:

- all 8 `btc_eth_ols_daily_v1` definitions are primary-model definitions and are evaluated independently;
- any one may earn permission for its own later controls only by passing every gate;
- definitions may never be pooled to pass a gate;
- all 8 `btc_only_ols_daily_v1` definitions are robustness-only and cannot grant permission or rescue a primary failure;
- no result in this task authorizes Level-4 execution.

Level-3 gates use full-period **base fee/slippage net fixed-notional bps**, excluding funding. Report funding-adjusted partitions separately under the frozen hierarchy; they cannot rescue a gate failure.

For concentration metrics, freeze these formulas before outcome access:

```text
total_net = sum(base_fee_slippage_net_bps) for executed trades in one definition
max_symbol_share = max(positive symbol aggregate net) / total_net
max_episode_share = max(positive episode aggregate net) / total_net
max_year_positive_share = max(positive year aggregate net) / sum(positive year aggregates)
```

If a required denominator is non-positive or undefined, the corresponding gate fails closed.

Use the exact bootstrap helper and constants in `tools/kraken_c01_prerun_contract.py`; do not rewrite or alter its semantics.

## Phase A — Implement without real outcomes

Create:

```text
tools/run_kraken_c01_level3_economic.py
unit_tests/test_kraken_c01_level3_economic.py
```

Before reading real post-onset paths, implement and pass synthetic tests for:

- all 16 definitions, including zero-trade rows;
- smooth and jump-confirmation entries;
- dominant-bar identity and 24h confirmation boundary;
- mark-close stop trigger and next-trade-open execution;
- timeout exit at the first executable trade-bar open at or after entry plus 6h/24h;
- same-bar stop/timeout ambiguity failing closed;
- missing trade/mark/next-open/funding and lifecycle-invalid rows failing closed;
- no artificial close at 2026-01-01;
- fixed-notional gross, base net, and stress net bps;
- signed funding cashflow and exact/mixed/imputed/zero partitions;
- chronological actual-exit non-overlap by symbol and definition;
- skipped onsets excluded from economic metrics;
- exact concentration formulas and gate boundaries;
- deterministic 10,000-resample episode bootstrap;
- protected, pre-2023, non-Kraken, duplicate-address, and mixed-file rejection.

Run the relevant repository guard and regression suites. Obtain an independent pre-run review of the actual runner diff, tests, schemas, input guards, and command.

Create one clean implementation commit before economic execution. Record that commit as the runner commit. Do not execute if the review is not `approve`.

## Phase B — Execute once

Use the frozen command interface with a new UTC-suffixed root:

```bash
./.venv/bin/python tools/run_kraken_c01_level3_economic.py \
  --contract docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md \
  --definition-register docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_LEVEL3_DEFINITION_REGISTER.csv \
  --event-tape docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1/C01_ONSET_EVENT_TAPE.parquet \
  --run-root results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_<UTC_SUFFIX> \
  --execute-economic-run
```

The runner must refuse an existing root and verify all hashes before the first outcome read.

Execute every eligible onset; no caps, sampling, threshold changes, or branch deletion. Process each definition independently. Non-overlap must follow actual executed exits. No combined portfolio.

## Required run artifacts

At minimum:

```text
RUN_MANIFEST.json
INPUT_AND_HASH_AUDIT.json
PERIOD_AND_PROTECTED_AUDIT.json
DEFINITION_REGISTER.csv
EVENT_ELIGIBILITY_AND_SKIP_LEDGER.parquet
TRADE_LEDGER.parquet
DEFINITION_METRICS.csv
LEVEL3_GATE_MATRIX.csv
FUNDING_PARTITION_REPORT.csv
CONCENTRATION_REPORT.csv
BOOTSTRAP_REPORT.csv
DECISION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
```

The trade ledger must include causal entry/confirmation, actual exit and reason, trade/mark path references, gross return, base and stress fee/slippage net bps, funding cashflow and partition, canonical episode, symbol/year, invalid/skip reason, code/config/data hashes, and protected-row count.

Report for every definition:

- onset, confirmed, invalid, skipped, and executed counts;
- counts by year and symbol;
- gross, base-net, stress-net mean and median;
- exact-funded subset by calendar period;
- mixed/imputed/zero-boundary sensitivity separately;
- bootstrap CI;
- concentration shares;
- every gate result.

## Decision rule

Return exactly one family decision:

```text
level3_no_primary_pass_stop
```

when no primary-model definition passes all gates, or:

```text
level3_primary_pass_controls_pending_separate_approval
```

when at least one primary-model definition passes all gates.

A pass means only that the named definition may be considered for the frozen Level-4 controls after another human approval. It is not validation, promotion, or live readiness.

Do not run controls in this task, even after a pass.

## Post-run review and integration

Independently review the actual runner commit, run root, ledgers, gate calculations, funding partitions, hashes, protected audit, and claims. Preserve any failed or partial root as provenance; never overwrite it.

After review:

- update the task archive and required registries/continuity records;
- create at most one post-run documentation/registry commit;
- fast-forward and non-force push only under the standing reviewed-task workflow;
- package and upload with `drive_handoff: approved_default`;
- verify remote bytes and SHA-256;
- retain the local package and full run root.

## Prohibited

- 2026+ or mixed protected payload reads;
- Level-4 controls or matched non-events;
- threshold, cohort, branch, timeout, cost, funding, or gate changes;
- event sampling/caps;
- pooling definitions or funding partitions;
- using BTC-only to rescue primary failure;
- holdout access;
- capture access;
- validation, deployment, or live claims;
- push/merge outside the exact reviewed commits above.

## Final response

```text
status:
actual_starting_commit:
runner_commit:
run_root:
contract_and_input_hash_verification:
definition_execution_counts:
primary_results_and_gate_matrix:
robustness_results:
funding_partition_results:
concentration_and_bootstrap:
protected_rows_opened:
Level4_controls_run: no
family_decision:
files_and_commits:
tests_and_reviews:
artifacts_and_sha256:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
rollback:
human_approval_required:
```
