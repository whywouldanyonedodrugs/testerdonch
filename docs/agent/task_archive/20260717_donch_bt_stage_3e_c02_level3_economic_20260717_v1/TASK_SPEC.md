# Stage 3E — C02 Level-3 Runner Implementation and Economic Execution

```text
task_id: donch_bt_stage_3e_c02_level3_economic_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: yes — exactly one C02 Level-3 run under the frozen contract below
protected_outcome_access: no
Level4_controls_authorized: no
new_market_data_acquisition: no
```

## Exact authorization

Implement, independently review, and execute one C02 Level-3 train-period economic run for:

```text
lineage:
    C02_positive_resolution_aware_spot_led_continuation_v1

venue:
    Kraken only

rankable interval:
    [2023-01-01, 2026-01-01)

Stage 3D commit:
    b4441477791814f9f173df01fe452e93e5e94a07

final contract SHA-256:
    1c4f7f6ec81fa86d1c1355ce899570bdec85e413ad4febe110f33cc4ec565496

primary identity-set hash:
    7dbdb3763b9131480f712f60c2e7a4d0822f65a276b4ed5c5c00bdb804e3c42c

robustness identity-set hash:
    f3284aaf54da7c2f53d6a3561eab8e92cc639c40c7b9c025ed1991ac63bf7ca1
```

This authorization permits economic outcome computation only for the four frozen definitions. It does not authorize controls, negative spot-led, perp-led, failure, shifted-clock, alternative-threshold, alternative-horizon, or protected-period work.

## Start and authority checks

Verify:

- repository root and applicable `AGENTS.md` chain;
- clean synchronized `main`;
- current machine contracts;
- immutable Stage 3A–3D authority and hashes;
- supported repository commands.

Record the actual starting commit. A clean descendant of `b444147...` containing only expected task/archive records is acceptable. Stop for a material code, data, contract, or protected-boundary change.

Required immutable files:

```text
docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_FINAL_LEVEL3_ECONOMIC_CONTRACT.md
docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_LEVEL3_DEFINITION_REGISTER.csv
docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_LEVEL3_DECISION_RULES.json
docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_POSITIVE_SPOT_LED_EVENT_SET.csv
docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_LEVEL4_CONTROL_CONTRACT.md
```

Verify at least:

```text
definition-register file SHA-256:
    c785a9db740d9c6d984dcdd3ddd2f91ace2012b3dd3d916afd19114677a8c693

decision-rules file SHA-256:
    07e6d8395f666a4a916625a91ff0e9b113a5434ede89474c773a5c4737e335d9

event-set CSV SHA-256:
    4d585f7c4417b47ad54c342dfe20827031a1e5555bcbe9d87bf83b8f98786b6e

Stage 3C resolution contract:
    ce65c62edfb80f5fb83e9b8b6bae1d3eb9c981f8e9a1bcad3b285fdce46cca51

Stage 3C event tape:
    c73344b1bd104c0816d731a1002f729b49100385a34b9b56ec4b2be66dad71ad

Stage 3B source contract:
    25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb

spot manifest:
    3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046

PF cohort:
    768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15
```

## Frozen definitions

Execute exactly:

```text
primary:
    c02_l3_primary_all_1h
    c02_l3_primary_all_6h

robustness_only:
    c02_l3_30m_agreement_1h
    c02_l3_30m_agreement_6h
```

Primary source population: 489 events.  
Robustness population: 425 events.

Definitions run independently. Robustness definitions cannot earn Level-3 permission or rescue a failed primary definition. Do not pool definitions.

## Frozen execution

For each event:

```text
direction:
    long only

decision:
    completed Stage 3C onset-bar availability time

entry:
    first executable Kraken PF five-minute trade-bar open
    strictly after decision

exit:
    first executable Kraken PF five-minute trade-bar open
    at or after entry + 1h or 6h, according to definition

exposure:
    fixed notional

exit type:
    timeout only
```

No stop, trailing exit, partial exit, add, passive fill, leverage optimization, touch fill, or portfolio combination.

## Eligibility and non-overlap

Fail closed for:

- missing next-open entry or exit;
- non-finite or non-positive price;
- pre-2023, 2026+, non-Kraken, or mixed protected file;
- known lifecycle-invalid interval;
- duplicate economic address;
- event/path crossing `2026-01-01`;
- artificial sample-boundary close;
- input/hash mismatch.

For each symbol and definition:

1. sort events chronologically;
2. enter only while flat;
3. ignore later events until the actual timeout exit executes;
4. record every skipped or invalid event and reason.

No event sampling or cap.

## Costs and funding

Use:

```text
base:
    5 bps taker per side
    4 bps round-trip slippage
    14 bps total

stress:
    10 bps taker per side
    12 bps round-trip slippage
    32 bps total
```

Primary Level-3 gates use full-period base fee/slippage net fixed-notional bps, excluding funding.

Compute and report signed funding cashflow separately under:

```text
fully_exact_funded
mixed
fully_imputed
zero_boundary
```

Mixed, imputed, or robustness results cannot rescue a primary gate failure. Do not pool funding partitions as promotion evidence.

## Gate interpretation

Use `C02_LEVEL3_DECISION_RULES.json` and repository-tested shared gate helpers without modification.

For each primary definition, all must pass:

```text
executed trades >= 100
executed trades in 2023, 2024, and 2025 >= 20 each
mean base-net bps > 0
median base-net bps > 0
canonical-episode bootstrap 95% CI lower bound >= -5 bps
maximum positive symbol aggregate / total net <= 25%
maximum positive episode aggregate / total net <= 10%
maximum positive-year aggregate / total positive-year aggregate <= 70%
stress-net mean >= -10 bps
```

Bootstrap:

```text
unit:
    canonical_episode

resamples:
    10,000

seed:
    20260717
```

If a concentration denominator is non-positive or undefined, that gate fails closed.

A primary pass permits consideration of the frozen controls only after another human approval. It is not validation, promotion, or deployment authority.

## Phase A — Implement before real outcome access

Create:

```text
tools/run_kraken_c02_level3_economic.py
unit_tests/test_kraken_c02_level3_economic.py
```

Before opening real post-decision paths, pass synthetic tests covering:

- all four definitions, including zero-trade output rows;
- primary versus robustness permissions;
- exact event-set membership and hash checks;
- next-open entry;
- 1h and 6h timeout exits;
- definition-local actual-exit non-overlap;
- missing/invalid entry and exit;
- lifecycle and protected-boundary rejection;
- no artificial endpoint close;
- fixed-notional gross, base-net, and stress-net bps;
- signed funding and exactness partitions;
- eligibility/skip reconciliation;
- deterministic economic addresses;
- deterministic canonical-episode bootstrap;
- exact concentration formulas and boundary failures;
- robustness cannot rescue primary failure;
- no Level-4 control reader or output.

Run relevant Stage 3A–3D, loader, lifecycle, funding, protected-boundary, archive, and identity regressions.

Obtain an independent pre-run review of the actual diff, schemas, tests, input guards, and exact command. Create one clean runner implementation commit before economic execution. Do not execute unless review is `approve`.

## Phase B — Execute once

Use a fresh UTC-suffixed root and an explicit execution flag:

```bash
./.venv/bin/python tools/run_kraken_c02_level3_economic.py \
  --contract docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_FINAL_LEVEL3_ECONOMIC_CONTRACT.md \
  --definitions docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_LEVEL3_DEFINITION_REGISTER.csv \
  --event-set docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1/C02_POSITIVE_SPOT_LED_EVENT_SET.csv \
  --output-root results/rebaseline/phase_kraken_c02_positive_spot_led_level3_20260717_v1_<UTC_SUFFIX> \
  --execute-economic-run
```

The runner must refuse an existing root and verify every input hash before the first outcome read.

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

For every definition report:

- source, invalid, skipped, and executed counts;
- counts by year and symbol;
- gross, base-net, and stress-net mean and median;
- funding partitions separately;
- bootstrap CI;
- concentration shares;
- every frozen gate result.

The trade ledger must retain event identity, symbol, decision, entry/exit times and prices, actual-exit reason, canonical episode, gross/base/stress bps, funding partition/cashflow, path references, code/config/data hashes, and protected-row count.

## Decision

Return exactly one family decision:

```text
level3_no_primary_pass_stop
```

when neither primary definition passes every gate, or:

```text
level3_primary_pass_controls_pending_separate_approval
```

when at least one primary definition passes every gate.

Do not run Level-4 controls in this task.

## Post-run review, records, and handoff

Independently review:

- runner implementation commit;
- full run root;
- input/hash and protected audits;
- ledgers and reconciliation;
- metrics, gates, funding partitions, bootstrap, and concentration calculations;
- decision wording.

Preserve any failed or partial root as provenance.

After review:

- update the task archive and factual family/multiplicity/continuity records;
- create at most one post-run documentation/registry commit;
- non-force push only under the standing reviewed-task workflow;
- package and upload with `drive_handoff: approved_default`;
- verify remote bytes and SHA-256;
- retain the full local run root and archive.

## Prohibited

- Level-4 controls or matched non-events;
- negative spot-led, perp-led, failure, or shifted-clock branches;
- new thresholds, horizons, exits, stops, filters, or subsets;
- event sampling or caps;
- pooling primary/robustness definitions;
- pooling funding partitions to rescue failure;
- protected-period or capture access;
- holdout, validation, promotion, portfolio, or live claims.

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
