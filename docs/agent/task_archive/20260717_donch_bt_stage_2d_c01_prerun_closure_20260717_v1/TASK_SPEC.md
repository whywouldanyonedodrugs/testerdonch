# Stage 2D — C01 Pre-Run Closure

```text
task_id: donch_bt_stage_2d_c01_prerun_closure_20260717_v1
target_environment: backtesting Codex
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
```

## Objective

Produce one approval-ready C01 Level-3 economic-run contract and close the legacy external-review package blocker by accurately narrowing its claim.

Do not compute C01 outcomes.

## Accepted C01 authority

```text
repository lineage commit:
    45d92488a41fb97a9a30936075c19581f358357d

family:
    C01_debetaed_residual_shock_path_bifurcation

generator contract hash:
    3464e79a79956c881c7418840068a61e3f3a47776a5a4d3a669e98df124fd970

economic draft hash:
    f1c8c612ea9f7ffcc2abad3f2efde36b5dfb68fde20d2769fdc5ce40ab306c13

feature contract hash:
    c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb

cohort hash:
    768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15

reference panel hash:
    2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763
```

Stage 2C1 facts:

```text
accepted onset rows: 38,347
canonical episodes: 11,343
primary BTC+ETH onsets: 19,276
primary economic-branch onsets before entry/confirmation: 2,288
protected outcomes opened: zero
economic outputs computed: zero
```

The candidate cohort remains `current_roster_bar_existence_cohort`; no survivorship-free or continuous-tradeability claim is allowed.

## Part A — External-review package disposition

The package currently has unresolved historical gaps:

- raw trade/mark/funding verification extracts absent;
- historical index/spot source unavailable;
- one prior-high source snapshot absent;
- test counts absent for some prior lineages;
- reproducibility hashes absent for five lineages;
- TSMOM path coverage incomplete;
- full cross-family causal episode starts unavailable.

Do not regenerate economics or inspect protected outcomes.

Create a machine-readable and human-readable disposition that:

```text
protocol_disposition:
    closed_by_claim_narrowing

package_role:
    strategic_and_continuity_review_only

package_release_ready_for_independent_reproduction:
    false
```

Every missing item must be `unavailable`, `irrecoverable`, or `deferred_with_exact_task`; never silently pass it.

The narrowed package may support:

- research history;
- supersession and negative-decision continuity;
- broad multiplicity context;
- identification of nearest prior families.

It may not support:

- independent row-level reproduction;
- validation-grade claims;
- proof that all old tests ran;
- exact cross-family episode overlap;
- promotion of C01 or any prior family.

Update the smallest authoritative package/readiness records necessary so that `blocked_by_protocol_issue` is no longer an ambiguous open workflow blocker. Preserve `release_ready=false`.

If current machine authority requires actual missing payload generation rather than claim narrowing, stop and identify that exact contract. Do not broaden the task.

## Part B — Final C01 Level-3 contract

Create a new final contract hash. Do not modify generator, feature, cohort, reference-panel, onset, or path thresholds.

### Definitions and multiplicity

Primary model:

```text
btc_eth_ols_daily_v1
```

Economic branches:

```text
positive_smooth_long
negative_smooth_short
positive_jump_completed_failure_short
negative_jump_completed_failure_long
```

Intermediate states remain diagnostics only.

Timeout definitions:

```text
6h primary
24h robustness
```

The eight primary-model branch × timeout combinations are eight economic definitions and must all be registered, including zero-trade definitions.

BTC-only versions are robustness definitions only and cannot rescue a failed primary definition. Register all eight separately.

Total planned economic definitions:

```text
16
```

All controls/ablations are additional registered attempts when later run.

### Entry and confirmation

Preserve the Stage 2C draft rules exactly.

For jump branches:

- confirmation must occur within 24h after onset;
- confirmation and dominant-bar identity are causal and deterministic;
- no confirmation means no trade;
- entry is next executable trade-bar open.

### Outcome boundary

A trade is eligible only when every bar needed for:

- confirmation;
- entry;
- stop monitoring;
- the selected timeout;
- funding accounting;
- next-open execution

lies wholly inside `[2023-01-01, 2026-01-01)`.

Do not read a mixed 2025/2026 payload. Do not artificially close a trade at the sample boundary.

### Non-overlap

Run each economic definition independently.

Within each symbol and definition:

1. process onsets chronologically;
2. open only while flat;
3. ignore later onsets while the position remains active;
4. eligibility resumes only after the actual executed exit.

Do not use nominal maximum-hold preblocking. Do not combine definitions into a portfolio.

### Execution and invalid rows

Specify fail-closed handling for:

- missing next-open bar;
- missing mark bar during stop monitoring;
- zero or negative structural stop distance;
- stop already breached before entry;
- non-finite price or funding;
- known lifecycle-invalid interval;
- duplicate economic address;
- same-bar ambiguity.

No touch fills, partial fills, adds, leverage optimization, or artificial boundary closes.

### Outcome units

Use fixed notional exposure. Do not size by structural stop distance.

Primary report units:

```text
net return on fixed notional
basis points
```

Structural `R` may be reported only as a diagnostic. It cannot be the sole promotion metric, and denominator distributions/extremes must be disclosed. Do not introduce a post-outcome risk floor.

### Costs and funding hierarchy

Costs remain:

```text
base:
    5 bps taker per side
    4 bps round-trip slippage

stress:
    10 bps taker per side
    12 bps round-trip slippage
```

Freeze the evidence hierarchy:

1. full-period gross and fee/slippage-net results: kill-screen evidence only;
2. fully exact-funded rows: primary funding-valid subset, reported by calendar period;
3. mixed/imputed-funded full-period results: sensitivity only, never promotion evidence;
4. zero-boundary rows: report separately when present.

No pooled funding partition may rescue a definition.

### Level-3 decision rule

Level 3 cannot claim incremental value because controls have not run.

A primary definition may proceed to Level 4 only when all are true under base costs:

```text
executed trades >= 100
trades in each calendar year >= 20
positive fixed-notional mean net return
positive fixed-notional median net return
bootstrap 95% CI lower bound for mean is not below -5 bps
no single symbol contributes > 25% of aggregate net PnL
no single canonical episode contributes > 10% of aggregate net PnL
no single calendar year contributes > 70% of aggregate positive net PnL
stress-cost mean is not worse than -10 bps
```

Use canonical-episode cluster bootstrap with deterministic seed and a frozen resample count of 10,000.

These are **permission-to-test-controls gates**, not validation or promotion gates.

If no primary definition passes, C01 stops at Level 3. BTC-only robustness or secondary branches cannot rescue it.

### Level-4 controls, only for a passing primary definition

Pre-register, but do not execute in this task:

1. raw 6h USD shock without residualization;
2. residual shock without path separation;
3. raw-return path classification at the same causal timestamps;
4. matched non-events;
5. BTC-only residual robustness.

Matched non-event contract:

```text
ratio:
    1 control per event

same:
    symbol
    calendar year
    direction

calipers:
    lagged 24h volatility within 20%
    absolute BTC 6h return within 50 bps
    absolute ETH 6h return within 50 bps

time exclusion:
    not inside any same-symbol C01 canonical episode
    at least 48h from the event onset

selection:
    nearest deterministic match
    tie break by timestamp
```

If no match exists, mark unavailable; do not widen calipers after outcomes.

### Claim boundary

A passing Level-3 result may be described only as:

```text
train-period event-ledger economics sufficient to justify
predeclared controls under current-roster, funding, execution,
multiplicity, and lifecycle caps
```

It is not validation, robustness, or live readiness.

## Required outputs

```text
EXTERNAL_REVIEW_PACKAGE_DISPOSITION.md
EXTERNAL_REVIEW_PACKAGE_DISPOSITION.json
C01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md
C01_LEVEL3_DEFINITION_REGISTER.csv
C01_LEVEL3_DECISION_RULES.json
C01_LEVEL4_CONTROL_CONTRACT.md
C01_PRERUN_APPROVAL_PACKET.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

`C01_PRERUN_APPROVAL_PACKET.md` must give the exact final contract hash, input hashes, authorized command to be used later, expected run root, rollback, and prohibited actions.

Do not execute that command.

## Tests and review

Use synthetic fixtures to verify:

- 16 economic definitions retained;
- actual-exit non-overlap;
- protected-boundary exclusion;
- no endpoint artificial close;
- fixed-notional arithmetic;
- funding partitions remain separate;
- concentration and cluster-bootstrap rules;
- BTC-only cannot rescue primary failure;
- matched-control calipers never widen;
- package claim narrowing does not convert unavailable evidence to pass.

Run focused tests and repository guard tests. Require independent review of actual diffs and artifacts.

## Integration and Drive handoff

After tests and review pass:

- create one task-scoped commit;
- integrate and non-force push under standing reviewed-task workflow;
- upload with `drive_handoff: approved_default`;
- verify remote size and SHA-256;
- keep local archive.

## Final status

Return exactly one:

```text
ready_for_human_C01_Level3_run_approval
```

or:

```text
blocked_with_exact_non_economic_remedy
```

Do not run economics.

## Final response

```text
status:
actual_starting_commit:
package_protocol_disposition:
package_release_ready:
final_C01_contract_hash:
economic_definition_count:
primary_definition_count:
robustness_definition_count:
funding_evidence_hierarchy:
Level3_gates:
Level4_control_contract:
tests_and_results:
protected_outcomes_opened: no
economic_outputs_computed: no
local_commit:
origin_main_updated:
task_archive:
Drive_handoff_path:
Drive_handoff_verification:
remaining_limits:
next_status:
human_approval_required:
```
