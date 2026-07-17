# Stage 3C — C02 Alignment Adjudication

```text
task_id: donch_bt_stage_3c_c02_alignment_adjudication_20260717_v1
repository: /opt/testerdonch
mode: direct_apply
drive_handoff: approved_default
economic_run_authorized: no
candidate_returns_authorized: no
protected_outcome_access: no
new_market_data_acquisition: no
```

## Objective

Determine whether C02 leadership can be represented honestly with the existing five-minute Kraken spot/PF data.

Replace the unstable one-bar ordering label with a resolution-aware, interval-censored classification. Produce counts and a frozen generator-contract recommendation only.

Do not calculate returns, PnL, exits, MAE/MFE, control performance, or economic rankings.

## Accepted evidence

```text
Stage 3B task:
    donch_bt_stage_3b_c02_leadership_generator_20260717_v1

generator contract:
    25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb

spot manifest:
    3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046

Stage 3B status:
    alignment_fragile_requires_review
```

Preserve the original Stage 3B lineage and its failed alignment gate. Do not rewrite it as a pass.

## Start check

Verify clean synchronized `main`, the applicable `AGENTS.md` chain, and all input hashes.

Reconcile this handoff identity discrepancy before changes:

```text
agent summary local commit:
    408b9d99686a5506a0ee7e6f42074d56bcfa8cfb

Drive v02 transfer-manifest commit:
    64c65e2bed2e5da38cb808d185cd64298d4ca3ff
```

Proceed only if the latter is an expected handoff/archive-only descendant or the repository evidence otherwise resolves the lineage. Record the actual starting commit and resolution.

## Adjudication

### 1. Clock semantics

Verify and document from source and implementation:

- normalized spot and PF timestamps denote UTC five-minute interval opens;
- each value becomes available only at interval close;
- exact UTC alignment is the authoritative grid;
- ±5-minute shifts are perturbation diagnostics, not candidate alignments;
- sparse spot intervals remain missing and are never filled.

### 2. Explain the frozen failure

Produce transition matrices for exact versus `-5m` and `+5m` classifications, by exact leadership state.

Separate:

- event disappearance or episode/direction change;
- simultaneous becoming spot-led under `-5m`;
- simultaneous becoming perp-led under `+5m`;
- genuine leader reversals;
- ambiguous cases.

No outcome data may be read.

### 3. Resolution-aware leadership

Keep the original impulse, z-score, reset, eligibility, cohort, and lifecycle rules unchanged.

Treat each first threshold crossing as interval-censored to its observed five-minute bar.

Primary 15-minute classification:

```text
resolved_spot_led:
    spot first-crossing bar open is at least 10 minutes
    earlier than the perp first-crossing bar open

resolved_perp_led:
    perp first-crossing bar open is at least 10 minutes
    earlier than the spot first-crossing bar open

coincident_or_unresolved:
    all other cases, including same-bar, five-minute
    separation, missing first crossing, or overlap
```

The 10-minute bar-open separation is not a performance threshold. It is the minimum that guarantees non-overlapping crossing intervals and at least a bounded five-minute lead.

Apply the same rule with the frozen 30-minute lookback as robustness. Do not choose a lookback from counts.

Completed failure is retained only for `resolved_perp_led`.

Do not use shifted data to create or select events.

### 4. Counts and feasibility gates

Report counts by year, direction, symbol, BTC/ETH versus alt, primary/robustness state, and failure confirmation.

A branch is mechanically sufficient for later contract review only if it has:

```text
at least 100 primary resolved events total
at least 20 primary resolved events in each of 2023, 2024, and 2025
same leader under the 30-minute robustness lookback for at least 80%
```

These are feasibility gates only. They do not imply edge.

## Required outputs

```text
C02_CLOCK_SEMANTICS_REPORT.md
C02_ORIGINAL_ALIGNMENT_TRANSITION_MATRIX.csv
C02_RESOLUTION_AWARE_GENERATOR_CONTRACT.md
C02_RESOLUTION_AWARE_EVENT_TAPE.parquet
C02_RESOLUTION_AWARE_COUNT_MATRIX.csv
C02_15M_30M_AGREEMENT.csv
C02_ALIGNMENT_ADJUDICATION.md
VALIDATION.md
REVIEW.md
ARTIFACT_MANIFEST.json
COMPLETION.md
NEXT_ACTION.md
```

No output may contain post-decision outcomes.

## Tests

Use synthetic fixtures for:

- interval-open and close-availability semantics;
- sparse gaps remain missing;
- exact 10-minute boundary;
- same-bar and five-minute-separated crossings remain unresolved;
- 15m/30m classifications;
- completed failure only after resolved perp leadership;
- deterministic identities;
- no future, protected, non-Kraken, lifecycle-invalid, or outcome rows.

Run relevant Stage 3A/3B and protected-boundary regressions. Require independent review.

## Decision

Return exactly one:

```text
ready_for_C02_resolution_aware_contract_review
C02_5m_leadership_unavailable
blocked_with_exact_non_economic_remedy
```

If five-minute leadership is unavailable, recommend—but do not perform—one bounded one-minute PF-data feasibility pilot. Do not select a clock shift, change impulse thresholds, or run economics.

## Integration and handoff

After review:

- create task-scoped commit(s);
- non-force push under the standing reviewed-task workflow;
- upload with `drive_handoff: approved_default`;
- omit large local tapes and raw market data from Drive;
- retain local copies and hashes.

## Final response

```text
status:
actual_starting_commit:
handoff_commit_discrepancy_resolution:
clock_semantics:
original_transition_summary:
resolved_event_counts:
counts_by_year_direction_state:
15m_30m_agreement:
failure_counts:
tests_and_review:
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
