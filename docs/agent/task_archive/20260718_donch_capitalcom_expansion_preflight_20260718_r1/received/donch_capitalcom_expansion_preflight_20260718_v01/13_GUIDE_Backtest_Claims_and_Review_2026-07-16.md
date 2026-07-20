---
status: current claims and review guide
date: 2026-07-16
revision: 1.0
scope: wording, review gates, and decision vocabulary for Kraken research artifacts
authority: operating contract and test/evidence manual
supersedes: promotion-style labels unsupported by event evidence and ambiguous pass-like review statuses
provenance: testmanual evidence levels; July 16 research audit; external-review protocol findings
known limitations: exact family decisions belong in the merged registry and machine manifests
---

# Backtest Claims and Review Guide

## Claim ladder

| Proposed claim | Minimum evidence | Allowed wording | Disallowed leap |
|---|---|---|---|
| An idea is recorded | Level 0 | “Falsifiable hypothesis; untested.” | “Candidate” or “edge.” |
| Generator is causal | Level 1 plus tests | “Generator passed named causal/identity tests.” | “Profitable setup.” |
| Path supports an entry question | Level 2 | “Path/MAE-MFE evidence supports testing this entry.” | “Trade expectancy.” |
| Train event economics exist | Level 3 | “Event-ledger result on the stated train cohort.” | “Beats controls,” “robust,” or “validated.” |
| Incremental value versus controls | Level 4 | “Under named controls, component X added/did not add value.” | Family-wide or live claim. |
| Train-only stability candidate | Level 5 | “Frozen train-only validation candidate under stated multiplicity and concentration caps.” | “Holdout validated.” |
| Holdout remains sealed | Level 6 | “Candidate frozen; final holdout still sealed.” | Any statement about holdout performance. |
| Execution calibration exists | Level 7 | “Observed calibration/feasibility under the stated protected purpose.” | Directional validation or deployment. |
| Validation-grade | Approved frozen protocol plus authorized independent evidence | State exact period, test, and remaining caps. | Use for any current family; none qualifies. |
| Live-ready | Independent deployment, safety, risk, and human approval | None under current authorization. | Any current report calling a strategy live-ready. |

## Mandatory qualifiers

Every economic claim must include:

- Kraken venue;
- 2023-2025 rankable period;
- PIT universe description and survivor cap;
- event and episode independence;
- cost and funding exactness;
- evidence level;
- reproducibility status;
- multiplicity and concentration status;
- protected-row count;
- current decision and prohibited next action.

## Review order

### 1. Identity and authority

- Is this the current authoritative root?
- Are superseded roots excluded from current decisions?
- Do manifest paths, sizes, and hashes match?
- Does the prose agree with machine status fields?

### 2. Boundary

- Kraken only?
- Rankable rows limited to 2023-2025?
- Protected rows zero in economic outputs?
- Protected files excluded before scoring and selection?
- No paid-vendor or private-data dependency?

### 3. Event mechanics

- Causal signal and decision time?
- Point-in-time membership?
- Actual executable exits used for non-overlap?
- Trade, mark, and funding fields used with correct semantics?
- No touch-fill, synthetic liquidity, or chart-liquidation inference?

### 4. Controls and independence

- Real controls with comparable prospective risk?
- Component ablations?
- Exact and nearby episode overlap separated?
- Definition rows not treated as independent trades?
- Cross-family episode identity present?

### 5. Statistics and selection

- Every attempted variant registered?
- Purge and embargo match the horizon?
- Cluster and symbol concentration disclosed?
- Multiplicity-adjusted result reported?
- Parameter stability shown rather than one optimum?

### 6. Reproducibility

- Code, config, data, universe, funding, and taxonomy hashes?
- Exact commands and environment?
- Test counts, failures, and exit codes?
- Raw verification extracts consistent with release claim?

### 7. Decision language

- Claim does not exceed evidence level?
- Missing data produce a cap, not an invented proxy?
- Negative result is scoped to the exact translation?
- Next action does not reopen a closed same-sample search?

## Red-flag wording

Stop review when a report says any of the following without the required evidence:

- “validated,” “confirmed,” “prelead,” “beats controls,” “stress survives,” or “release-ready” from a filename or summary table;
- “portfolio return” from pooled definition rows;
- “survivorship-free” from the current Kraken roster;
- “exact funding” outside the retained exact slice;
- “liquidation” from a wick, volume spike, or generic trade label;
- “filled” because a candle touched a price;
- “test pass” with blank counts or `source_not_recorded`;
- “historical OI/basis/spread” from current snapshots;
- “prospective validation” when rules changed after observations;
- “live-ready” without deployment authorization.

## Evidence status vocabulary

Use one of:

```text
verified
inferred
proposed
unavailable
blocked
not_applicable
```

Do not merge `verified` and `unavailable` into one pass-like label.

## Decision vocabulary

Use bounded family decisions:

```text
hypothesis_preserved
current_translation_rejected_only
mechanism_preserved_current_translation_weak
feature_or_context_only
sample_limited_mechanism
fragile_frozen_sleeve
candidate_library_only
prospective_only
blocked_by_protocol_issue
```

Then add:

```text
evidence_level
reproducibility_status
validation_status
deployment_status
prohibited_same_sample_action
permitted_next_action
```

## Review outcome template

```text
artifact_or_run:
authoritative_root:
identity_check:
venue_period_boundary:
protected_data_status:
event_and_control_integrity:
overlap_and_multiplicity:
cost_and_execution_status:
reproducibility_status:
evidence_level:
claim_supported:
claim_not_supported:
decision:
required_repairs:
prohibited_next_actions:
human_approval_required:
```

“No issue found” applies only to the checks actually performed. It is not a statement that absent artifacts never existed or that unreviewed layers passed.
