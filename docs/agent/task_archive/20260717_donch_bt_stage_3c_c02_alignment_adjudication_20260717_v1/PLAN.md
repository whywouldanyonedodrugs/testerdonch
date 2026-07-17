# Execution Plan

## Objective
Adjudicate whether the frozen C02 mechanism can represent spot/perpetual leadership honestly at five-minute resolution by replacing unstable one-bar ordering labels with a fixed interval-censored classification. Produce counts and a contract recommendation only.

## Authority and start state
- Starting commit: `408b9d99686a5506a0ee7e6f42074d56bcfa8cfb`; clean and synchronized with `origin/main`.
- Stage 3B contract hash: `25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb`.
- Stage 3A spot manifest hash: `3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046`.
- Handoff discrepancy: `64c65e2...` is an ancestor implementation commit of final handoff commit `408b9d9...`; no divergent evidence lineage.
- Rankable data interval remains `[2023-01-01, 2026-01-01)`; protected outcomes remain unopened.

## Non-goals and forbidden actions
No returns, PnL, exits, MAE/MFE, controls, economic ranking, threshold selection from counts, new data acquisition, clock-shift selection, or protected/capture access.

## Milestones
1. Verify source hashes, Stage 3B tape hashes, timestamp semantics, and safe schemas.
2. Implement a minimal outcome-free adjudicator that reconstructs exact-grid crossings and validates frozen Stage 3B identities.
3. Add synthetic tests for interval censoring, sparse gaps, failure eligibility, temporal boundaries, and deterministic identity.
4. Run the complete adjudication, write the required tape/count/transition/agreement artifacts, and evaluate only frozen feasibility gates.
5. Independently review temporal semantics, schemas, hashes, determinism, counts, and prohibited fields.
6. Commit, non-force push, and create the approved non-overwriting Drive handoff excluding Parquet/raw payloads.

## Failure response
Stop with `blocked_with_exact_non_economic_remedy` on hash mismatch, reconstructed Stage 3B label mismatch, protected/outcome field access, nondeterminism, temporal ambiguity, or review failure.

## Rollback
Revert task commits normally. Preserve Stage 3B and all generated local artifacts; never delete or rewrite finalized roots.
