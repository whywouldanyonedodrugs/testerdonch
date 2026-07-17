# Execution Plan

Task: `donch_bt_stage_2c1_volume_authority_resume_20260717_v1`
Class: bounded data-authority acquisition plus non-economic C01 identity/contract work.
Starting commit: `9b4fbe1bcf9e8a0f79fedde758534dc0d7c86611`.
Branch: `feature/stage-2c1-volume-authority-resume-20260717`.

## Objective
Establish or reject official base-volume semantics using exact archived sources and a frozen 4-symbol/12-interval current calibration. Resume Stage 2C only on a complete pass, producing causal liquidity-cohort and onset identities without outcomes.

## Frozen calibration
Symbols are fixed before calibration requests: `PF_XBTUSD`, `PF_ETHUSD`, `PF_XRPUSD` (ordinary alt with integer base-unit precision), and `PF_AAVEUSD` (fractional alt with 0.01 base-unit precision). `PF_DOGEUSD` was replaced before calibration because the official instrument snapshot confirmed integer rather than fractional sizing. The frozen UTC intervals are `[2026-07-17T08:30:00Z,08:35:00Z)`, `[08:35:00Z,08:40:00Z)`, and `[08:40:00Z,08:45:00Z)`. They were fixed before interval data were requested and were not selected by price behavior.

## Hard boundaries
No candidate outcomes, forward returns, controls, economics, protected strategy inspection, capture, threshold changes, or new strategy joins. Current calibration fields are restricted to symbol, trade timestamp/size, trade price needed for interval identity, candle timestamp/volume, and source identity.

## Milestones
1. [completed] Verify source and endpoint identity; archive exact bytes, headers, UTC time, URL, and hashes.
2. [completed] Freeze completed intervals and compare complete paginated trade-size sums to candle volume with exact decimals.
3. [completed] Audit historical contract semantics; fail closed on ambiguity.
4. [completed] Only after authority passes, implement clarified Stage 2C proxy, onset, identity, agreement, branch, and draft-contract outputs.
5. [completed] Run focused and guard tests, independent review, manifest/secret checks, then authorized commit/push and reduced Drive handoff.

## Failure response
If any interval differs materially, pagination/boundaries are incomplete, or historical base-unit semantics remain ambiguous, preserve authority evidence and stop without C01 implementation changes.
