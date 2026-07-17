# Execution Plan

## Objective
Build and independently review a deterministic, causal, outcome-free generator for `C02_spot_led_vs_perp_led_impulse`, using only the accepted Stage 3A sparse spot panel and existing authorized PF/lifecycle/cohort inputs. No returns after decision, PnL, controls, thresholds selected from counts, protected outcomes, capture, or market-data acquisition.

## Authority
- Starting commit: `2a83432a5ecd94b284b3a9c8f6366e4e0ae8df1f`.
- Spot manifest content hash: `3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046`.
- Spot alignment contract and pair authority: Stage 3A task archive.
- PF cohort/liquidity and lifecycle authority: Stage 2C/2C1 finalized artifacts, subject to exact hash verification before generation.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.

## Milestones
1. Resolve exact PF trade/mark, lifecycle, daily liquidity, identity, and nearest-family causal-ledger paths. Fail closed on ambiguity or hash mismatch.
2. Freeze family, attempt register, generator contract, feature schema, and multiplicity before reading generator counts.
3. Implement minimal sparse alignment, prior-day eligibility/scale, onset, leadership, completed-failure, shifted sensitivity, deterministic identity, and episode utilities with synthetic tests.
4. Run the complete non-economic generator only after mechanical tests pass. Write immutable local Parquet outputs and compact tracked summaries/manifests; compute no post-decision outcome.
5. Independently verify temporal semantics, deterministic replay, hashes, prohibited fields, boundaries, identity, counts, and overlap preflight.
6. Commit, non-force push, and produce the approved default Drive handoff excluding large Parquet/raw payloads.

## Completion State

- Milestones 1-5: completed.
- Milestone 6: in progress pending commit, push, and round-trip Drive verification.

## Failure response
Stop with `blocked_with_exact_non_economic_remedy` on unresolved input authority, mixed/protected payload, unit ambiguity, lifecycle ambiguity that cannot be masked, temporal leak, identity collision, nondeterminism, or review failure. Preserve old roots and all completed task evidence.

## Rollback
Revert task commits normally without force push. Never delete data or historical roots. Keep local generated artifacts hash-manifested for audit.
