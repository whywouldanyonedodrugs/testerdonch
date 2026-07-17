# Execution Plan

## Objective
Establish official Kraken USD spot-reference authority for C02 and, only after a synthetic/pilot/review gate passes, acquire deterministic rankable 5-minute bars for defensibly mapped Stage 2C mechanism-proof PF assets. No signals, outcomes, lead/lag labels, thresholds, or economic analysis.

## Authorities
- Task specification: `TASK_SPEC.md`.
- Stage 2C cohort: `docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1/C01_MECHANISM_PROOF_COHORT.csv`.
- Protected interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Sources: official Kraken downloadable historical market data, Spot API metadata, and official documentation only.
- Starting commit: `b4e6c24c7c1f6c54dea931d824383590d087b819`.

## Milestones
1. Resolve official source URLs and pair metadata; archive source-response metadata and hashes without opening price payloads.
2. Implement a minimal acquisition/normalization authority tool and synthetic tests for identity, boundaries, aggregation, gaps, deterministic hashes, and unknown lifecycle.
3. Freeze XBT, ETH, and the first two eligible additional canonical assets before pilot price reads.
4. Acquire and validate only the three frozen pilot windows. Stop on identity, protected-boundary, schema, duplicate/order, or source-authority failure.
5. Obtain independent read-only pilot review. Only on approval, acquire rankable source files and normalize eligible mapped pairs with explicit bounded coverage masks.
6. Produce required authority artifacts, independent final review, archive, commit, non-force push, and approved Drive handoff excluding raw/normalized market payloads.

## Safety and failure response
- Archive/container metadata must establish an eligible official file before member payload reads.
- Protected or mixed/unrankable members fail closed; no 2026 trade row may be opened.
- Unknown historical pair intervals remain unknown and masked.
- No current pair listing proves uninterrupted historical availability.
- No economic output is permitted.
- Existing artifacts are never overwritten or deleted; data writes are atomic and content-addressed/manifested.
- On pilot failure, emit `C02_historical_data_path_unavailable` or `blocked_with_exact_data_remedy` and stop before full acquisition.

## Rollback
Revert task commits normally without force push. Preserve downloaded official files, manifests, and task archive as provenance; do not delete remote handoffs.
