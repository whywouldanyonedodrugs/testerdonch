# Stage 14 execution plan

Task: `donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1`
Base: `e14bbd0d26c14e48a347481f170fcfe8851df625`
Worktree: `/opt/testerdonch-stage14-20260719`
Branch: `agent/stage14-close-derivatives-phase1-20260719`

## Objective and non-goals

Materialize causal contemporaneous Phase-1 evidence for KDA02B, KDA02C, and KDX01; reconcile the fixed cohort and funding authority; benchmark bounded searches; issue one truthful admission per family; and regenerate a non-authorizing Phase 2–5 packet only if a lane is ready.

No post-decision price or cashflow, entry/exit price, forward label, PnL, bootstrap economics, candidate ranking, protected row, Capital.com payload, C17 work, data acquisition, or Phase 2 execution is permitted.

## Authorities and storage

- Read only the hash-verified Stage 8 causal feature cache, registered Stage 8–11 outcome-free identities, lifecycle/cohort manifests, and funding metadata needed for partition coverage.
- Large generated tapes go to `/opt/parquet/kraken_derivatives/analytics/stage14_phase1_v1`; repository outputs contain summaries and a hash manifest.
- The dirty primary checkout is preserved at `/opt/testerdonch-stage14-dirty-recovery-20260719`; all changes occur in the isolated worktree.

## Milestones and gates

1. Verify repository, protocols, Stage 13 hashes, data manifests, cache hashes, schemas, and exact permitted columns. Stop globally on mismatch.
2. Add a fail-closed loader/read-spy and causal episode/breadth/component measurement pipeline with synthetic boundary tests.
3. Materialize complete pre-2026 populations, retention/universe/funding ledgers, summaries, and a deterministic resource benchmark. No sampling or event cap.
4. Decide each family using only Phase-1 gates. Derive bounded search spaces from recorded measurements, not outcomes.
5. If at least one family is ready, regenerate the campaign files using only ready lanes and bind all hashes; otherwise record no packet regeneration.
6. Run deterministic replay, manifest reconciliation, historical-decision comparison, boundary/secret/link checks, and independent adversarial review.
7. Commit, push non-force, and upload one new collision-safe package to the approved Drive root with round-trip SHA-256 verification.

## Failure and rollback

Family-specific measurement or identification failure stops only that family. Shared authority, timestamp, protected-exposure, storage, or replay failure stops the task globally. Source caches and historical archives are immutable; rollback is removal of the isolated branch/worktree after preserving the task archive and local generated-root manifest.
