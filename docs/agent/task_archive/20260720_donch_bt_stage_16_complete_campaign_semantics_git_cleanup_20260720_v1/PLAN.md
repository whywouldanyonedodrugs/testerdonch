# Stage 16 Complete Campaign Semantics Plan

Status: in_progress

## Objective

Freeze a launch-complete but non-authorizing derivatives campaign contract, prove it with synthetic-only tests and independent review, preserve prior packets unchanged, close Git state safely, and hand off a verified archive.

## Authority and boundaries

- Base: `8b8e4b15c0bc89d68a0748c2e26823e024a0279b`.
- Venue: Kraken linear PF derivatives only.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Economic run: not authorized.
- Forward outcomes, protected outcomes, Capital.com payloads, acquisition, and real Telegram: forbidden.
- Remote writes: non-force Git push and unique approved-default Drive folder only.

## Repository preservation

- Original checkout was staged-dirty at `baaa10c224807e1dc7e32bfee7227711cb0c1279`.
- External recovery bundle: `/opt/testerdonch-stage16-dirty-recovery-20260720`.
- Preserved commit/branch: `7397b9c` on `preservation/stage16-original-checkout-20260720`.
- Isolated worktree: `/opt/testerdonch-stage16-20260720`.
- Task branch: `agent/stage16-complete-campaign-semantics-20260720`.

## Milestones

1. Verify authority, preserve original checkout, and establish clean isolated base.
2. Implement exact bin, fold, metric, Pareto, beam, translation, boundary, funding, and supervision contracts.
3. Add fail-closed validator, read-spy, and synthetic end-to-end canary.
4. Obtain independent adversarial review and repair findings before commit.
5. Commit semantic implementation; bind and archive packet in a separate commit.
6. Push non-force, update `origin/main`, build and round-trip verify Drive handoff, record final commit.

## Rollback

All new work is isolated on a task branch. Prior Stage 14/15 artifacts are immutable. Original pending work is independently recoverable from the pushed preservation commit and external bundle. No reset, clean, force push, overwrite, or deletion is permitted.
