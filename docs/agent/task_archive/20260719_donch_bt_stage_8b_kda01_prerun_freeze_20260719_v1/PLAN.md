# Stage 8B - KDA01 Mechanism Adjudication and Pre-Run Freeze

Status: in_progress
Owner: backtesting agent
Created UTC: 2026-07-19
Repository root and commit: `/opt/testerdonch-stage8b-20260719` at `41b64b52a9146669eb26dcf25a86523a35219b8d`

## Objective

Replace broad Stage 8A KDA01 attempts with a causal, episode-level, outcome-free KDA01 v2 tape and freeze an approval-ready Level-3 economic contract only for branches passing the predeclared mechanical gates.

## Non-goals

- No candidate returns, outcomes, PnL, MAE/MFE, economic ranking, or control outcomes.
- No KDA02/KDA03 changes, threshold adaptation, new acquisition, protected data, capture, Capital.com, account, or order access.
- No economic runner implementation or execution.

## Authority and boundaries

- Base and Stage 8A published commit: `41b64b52a9146669eb26dcf25a86523a35219b8d`.
- Kraken train interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Stage 8A cache: `/opt/parquet/kraken_derivatives/analytics/stage8a_foundation_v1_exact`.
- Economic run authorized: no.
- Drive handoff: approved default, unique non-overwriting folder with round-trip verification.

## Repository preservation

- Main checkout had zero staged/unstaged/conflicted paths and three known pre-existing untracked paths.
- Main checkout remains untouched.
- Isolated branch: `agent/stage8b-kda01-prerun-freeze-20260719`.
- Isolated worktree: `/opt/testerdonch-stage8b-20260719`.

## Milestones

1. Verify Stage 8A machine hashes, tape hash, cache, and protected boundary.
2. Implement causal OI and price-progress normalization plus KDA01 v2 episode/event generator.
3. Add focused deterministic PIT/identity/episode tests.
4. Generate complete outcome-free tapes, counts, feasibility gates, and frozen contracts.
5. Run regressions and independent review; fail closed on any claim mismatch.
6. Commit, non-force push, create task archive, and perform approved Drive handoff.

## Failure response

- Authority/hash mismatch: stop before generation.
- Any protected/outcome column or row: fail closed and omit approval packet.
- No feasible primary branch: register failures, omit approval packet, return `KDA01_mechanically_unavailable`.
- Test/review failure: preserve artifacts and return `blocked_with_exact_non_economic_remedy`.
