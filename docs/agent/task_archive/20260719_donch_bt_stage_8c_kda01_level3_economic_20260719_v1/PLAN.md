# Stage 8C KDA01 v2 Frozen Level-3 Economic Execution

Status: complete
Owner: backtesting agent
Created UTC: 2026-07-19
Repository root and commit: `/opt/testerdonch`, `55c75ef0564b004413c19d670625e44a1838a537`

## Objective
Implement and execute once the exact 16-definition KDA01 Level-3 v2 contract, without controls or protected data, and emit one authorized terminal decision.

## Non-goals
No controls, tuning, alternative definitions, Level 4, validation, Capital.com, protected-period, portfolio, acquisition, or other-family work.

## Authority and gates
- Contract hash: `d2112d8fb4f0666890761b70f7fa71c9e8cb4f43c704a04cdf9470e949a177b3`.
- Contract/register/cluster hashes must equal the task specification before price access.
- Schedule must reconstruct exactly `204272 / 183744 / 20528`, including `20473` overlap and `55` missing-exit records.
- Fills use exact official PF 5m trade-bar `open`; funding is diagnostic and excluded from gates.
- Train interval is `[2023-01-01, 2026-01-01)`; protected rows opened must remain zero.

## Repository preservation
- Main checkout had zero staged/unstaged/conflicted and three unrelated untracked paths; content was not copied or modified.
- Isolated worktree: `/opt/testerdonch-stage8c-20260719`.
- Branch: `agent/stage8c-kda01-level3-economic-20260719`.
- Base: `55c75ef0564b004413c19d670625e44a1838a537`.

## Milestones
1. Verify authority and write synthetic tests. Stop on any hash/schema mismatch.
2. Implement exact schedule reconstruction, open lookup, returns, bootstrap, concentration, funding diagnostics, and manifests.
3. Run focused/relevant tests and pre-price mechanical reconstruction.
4. Execute once, then deterministic replay without changing the contract.
5. Independently review code, outputs, gates, claims, protected counts, and package.
6. Commit, non-force publish, and round-trip the approved compact Drive handoff.

## Expected code
- `tools/qlmg_kda01_level3_economic.py`
- `tools/run_kda01_level3_economic.py`
- `unit_tests/test_kda01_level3_economic.py`

## Failure response
Preserve the new run root and return `blocked_with_exact_mechanical_remedy`; do not modify thresholds or inspect another hypothesis.
