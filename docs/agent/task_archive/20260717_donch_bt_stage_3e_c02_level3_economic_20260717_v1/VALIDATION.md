# Validation

Status: `pass`.

- Frozen contract/register/rules/event hashes: pass before outcome access.
- Compile: pass.
- Focused and relevant regression suite: 58 tests passed, 0 failures/errors.
- Four definitions and exact frozen event populations were enforced.
- Eligibility reconciliation: 1,828 = 1,806 executed + 22 actual-overlap skips + 0 invalid.
- Duplicate economic addresses: 0.
- Funding boundary count and cashflow nulls: 0; report partitions reconcile to all 1,806 trades.
- Protected rows opened: 0; artificial endpoint exits: 0.
- Independent metric/gate recomputation mismatches: 0.
- Run-root artifact hash mismatches: 0 across 16 finalized artifacts.
