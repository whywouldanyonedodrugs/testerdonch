# Commands and Results

- Source-only timestamp audit: defect confirmed before repaired outcomes.
- Original Stage 8C independent recomputation: 16/16 rows passed within 1e-10.
- Timestamp-only repaired schedule: deterministic `204272/183744/20528`; overlap/missing `20473/55`; protected/duplicates `0/0`.
- Pre-outcome review: approved contract `2806740e...` with no blocking findings.
- Single repaired execution: exit 0 in 4m25.19s; peak RSS 2,592,736 KiB; no swap.
- Post-run independent recomputation: maximum metric difference 7.11e-15; gate mismatches 0.
- Focused tests: 30/30 passed. Full relevant tests: initial 175 pass/4 environment errors, then 179/179 passed after linking two repository-ignored metadata fixtures.
