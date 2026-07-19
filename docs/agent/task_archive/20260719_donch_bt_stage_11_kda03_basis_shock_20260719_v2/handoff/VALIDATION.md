# Validation

Status: `KDA03_level3_routes_assigned`.

- Exact base `e841469984478f7436db824587eac46dcd454c6d`, policy SHA-256 `c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa`, and all Stage 8A authorities reconciled.
- The 494,270 Stage 8A masks reconcile as `186,265 + 247,169 + 60,836`; they were adjudication provenance, not economic events.
- Outcome-free generation produced 104,937 unique parent episodes and 113,539 unique events from the 187-symbol authorized cohort. Duplicate episodes/events/economic addresses and protected rows were `0/0/0/0`.
- All six primary branches were mechanically feasible. The final register contains 12 primary and 12 non-rescuing robustness definitions.
- The corrected freeze and an independent replay were byte-identical across contract, definition, event/parent tape, feasibility, count, attempt, and feature-contract artifacts.
- The independent pre-outcome review approved exact contract hash `5f1abdc7e21ab9e3a6851b21930a7af745cbb865be334f8be45c9a561d2411e7` after 21 focused tests and full outcome-free tape checks.
- The one authorized economic run executed at commit `8f365a47625e338b5a518a22866d41e41244e458`: 227,078 schedule rows, 199,787 accepted/priced trades, zero price rejections, and zero duplicate Level-3 addresses.
- Official PF opens were manifest- and content-hash-bound. Entry/exit prices were positive; 14/32 bps cost arithmetic had zero error; protected entries/exits were `0/0`.
- Eleven primary definitions route to `translation_rejected`. Negative completed-basis rejection 6h routes to `sample_limited_prospective_candidate`; it is not control-eligible.
- Controls executed: `0`. Protected rows opened: `0`. Capital.com payloads opened: `0`. KDA01/KDA02 outcomes opened: `0`.

Focused command: `/opt/testerdonch/.venv/bin/python -m unittest unit_tests.test_kda03_v1 unit_tests.test_kda03_level3` — `21 passed`.

The earlier broad repository check completed 87 tests and reported five environment/fixture errors caused by absent historical seal/audit trees in the isolated worktree; none exercised or failed a Stage 11 surface. `git diff --check` passed.
