# Decisions and Progress

## Decisions

- Use user-authorized starting commit `992e7928d0dd948c0bb3f3fc3c74b1095648df1b` on branch `fix/rankable-loader-boundary-20260716`.
- Limit production ownership to `load_symbol_bars()` and `load_funding()` plus their smallest local boundary helpers.
- Use synthetic file authority supplied at the affected loader boundary; do not create a repository-wide catalog or partition framework.
- Treat file-purpose uncertainty as a pre-read error. Treat pre-2023 and non-Kraken rows as downstream-boundary exclusions, not universal pre-open errors.

## Progress

- Verified Git root, branch, clean state, `HEAD`, local `main`, and `origin/main`.
- Read active instructions, repository map, data/protected rules, run contract, review rules, and known failure patterns.
- Located affected call paths in `tools/run_kraken_family_engine_aggregate_first_sweep.py`.
- Verified the archived readiness report SHA-256 and inspected its synthetic defect evidence without opening real market or protected payloads.
- Wrote `PLAN.md` before production or test changes.

Next: preserve the exact task specification, add focused failing synthetic tests, and capture the pre-patch failure.

## Final progress

- Pre-patch reproduction: 6 tests with 14 failures across market and funding paths.
- Post-patch focused tests: 8/8 pass.
- Owning-module tests: 286/286 pass.
- Repository guard tests: 9/9 pass.
- Independent review: approve for bounded patch.
- Readiness rerun: complete but blocked by three sibling reader paths and unchanged protocol/data gaps.
- No economic, protected, capture, acquisition, network, push, or merge action occurred.
