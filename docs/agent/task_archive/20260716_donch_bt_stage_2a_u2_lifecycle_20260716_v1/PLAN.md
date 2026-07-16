# Stage 2A U2 Lifecycle Authority and Anchor Cohort

Status: in_progress
Owner: backtesting Codex
Created UTC: 2026-07-16
Repository: `/opt/testerdonch`
Starting commit: `c198cf0059128b2beb917eda19fbfde6695ed7a9`
Branch: `data/u2-lifecycle-20260716`

## Objective

Build a narrow, outcome-free, official-Kraken lifecycle authority sufficient to admit or reject a continuously eligible U2 anchor cohort for `[2023-01-01, 2026-01-01)`.

## Frozen selection rule

A contract is included only when stable Kraken identity, an eligible start, no unresolved lifecycle interruption in the claimed interval, complete official trade/mark metadata coverage, and protected-data independence are all verified. Unknown state excludes; current roster and bar existence alone do not prove eligibility.

## Non-goals

No C01/C02/C03 implementation, economic output, candidate return, protected payload, capture, paid source, complete historical universe reconstruction, funding repair, package repair, or generalized lifecycle platform.

## Expected changes

- `tools/build_kraken_u2_lifecycle_authority.py`: bounded deterministic normalization and validation only, if source parsing requires code.
- `unit_tests/test_kraken_u2_lifecycle_authority.py`: synthetic identity/interval/hash fixtures.
- This archive: cached official sources, seven required outputs, plan/commands/review/completion/manifest.
- Current lifecycle/data-capability authority record: factual update only if acquired evidence warrants one.

## Milestones

1. Inspect authorized local instrument and trade/mark manifest metadata without outcomes.
2. Acquire only bounded official Kraken lifecycle sources for a narrow considered set; cache URL/status/time/hash.
3. Add failing synthetic parser/interval tests, then the smallest deterministic tool.
4. Generate lifecycle, inclusion/exclusion, coverage, and claim-boundary artifacts.
5. Run focused and repository guard tests; independent diff/source review.
6. If all gates pass and main remains unchanged, create one commit, fast-forward main, and push without force.

## Safety

- Public network: official Kraken and official-page archives only.
- Protected outcome payload reads: forbidden.
- Current 2026 instrument metadata: identity/status evidence only.
- Rollback: non-destructive revert of the task commit; raw/cache artifacts remain traceable.
