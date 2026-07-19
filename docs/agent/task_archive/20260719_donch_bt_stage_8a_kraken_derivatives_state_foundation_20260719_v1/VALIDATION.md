# Validation

Status: `pass`.

## Mechanical and lineage gates

- Stage 7C manifest content hash: `f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d`.
- Semantic contract hash: `289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60`.
- Shared feature contract hash: `4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4`.
- Generator contract hash: `c285af918e317722e91e3c1df5e44eb48abf27c97fedfe251b0881e3aa8f0017`.
- Cohort hash: `5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636`.
- Feature partitions: `187/187`, all content hashes verified.
- KDA01 event IDs/economic addresses: `2,639,115 / 2,639,115`; duplicates `0/0`.
- KDA02 event IDs/economic addresses: `1,252,232 / 1,252,232`; duplicates `0/0`.
- KDA03 feasibility rows: `494,270`.
- Registered attempts: `15`; one semantic duplicate retained as an explicitly killed zero-tape branch.
- Pre-2023 events: `0`.
- Protected events: `0`.
- Outcome columns/readers/economic outputs: `0/0/0`.
- BTC/ETH one-minute-to-five-minute comparisons: `1,842,048` exact overlapping metric-symbol observations across six cells; aggregation mismatches `0`. Exact per-cell counts are in `KDA_ONE_MINUTE_TIMING_DIAGNOSTICS.csv`.

## Deterministic replay

A cache-only independent replay reproduced both final event-tape bytes exactly:

- KDA01 SHA-256: `583c1f940f185cf01417a1f5ba6540c6a1b6545c0532851d1dabf200d8c874ce`.
- KDA02 SHA-256: `c4d553267e2107beeb042bc22f3280013c41a962d1674b4942d2b7f0de5e2b43`.

## Tests and resources

- Focused Stage 8A tests: `22 passed`.
- Relevant loader, sealed-boundary, lifecycle, C01/C02, and analytics suite: `176 passed`.
- Compilation: passed.
- Documentation links: passed.
- `git diff --check`: passed.
- Authoritative runtime: `1,402.83 seconds` inside the generator, `23m32.87s` process wall time.
- Whole-run peak RSS: `2.667 GiB` (`2,796,504 KiB` from `/usr/bin/time`), below the 4 GiB gate.
- Swap activity: `0`.

## Coverage limits

Exact five-minute trade/mark/analytics intersections begin `2023-03-07T11:40:00Z`, reflecting OI retention truncation. The cohort is current-roster capped with PIT prior-day liquidity and known lifecycle masks; unknown lifecycle remains a claim cap. Liquidation side is price-inferred only. No economic inference is authorized.
