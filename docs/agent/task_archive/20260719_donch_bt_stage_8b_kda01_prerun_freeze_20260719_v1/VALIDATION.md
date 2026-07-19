# Validation

Status: pass; decision `ready_for_human_KDA01_Level3_run_approval`.

## Final mechanical evidence

- Stage 8A commit/hash authorities: verified.
- Stage 8A KDA01 tape SHA-256: `583c1f940f185cf01417a1f5ba6540c6a1b6545c0532851d1dabf200d8c874ce`.
- KDA01 v2 feature extension hash: `6934f48dceae6a12c92198a689d773206710b91b543e263330164b856364b157`.
- KDA01 v2 generator hash: `aa83c46a73068872986277741407d05ab0156a5660cb97bfd5d453401123d017`.
- Frozen Level-3 contract hash: `2eef5efb631e49014ea239eef5b90d4f2d5932fdcd33e97ec26067b5288ef938`.
- Complete feature/episode/event shards: `187/187`; hash verification passed.
- Stage 8A-to-OHLC row attrition mismatches: `0`.
- Parent episodes: `214,138`; candidate events: `102,136`; parent episodes with no candidate: `117,839`.
- Duplicate parent episodes/event IDs/economic addresses: `0/0/0`.
- Invalid onset classifications: `0`.
- Pre-2023/protected rows: `0/0`.
- Outcome columns/readers/economic outputs: `0/0/0`.
- Definitions frozen: `16`; controls frozen and unexecuted: `7`.
- Deterministic cache-only replay reproduced parent and event Parquet bytes exactly.
- Parent tape SHA-256: `ae470d06c7da049d145ac8e083d0c4711c9c2dd6e4ec32fedf79224e8885ddbd`.
- Event tape SHA-256: `7c9e682380681aa6fc83161ae29dafab9a047bdb19fe1fa8ade3ba5a4eb9c2e5`.

## Primary feasibility gates

All four primary branches passed total events, annual events, symbol count, maximum symbol share, duplicate identity, and protected-row gates. Exact counts are in `KDA01_V2_FEASIBILITY_GATES.csv` and `KDA01_V2_COUNT_MATRIX.csv`.

## Tests and resources

- Compilation: passed.
- Focused Stage 8B tests: `23 passed`.
- Final relevant Stage 8A/loader/sealed/lifecycle/C01/C02/analytics suite including Stage 8B: `199 passed`.
- Integrity validator: passed.
- Authoritative build: `24m18.11s` wall, `1,928,216 KiB` `/usr/bin/time` peak RSS, no swap.
- Generator-recorded peak RSS: `1.839 GiB`.
- Documentation links, `git diff --check`, artifact hashes, and secret scan are recorded in `COMMANDS_AND_RESULTS.md` after final closure.
