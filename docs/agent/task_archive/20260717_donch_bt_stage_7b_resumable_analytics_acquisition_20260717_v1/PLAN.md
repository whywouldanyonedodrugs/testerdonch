# Execution Plan

## Objective

Build and validate a resumable public Kraken Futures analytics downloader, freeze the C03-derived PF identity inventory, and complete only the bounded Phase A authority/pagination/units/storage audit. Phases B/C remain gated because the target filesystem starts below the mandatory 50 GiB free reserve.

## Authorities

- Starting commit: `7cd2444c92c5aecedeb1a897c919c25cbfd8749d`.
- Stage 7A hashes: retention `c125265f...`, request ledger `cc3a4faa...`, schema `e4a6ebd...` (full hashes validated in task evidence).
- C03 authoritative root: `results/rebaseline/phase_kraken_c03_pit_cohort_breadth_20260717_v1_20260717_173151`.
- Endpoint and type vocabulary: Stage 7A official Kraken schema snapshot.
- Data root: new local-only root under `/opt/parquet/kraken_derivatives/analytics/stage7b_v1`; review artifacts under a new ignored result root.

## Frozen Phase A

- Inventory: exact unique `PF_*` identities, no collision, with possible overlap before `2026-01-01`; exclusions retained with reasons.
- Audit symbols: PF_XBTUSD, PF_ETHUSD plus the first four canonical IDs with 1,095 trade and mark coverage days, excluding BTC/ETH, sorted by normalized canonical ID then PF symbol.
- Metrics: `open-interest`, `liquidation-volume`, `future-basis` only.
- Intervals: 60 and 300 seconds.
- Windows and bounds: the four exact seven-day windows in the task, always explicit and pre-2026.
- No price/return joins, signals, outcomes, ranks, or economic outputs.

## Milestones

1. Freeze/hash inventory and exclusions before Phase A value access.
2. Implement transactional SQLite jobs, bounded paging, immutable compressed raw evidence, deterministic Parquet parts, hash verification, stale recovery, safety guards, and synthetic tests.
3. Run Phase A only, record paging/bounds/schema/unit behavior and storage benchmark.
4. Independently review Phase A. Because initial free space is below 50 GiB, stop before Phases B/C regardless of projection.
5. Finalize local manifests, factual readiness records, compact package, reviewed commits, non-force push, and approved Drive handoff.

## Failure and rollback

Protected/mixed payloads, conflicting duplicates, schema drift, ambiguous units, repeated request failures, or ledger/hash inconsistency fail closed. Existing data and roots are never deleted or overwritten. Verified parts are immutable. Interrupted jobs remain resumable through the SQLite ledger.
