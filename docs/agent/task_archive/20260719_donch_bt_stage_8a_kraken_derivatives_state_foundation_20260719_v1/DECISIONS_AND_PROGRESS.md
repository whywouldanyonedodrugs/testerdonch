# Decisions and Progress

## Frozen decisions

- Task: `donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1`.
- Work class: non-economic data semantics, causal feature foundation, and outcome-free event generation.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Semantic authority: transferred `inferred_authoritative_v1` decision, exact file SHA-256 verified before use.
- Stage 7C data authority: finalized manifest content hash `f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d` and all 1,836 final Parquet object hashes verified.
- No outcome reader, protected payload, capture source, Capital.com payload, or economic scoring is authorized.

## Implementation decisions

- Exact trade, mark, basis, OI, and liquidation rows are joined on completed five-minute timestamps.
- Raw decimal strings remain addressable; basis is converted to decimal/percent/bps, OI preserves exact OHLC tuple strings and base-unit values, and liquidation remains unsigned.
- Normalization uses distributions ending on the prior UTC day. Each intraday row is scored using its own current value; no same-day future aggregate is used.
- Lagged and rolling features fail closed unless the complete five-minute horizon is contiguous.
- Feature and event partitions are atomic, hash-manifested, and reusable only when contract and source hashes match.
- Event tapes are outcome-free. KDA03 remains feasibility-only.

## Progress and preserved attempts

1. Global DuckDB pivot attempt exceeded the bounded-memory objective; preserved locally and replaced by deterministic month shards.
2. A family-wide validity mask incorrectly coupled KDA01 to liquidation normalization; preserved and replaced by family-specific validity.
3. Non-positive OI caused non-finite ratios; repaired by explicit positive-denominator requirements and fail-closed non-finite handling.
4. Initial event reduction risked global Python accumulation; replaced by per-symbol event shards and a bounded DuckDB reducer.
5. Independent review found same-day normalization leakage and incomplete horizon-gap guards before publication. The generated tapes from that attempt are preserved under `attempts/pre_pit_repair_authoritative_attempt/` as non-authoritative evidence. The authoritative rebuild uses a fresh `stage8a_foundation_v1_exact` cache.
6. Identity review found that the robustness OI-vacuum attempt was mechanically identical to the primary OI-vacuum attempt and that event-shard reuse lacked a generator-contract hash. The duplicate attempt remains registered with a killed-branch reason; it emits no duplicate tape. Event identities and shard manifests now bind `generator_contract_hash`, and pre-repair event shards remain preserved as non-authoritative cache evidence.

No threshold was altered after counts, and no economic output was read or computed.
