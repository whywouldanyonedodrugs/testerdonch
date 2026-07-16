# Next Task Recommendation

## Exactly One Task

`Stage_2A0_rankable_input_firewall_and_venue_partition`

This is narrower than `Stage_2A_U2_and_common_feature_foundation` and must precede it.

## Objective

Implement and test a strategy-agnostic pre-read input firewall that admits only Kraken files/row groups wholly inside `[2023-01-01, 2026-01-01)`, excludes capture roots, and partitions funding before any protected row enters rankable processing memory.

## Allowed Scope

- Rankable data-path discovery and loader utilities.
- Evidence-contract code for input-manifest validation.
- Non-economic synthetic fixtures and reader-spy tests.
- Metadata/manifests needed to classify file and row-group boundaries.
- Task archive and documentation for this prerequisite.

No strategy logic, signal generation, candidate returns, public acquisition, protected payload inspection, capture access, or substantive registry retuning.

## Deliverables

1. Versioned rankable-input manifest contract with venue, dataset role, minimum/maximum timestamp, row-group boundaries, and source hash.
2. Fail-closed loader preflight for pre-2023, 2026+, non-Kraken, mixed/unpartitionable, and capture inputs.
3. Funding pre-partition contract.
4. Synthetic reader-spy audit proving rejected inputs never reach `read_parquet`.
5. Integration tests for trade, mark, funding, and path discovery.
6. Explicit migration plan for existing mixed files without reading protected outcomes during verification.

## Stop Conditions

- Any file or row group cannot be classified without opening protected payload.
- Venue identity is ambiguous.
- Mixed files cannot be safely partitioned into an approved pre-2026 artifact.
- Any rejected input reaches the parquet reader in a spy test.
- Capture or protected strategy data would need to be inspected.

After this task passes, rerun `donch_bt_first_wave_readiness_20260716_v1`. Do not advance directly to an economic task.
