# Donch Dynamic Continuity Ledger

This folder carries compact dynamic project continuity. It is not a market-data store, backtest result store, or authorization mechanism.

Read `CURRENT_STATE_POINTER.json`, download the referenced snapshot, and verify the pointer's `snapshot_sha256` against the exact snapshot bytes. Validate the snapshot's embedded self-hash using the v1 canonical-null rule documented in the repository's `docs/agent/DYNAMIC_CONTINUITY.md`.

`events/` and `snapshots/` are immutable. Only `CURRENT_STATE_POINTER.json` is replaceable, through a verified temporary-object transaction. `daily/` contains optional non-authoritative digests. `SCHEMA.json` is the supplied v1 snapshot schema.

The ledger excludes credentials, private/account data, protected strategy outcomes, partial candidate rankings, and raw market payload values. It cannot authorize economic work, protected access, acquisition, orders, or live trading.
