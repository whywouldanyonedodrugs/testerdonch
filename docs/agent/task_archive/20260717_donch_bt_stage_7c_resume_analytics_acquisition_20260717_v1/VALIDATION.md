# Validation

- Compile: pass.
- Focused and applicable repository tests: 72 passed, 0 failed.
- Deterministic inventory: 460 identities, 16 shards, maximum 32 symbols per shard.
- Schedule: 1,836 shard-month-metric units; 60-second BTC/ETH first, then 300-second BTC/ETH, then remaining frozen shard order.
- Final data-object target: 3,672 raw bundles plus Parquet files, below 5,000.
- Prestart projected free: 41,754,189,491 bytes; 25% threshold: 40,242,561,024 bytes; pass.
- Telegram configuration: enabled without exposing credentials.
- Economic fields/outputs: zero.
- Protected payload test: fail-fast before value traversal passes.
