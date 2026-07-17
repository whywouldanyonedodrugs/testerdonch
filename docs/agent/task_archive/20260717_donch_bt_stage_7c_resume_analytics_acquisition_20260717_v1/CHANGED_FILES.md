# Changed Files

- `tools/acquire_kraken_futures_analytics.py`: lossless semantic raw columns, schema/request provenance, and request-level progress callback.
- `tools/run_kraken_futures_analytics_stage7c.py`: deterministic shard-month acquisition, publication, bundling, compaction, resume, storage, heartbeat, and Telegram worker.
- `unit_tests/test_acquire_kraken_futures_analytics.py`: raw semantic and checkpoint-stop regressions.
- `unit_tests/test_kraken_futures_analytics_stage7c.py`: sharding, scheduling, protected bounds, bundle/compaction, cleanup, resume, and storage tests.
- Task archive: exact specification, plan, validation, review, and launch authority records.
