# Changed Files

- `tools/run_kraken_c02_spot_reference_authority.py`: official-source identity, bounded archive reader, deterministic sparse five-minute aggregation, gap masks, authority tables, and manifest writer.
- `unit_tests/test_kraken_c02_spot_reference_authority.py`: 12 synthetic identity, boundary, aggregation, gap, duplicate, streaming, and determinism tests.
- This task archive: source snapshots, pilot freeze/review, pair/source/coverage authorities, contracts, capability and continuity records, local-only manifest, validation, review, completion, and handoff records.

Large raw ZIPs, normalized Parquet bars, pilot bars, and gap-mask Parquets are local-only under `/opt/parquet/kraken_spot_reference/` and are not tracked or uploaded.
