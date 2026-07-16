# Changed Files

## Production

- `tools/run_kraken_family_engine_aggregate_first_sweep.py` (`+61/-9`): added a loader-local per-file authority assertion, Kraken row filter, fixed 2023 lower boundary, exact-funding restriction, and pre-reader checks in the reproduced market/mark/funding paths.

## Tests

- `unit_tests/test_rankable_loader_boundary.py` (new, 244 lines): synthetic file metadata, in-memory rows, payload-reader spies, and downstream spies for all required cases.
- `unit_tests/test_kraken_family_engine_aggregate_first_sweep.py` (`+27/-5`): two existing synthetic direct-loader fixtures now declare explicit rankable file authority.

## Task archive

- `docs/agent/task_archive/20260716_donch_bt_loader_boundary_repair_20260716_v1/`: task specification, context, plan, defect evidence, command/validation/review records, readiness rerun, manifests, completion, and next action.

No strategy runner, hypothesis, cost, universe, lifecycle, capture, acquisition, governance, dependency, raw-data, result-root, or substantive registry file changed.
