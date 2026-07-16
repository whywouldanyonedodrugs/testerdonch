# Commands and Results

| UTC | Command | Exit | Result |
|---|---|---:|---|
| 2026-07-16 | preflight Git identity/status/ancestor checks | 0 | required ancestor and clean branch verified |
| 2026-07-16 | `python -m unittest -v unit_tests.test_rankable_loader_boundary` before patch | 1 | 11 tests; 8 failures and 5 errors reproduced missing guards/filter/binding |
| 2026-07-16 | same focused command after patch | 0 | 11/11 pass |
| 2026-07-16 | `python -m unittest unit_tests.test_kraken_family_engine_aggregate_first_sweep` first run | 1 | 283 pass; 3 synthetic fixtures lacked explicit authority |
| 2026-07-16 | same owning-module command after fixture updates | 0 | 286/286 pass |
| 2026-07-16 | `python -m unittest -v unit_tests.test_project_deep_cleanup_20260624 unit_tests.test_sealed_slice_guard` | 0 | 9/9 pass |
| 2026-07-16 | `python -m py_compile` for production and changed tests | 0 | pass |
| 2026-07-16 | metadata-only real `data_paths()` authority binding | 0 | 166,408 existing trade/mark paths; no Parquet read |
| 2026-07-16 | AST reader call-order scan | 0 | 5/5 known readers guard before read and filter rows |
| 2026-07-16 | `git diff --check` | 0 | pass |
