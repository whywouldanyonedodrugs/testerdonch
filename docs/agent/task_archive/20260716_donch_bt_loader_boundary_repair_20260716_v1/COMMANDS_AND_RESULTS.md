# Commands and Results

Working directory for all commands: `/opt/testerdonch`.

| UTC | Command | Purpose | Exit | Result |
|---|---|---|---:|---|
| 2026-07-16 19:31 | `./.venv/bin/python -m pytest unit_tests/test_rankable_loader_boundary.py -q` | discover documented test command | 1 | pytest unavailable in venv; no dependency installed |
| 2026-07-16 19:31 | `./.venv/bin/python -m unittest -v unit_tests.test_rankable_loader_boundary` | pre-patch reproduction | 1 | 6 tests; 14 assertion/subtest failures for the demonstrated boundary defects |
| 2026-07-16 | same focused unittest command | post-patch focused verification | 0 | 8/8 passed |
| 2026-07-16 | `./.venv/bin/python -m unittest unit_tests.test_kraken_family_engine_aggregate_first_sweep` | complete owning-module regression | 0 | 286/286 passed |
| 2026-07-16 | `./.venv/bin/python -m unittest -v unit_tests.test_project_deep_cleanup_20260624 unit_tests.test_sealed_slice_guard` | repository-supported non-economic guards | 0 | 9/9 passed |
| 2026-07-16 | `./.venv/bin/python -m py_compile tools/run_kraken_family_engine_aggregate_first_sweep.py unit_tests/test_rankable_loader_boundary.py` | syntax | 0 | passed |
| 2026-07-16 | original-readiness archive hash script | authority/archive integrity | 0 | 22/22 entries matched |
| 2026-07-16 | AST raw-reader scan | readiness rerun | 0 | two repaired paths plus three active unguarded siblings identified |

No command opened a real market, funding, capture, or protected payload. No command calculated candidate returns or launched an economic process.
