# Commands and Results

- Verified clean synchronized `main` at `9949b29ead0e6d6e17543ddd955bff0234805006`; created `feature/stage-2b-c01-foundation-20260717`.
- Compiled with `.venv/bin/python -m py_compile tools/build_kraken_c01_foundation.py unit_tests/test_kraken_c01_foundation.py`: pass.
- Ran the builder with the accepted Stage 2A/2A1 market manifest, current instrument snapshot, reference manifest/final-day files, and cached official terminal lifecycle source. Final exit status: 0; runtime `14:09.60`; peak RSS `4,946,856 KiB`.
- Ran independent full-tape validation (`VALIDATION_EVIDENCE.json`): all hard checks pass.
- Ran `.venv/bin/python -m unittest -v unit_tests.test_kraken_c01_foundation unit_tests.test_kraken_c01_reference_panel_authority unit_tests.test_kraken_u2_lifecycle_authority unit_tests.test_rankable_loader_boundary unit_tests.test_sealed_slice_guard unit_tests.test_project_deep_cleanup_20260624`: 58 passed, 0 failed, 0 errors.
- No economic, protected-outcome, capture, funding, or new-data-acquisition command was run.
