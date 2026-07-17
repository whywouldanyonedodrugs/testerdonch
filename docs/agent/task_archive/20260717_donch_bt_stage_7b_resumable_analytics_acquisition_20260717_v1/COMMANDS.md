# Commands

Key non-economic commands executed:

- Verified repository, Stage 7A authority hashes, C03 inventory authority, filesystem space/inodes, and official public analytics endpoint scope.
- `./.venv/bin/python tools/acquire_kraken_futures_analytics.py --phase-a --replay ...` against the corrected fresh run/data roots.
- `./.venv/bin/python tools/finalize_kraken_futures_analytics_phase_a.py --run-root ... --data-root ...`
- `./.venv/bin/python -m py_compile tools/acquire_kraken_futures_analytics.py tools/finalize_kraken_futures_analytics_phase_a.py unit_tests/test_acquire_kraken_futures_analytics.py unit_tests/test_finalize_kraken_futures_analytics_phase_a.py`
- `./.venv/bin/python -m unittest unit_tests.test_acquire_kraken_futures_analytics unit_tests.test_finalize_kraken_futures_analytics_phase_a unit_tests.test_kraken_futures_analytics_retention unit_tests.test_rankable_loader_boundary unit_tests.test_sealed_slice_guard unit_tests.test_c16_flow_authority_preflight unit_tests.test_kraken_first_wave_closure_review`

The first audit attempt was stopped and quarantined after a protected-boundary timestamp was detected before analytics values were traversed. It was not resumed. No Phase B/C or economic command ran.
