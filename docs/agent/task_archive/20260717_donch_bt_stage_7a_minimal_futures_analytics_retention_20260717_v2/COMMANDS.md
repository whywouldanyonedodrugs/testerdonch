# Commands

- `python -m py_compile tools/probe_kraken_futures_analytics_retention.py unit_tests/test_kraken_futures_analytics_retention.py`
- `python -m unittest unit_tests.test_kraken_futures_analytics_retention`
- `python -m unittest unit_tests.test_kraken_futures_analytics_retention unit_tests.test_rankable_loader_boundary unit_tests.test_sealed_slice_guard unit_tests.test_c16_flow_authority_preflight unit_tests.test_kraken_first_wave_closure_review`
- `python tools/probe_kraken_futures_analytics_retention.py --output results/rebaseline/phase_kraken_futures_analytics_retention_probe_20260717_v2`
- Independent read-only CSV/raw-hash reconciliation (recorded in `INDEPENDENT_REVIEW_EVIDENCE.json`).

The probe command was executed once. It issued exactly 48 data requests: the frozen 24-cell matrix and one exact replay. No third matrix or network retries were run.
