# Commands

Pre-execution compile and test command:

```bash
./.venv/bin/python -m py_compile tools/run_kraken_c01_level3_economic.py unit_tests/test_kraken_c01_level3_economic.py
./.venv/bin/python -m unittest -v unit_tests.test_kraken_c01_prerun_contract unit_tests.test_kraken_c01_level3_economic unit_tests.test_kraken_c01_event_contract unit_tests.test_kraken_c01_foundation unit_tests.test_rankable_loader_boundary unit_tests.test_sealed_slice_guard
```

Economic execution command will be recorded only after the implementation commit.
