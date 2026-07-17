# Commands

Pre-execution compile and test command:

```bash
./.venv/bin/python -m py_compile tools/run_kraken_c01_level3_economic.py unit_tests/test_kraken_c01_level3_economic.py
./.venv/bin/python -m unittest -v unit_tests.test_kraken_c01_prerun_contract unit_tests.test_kraken_c01_level3_economic unit_tests.test_kraken_c01_event_contract unit_tests.test_kraken_c01_foundation unit_tests.test_rankable_loader_boundary unit_tests.test_sealed_slice_guard
```

Economic execution command will be recorded only after the implementation commit.

Authorized economic command executed once from runner commit `4571e097dcceddf228629c8e085e52dd1bfe47cb` with run root `results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227`. Exit 0; wall time `13:58.33`; maximum RSS `887132 KiB`.

## Handoff

- Checked the approved root with `rclone lsf` and confirmed the versioned child name was unused.
- Uploaded the five declared handoff files with immutable/no-overwrite behavior.
- Downloaded the remote folder to a separate temporary path and compared every filename, byte size, and SHA-256.
- Resolved task folder ID `1iP7O4sb91YoGomqQkH8pBiS56u8r0Eha`.
