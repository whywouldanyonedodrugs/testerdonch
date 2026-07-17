# Commands

Key commands executed:

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/main
git switch -c task/stage-6a-c16-flow-authority-20260717
curl -I <official dated SEC URL>
curl --fail <three immutable dated SEC approval PDFs>
./.venv/bin/python -m py_compile tools/build_c16_flow_authority_preflight.py unit_tests/test_c16_flow_authority_preflight.py
./.venv/bin/python tools/build_c16_flow_authority_preflight.py --raw-root <local raw> --run-root <local run> --archive-root <task archive> --as-of-utc <frozen UTC>
./.venv/bin/python -m unittest unit_tests.test_c16_flow_authority_preflight unit_tests.test_rankable_loader_boundary unit_tests.test_sealed_slice_guard unit_tests.test_kraken_first_wave_closure_review unit_tests.test_kraken_c03_pit_context
```

No command opened Kraken prices, returns, signals, outcomes, protected observation rows, or capture data.
