# Validation

Status: **pass**

## Source And Coverage

- Exact authorized requests: 4/4 HTTP 200.
- Rows: 288 in each symbol/tick slice; 1,152 total source rows.
- First/last candle opens: `2025-12-31T00:00:00Z` / `2025-12-31T23:55:00Z`.
- Missing five-minute intervals: 0.
- Duplicate timestamps: 0.
- Out-of-order timestamps: 0.
- `more_candles=true`: 0.
- Returned rows at or after `2026-01-01T00:00:00Z`: 0.
- Known mixed 2025/2026 chunk opened: no.
- Safe old/new midnight boundary mismatches: 0/4.
- Same-directory and cross-output-directory deterministic builder replay mismatch: 0.

## Authority

- Prior Stage 2A authority reconciled for both symbols through `2025-12-31T00:00:00Z` exclusive.
- Official cumulative terminal-event rows parsed: 183.
- `PF_XBTUSD` and `PF_ETHUSD` entries in terminal table: 0.
- Terminal-table absence converted to uninterrupted/no-outage claim: no.
- Reference members: 2, factor/reference-only.
- Continuous-tradeability claim: no.
- Survivorship-free claim: no.

## Tests

```text
python -m py_compile tools/build_kraken_c01_reference_panel_authority.py unit_tests/test_kraken_c01_reference_panel_authority.py
result: pass

python -m unittest -v \
  unit_tests.test_kraken_c01_reference_panel_authority \
  unit_tests.test_kraken_u2_lifecycle_authority \
  unit_tests.test_sealed_slice_guard \
  unit_tests.test_project_deep_cleanup_20260624
result: 28 tests, 0 failures, 0 errors
```

The repository map's `pytest` pattern could not run because `pytest` is not installed in the task environment. The same relevant modules were run directly with `unittest`; no dependency or environment mutation was made.

## Prohibited Work

- Protected outcome payloads opened: 0.
- Candidate/return/funding readers called: 0.
- Economic outputs computed: 0.
- Capture accessed: no.
- Secret scan findings: 0; transient CDN cookie values are redacted in retained headers.
