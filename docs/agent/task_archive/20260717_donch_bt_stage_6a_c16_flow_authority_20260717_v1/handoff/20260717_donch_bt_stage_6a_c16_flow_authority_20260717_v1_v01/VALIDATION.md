# Validation

Status: pass for the non-economic authority-unavailable conclusion.

## Tests

```bash
./.venv/bin/python -m unittest \
  unit_tests.test_c16_flow_authority_preflight \
  unit_tests.test_rankable_loader_boundary \
  unit_tests.test_sealed_slice_guard \
  unit_tests.test_kraken_first_wave_closure_review \
  unit_tests.test_kraken_c03_pit_context
```

- Total: 48 passed, 0 failed.
- C16 focused tests: 14 passed.
- Independent evidence checks: 20 passed, 0 failed.
- Deterministic full-build replay: 17 files compared, 0 mismatches.
- Artifact-manifest duplicate paths: 0.

The focused fixtures cover same-day and next-day publication, revision selection, current/mixed payload rejection before reader invocation, protected-boundary rejection, AUM/flow semantic separation, derived-share arithmetic, product identity, lifecycle/calendar coverage, and deterministic hashes.

## Boundary result

- Protected observation rows parsed/opened: 0.
- Kraken rows or outcomes opened: 0.
- Economic outputs computed: 0.
- One malformed historical-date issuer probe returned an HTML body instead of a dated CSV. It was detected before observation parsing, quarantined with mode `000`, and excluded from all panels, manifests, Git, and Drive.
