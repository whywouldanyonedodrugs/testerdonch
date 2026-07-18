# Commands

Python executable: `/opt/testerdonch/.venv/bin/python`

## Intake

```bash
git rev-parse HEAD
git rev-parse origin/main
git status --short
git remote get-url origin
git submodule status --recursive
sha256sum research_inputs/donch_to_backtesting_multiplatform_authority_v1_20260718.zip
python - <<'PY'
# Open ZIP, resolve INPUT_MANIFEST.json records under the ZIP top-level prefix,
# and compare each declared byte_size and sha256.
PY
```

Result: package SHA-256 `58991dad07d1e3563f101ef784a9ac5b50fb2ce95fc52126584d29787b34d89b`; manifest SHA-256 `b3e0a2d9d9bbc7156e39d85bd5d9163122c2f259098729d1c9ac35504ffb563f`; 17/17 records pass.

## Compile And Focused Tests

```bash
/opt/testerdonch/.venv/bin/python -m py_compile \
  tools/qlmg_rankable_source_contract.py \
  tools/capitalcom_data_adapter.py \
  tools/run_kraken_family_engine_aggregate_first_sweep.py \
  unit_tests/test_rankable_source_contract.py \
  unit_tests/test_capitalcom_data_adapter.py

/opt/testerdonch/.venv/bin/python -m unittest -v \
  unit_tests.test_rankable_source_contract \
  unit_tests.test_capitalcom_data_adapter \
  unit_tests.test_rankable_loader_boundary
```

Result: compile pass; 27/27 focused tests pass.

## Required Baseline Plus New Tests

The isolated worktree temporarily linked its ignored `results/rebaseline` path read-only to `/opt/testerdonch/results/rebaseline` for the one existing campaign fixture, then removed the link.

```bash
/opt/testerdonch/.venv/bin/python -m unittest -v \
  unit_tests.test_rankable_loader_boundary \
  unit_tests.test_kraken_readiness_repair \
  unit_tests.test_qlmg_signal_state_contract \
  unit_tests.test_qlmg_mechanical_qa_evidence_contract \
  unit_tests.test_rankable_source_contract \
  unit_tests.test_capitalcom_data_adapter
```

Result: 65/65 pass.

## Complete Unit-Test Discovery

```bash
/usr/bin/time -f 'elapsed_seconds=%e max_rss_kb=%M exit=%x' \
  /opt/testerdonch/.venv/bin/python -m unittest discover -s unit_tests -v
```

Result: 1,165 executed; 1,162 pass; 1 failure and 2 errors. The exact three failures reproduce on untouched starting `main`:

- `test_kraken_session_open_range_resolution...test_opening_range_is_completed_and_break_is_close_confirmed`: existing final-2025 chunk authority status fixture.
- `test_qlmg_corrected_event_level_development_sweep...test_stage_nulls_ingests_real_controls_and_relabels`: existing stale control fixture schema.
- `test_jt008_leakage_guards...test_repo_guards_pass`: existing repository-state guard result.

No task file was changed to mask these unrelated failures.

## Review Checks

```bash
git diff --cached --check
git diff --cached --name-only
python - <<'PY'
# AST import audit for the two new operational modules.
PY
rg -n '<secret-signature-patterns>' <staged operational diff>
python - <<'PY'
# Relative Markdown link existence check for six changed active documents.
PY
```

Result: authorized paths only; whitespace pass; no API/account/order client dependency; no secret-like signature; 0 missing documentation links.
