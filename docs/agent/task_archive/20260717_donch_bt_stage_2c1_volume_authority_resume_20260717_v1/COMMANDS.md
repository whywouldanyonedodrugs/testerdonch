# Commands and Results

- Compile: `./.venv/bin/python -m py_compile tools/kraken_candle_volume_authority.py tools/acquire_kraken_candle_volume_authority.py tools/build_kraken_c01_event_contract.py unit_tests/test_kraken_candle_volume_authority.py unit_tests/test_kraken_c01_event_contract.py`: pass.
- Focused tests: `./.venv/bin/python -m unittest unit_tests.test_kraken_candle_volume_authority unit_tests.test_kraken_c01_event_contract`: 9 passed.
- Bounded acquisition: `./.venv/bin/python tools/acquire_kraken_candle_volume_authority.py --output-dir docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1`: pass; 12/12 intervals exact.
- Stage 2C builder: `./.venv/bin/python tools/build_kraken_c01_event_contract.py ...`: pass; 2:46.58 final runtime; 4,792,152 KiB peak RSS.
- Broad regression/guard suite: 58 passed, 0 failed, 0 errors; it includes the 9 focused tests. Exact output is in `logs/unit_tests.log`.
- Full-tape non-economic reconciliation: pass; exact evidence in `VALIDATION_EVIDENCE.json`.
- Cost authority: `results/rebaseline/phase_kraken_c2_shock_episode_budget_repair_20260713_v1/contract/economic_cost_and_funding_policy_v2.md`, SHA-256 `09054ab7ff7794af3a3c58ecff986d9ce8d4af646319ac08146532d00ae98176`.
- No economic, candidate-return, control-outcome, protected-outcome, capture, or live command was run.
