# Commands and Results

- Verified clean isolated worktree at approved base `e14bbd0d26c14e48a347481f170fcfe8851df625`; preserved the dirty original checkout in `/opt/testerdonch-stage14-dirty-recovery-20260719` with verified hashes.
- Ran the Stage-14 builder over all 187 hash-verified Stage-8 feature partitions. It read 39,279,314 contemporaneous rows and wrote local outcome-free tapes under `/opt/parquet/kraken_derivatives/analytics/stage14_phase1_v1`.
- Ran `tools/finalize_derivatives_phase1_closure.py` after adversarial repairs. Final single-worker outcome-free replay: see `RESOURCE_PROJECTION.json`; no forward outcome, protected row, or Capital.com payload was opened.
- Ran `tools/build_stage14_campaign_packet.py` twice and compared all packet hashes; replay was byte-identical.
- Ran `tools/validate_stage14_closure.py`: pass.
- Ran `python -m unittest unit_tests.test_derivatives_phase1 unit_tests.test_qlmg_research_campaign -v`: 19 tests passed.
- Ran `python -m py_compile` on all five Stage-14 tools: pass.
- Ran `git diff --check`: pass.
- Verified every file, size, and SHA-256 in `LOCAL_STATE_TAPE_MANIFEST.json`: 563 files passed. Final campaign/packet deterministic replay passed against implementation commit `b4785ed2a06fbed50d20b7dcdf0bc27e93cd7bea`.
