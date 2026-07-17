# Commands and Results

- Verified clean `main`/`origin/main` at `b4e6c24c7c1f6c54dea931d824383590d087b819`; created `feature/stage-3a-c02-spot-reference-20260717`.
- Retrieved official Kraken support and AssetPairs snapshots and official Drive archive metadata. Downloaded the complete-through-Q2, Q3, and Q4 archives; Q1 2026 was not downloaded/opened.
- ZIP integrity and SHA-256 checks: pass for all three acquired archives.
- Pilot mode: pass; elapsed `1:00.49`, peak RSS `225092 KiB`, 12/12 cells. Independent reconstruction: exact.
- Full mode: pass; elapsed `3:25.89`, peak RSS `460836 KiB`, 210/210 mapped pairs attempted, 204 output files.
- Gap-mask postprocess: initial invocation rejected a NaN path before mutation; corrected resume verified existing normalized hashes and finalized all 204 masks without overwriting normalized data.
- Whole-panel independent validation: pass for 204 normalized pairs and 15 source records; final evidence in `logs/full_panel_validation_final.log`.
- Compile: `./.venv/bin/python -m py_compile tools/run_kraken_c02_spot_reference_authority.py unit_tests/test_kraken_c02_spot_reference_authority.py`: pass.
- Focused tests: 12 passed. Boundary/loader/reference tests: 26 passed. C01/U2 authority regressions: 27 passed.
- No economic, return, score, ranking, protected-outcome, capture, or live command was run.
