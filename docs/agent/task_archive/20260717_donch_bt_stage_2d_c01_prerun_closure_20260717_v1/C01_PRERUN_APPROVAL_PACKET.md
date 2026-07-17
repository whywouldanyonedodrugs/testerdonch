# C01 Level-3 Pre-Run Approval Packet

Final contract SHA-256: `c655e94c35412354356bb7f89c07ca17b71c2ae6537a2a1c42aa3dce928ba77d`.

Input hashes: generator `3464e79a79956c881c7418840068a61e3f3a47776a5a4d3a669e98df124fd970`; draft `f1c8c612ea9f7ffcc2abad3f2efde36b5dfb68fde20d2769fdc5ce40ab306c13`; feature `c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb`; cohort `768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15`; reference panel `2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763`.

Expected fresh run root: `results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_<UTC_SUFFIX>`.

Frozen later command interface (do not execute without a separate economic-run authorization):

```bash
./.venv/bin/python tools/run_kraken_c01_level3_economic.py --contract docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md --definition-register docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/C01_LEVEL3_DEFINITION_REGISTER.csv --event-tape docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1/C01_ONSET_EVENT_TAPE.parquet --run-root results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_<UTC_SUFFIX> --execute-economic-run
```

The later authorized execution task must first implement or verify the named runner at its approved commit, substitute an actual UTC suffix, confirm all input hashes, and refuse an existing root. This task does not create or execute the economic runner.

Rollback: stop before outcome access, preserve any incomplete fresh root as failed provenance, and do not mutate Stage 2B, Stage 2C1, the external package, or this contract root.

Prohibited: protected or mixed 2025/2026 reads; threshold or definition changes; event caps/sampling; artificial boundary closes; pooled funding rescue; nominal-hold preblocking; controls before a primary Level-3 pass; validation/promotion/live claims; overwriting prior roots.
