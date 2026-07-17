# Pre-Execution Validation

Status: `pass`.

- Frozen authority verification: 10/10 file or semantic hashes pass.
- Compile: pass.
- Focused plus relevant guard suite: 64 tests passed, 0 failures/errors.
- CLI refuses execution without `--execute-economic-run` and refuses an existing run root.
- Synthetic mechanics cover all 16 definitions including zero-trade rows; smooth/jump entries; dominant-bar identity; confirmation boundary; mark-close/next-open stops; timeout; ambiguity; missing inputs; lifecycle; protected boundary; fixed-notional costs; signed funding; partitions; actual-exit non-overlap; concentration; 10,000-resample bootstrap; and unsafe input rejection.
- Real post-onset outcomes opened before implementation commit: 0.
