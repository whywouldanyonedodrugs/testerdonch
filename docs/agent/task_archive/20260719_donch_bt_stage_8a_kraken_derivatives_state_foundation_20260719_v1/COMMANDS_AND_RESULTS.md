# Commands and Results

## Authority and repository preflight

- `git rev-parse main origin/main` and isolated-worktree checks: both started at `bf7a694c3d0764807cd12015046f633be54c53ab`; task branch `agent/stage8a-kraken-derivatives-state-20260719`.
- SHA-256 verification of transferred semantic JSON and research plan: passed (`c5ccd4f...` and `ed06fe...`).
- Stage 7C manifest and 1,836 final object hashes: verified by the authoritative runner before feature access.
- Default Drive root read/collision check through `qlmg_sweep_drive:`: passed; proposed `v01` task folder was absent.

## Implementation checks completed

- `/opt/testerdonch/.venv/bin/python -m py_compile tools/qlmg_kraken_derivatives_state.py tools/build_kraken_derivatives_state_foundation.py unit_tests/test_kraken_derivatives_state_foundation.py`: passed.
- `/opt/testerdonch/.venv/bin/python -m unittest unit_tests.test_kraken_derivatives_state_foundation`: 22 passed.
- Relevant loader, sealed-boundary, lifecycle, C01/C02, and analytics suite: 176 passed after the isolated worktree was supplied with symlinks to the existing ignored seal manifest and data-manifest-hash authority in the main checkout.
- Two earlier 176-test invocations produced four errors each solely because those ignored authority files were absent from the isolated worktree; their logs are preserved under `attempts/`, and no test assertion failed.
- `python -m ruff`: unavailable because Ruff is not installed in the repository environment; no installation was attempted.
- `git diff --check`: passed at each reviewed checkpoint.

## Authoritative generator

The final command is:

```bash
/usr/bin/time -v /opt/testerdonch/.venv/bin/python \
  tools/build_kraken_derivatives_state_foundation.py --tg-auto-chat
```

The generator is outcome-free and reads only authorized 2023–2025 trade, mark, lifecycle/liquidity authority, and Stage 7C analytics. Final runtime, RSS, validation, deterministic replay, commit, publication, and Drive results are recorded in `VALIDATION.md` and `COMPLETION.md`.

## Final measured results

- Final authoritative generator: exit `0`; `23m32.87s` wall; `2,796,504 KiB` max RSS; no swap.
- `/opt/testerdonch/.venv/bin/python /tmp/validate_stage8a.py`: passed all feature hashes, row reconciliation, duplicate, boundary, attempt, and one-minute gates.
- Cache-only deterministic replay: KDA01 and KDA02 Parquet SHA-256 values matched the authoritative tapes byte-for-byte.
- Final relevant suite after all code changes: `176 passed`.
- No economic runner, outcome reader, protected payload, capture source, new acquisition, Capital.com payload, private endpoint, or order path was invoked.
