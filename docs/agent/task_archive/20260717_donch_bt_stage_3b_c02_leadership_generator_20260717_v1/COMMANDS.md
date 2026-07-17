# Commands

- `git status --short && git branch --show-current && git rev-parse HEAD` — clean starting state apart from this new task archive; branch based on `2a83432a5ecd94b284b3a9c8f6366e4e0ae8df1f`.
- `./.venv/bin/python -m py_compile tools/build_kraken_c02_leadership_generator.py tools/watch_kraken_c02_leadership_generator.py unit_tests/test_kraken_c02_leadership_generator.py` — pass.
- `./.venv/bin/python -m unittest unit_tests.test_kraken_c02_leadership_generator unit_tests.test_kraken_c02_spot_reference_authority unit_tests.test_kraken_c01_foundation unit_tests.test_rankable_loader_boundary unit_tests.test_kraken_u2_lifecycle_authority` — 65 passed.
- `./.venv/bin/python tools/build_kraken_c02_leadership_generator.py` under detached tmux — final pass completed 204/204; 450.59 seconds; 983,384 KiB peak RSS; exit 0.
- Independent hash/identity/schema audit — event-ID mismatches 0; episode-ID mismatches 0; prohibited outcome fields 0; protected rows 0.

Earlier diagnostic executions were stopped before finalization after exposing quadratic reporting paths or incomplete reporting fields. Their logs are retained under `logs/`; no diagnostic output is authoritative.
