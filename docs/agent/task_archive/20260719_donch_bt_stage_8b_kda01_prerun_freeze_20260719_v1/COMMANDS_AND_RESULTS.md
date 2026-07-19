# Commands and Results

- Repository/main/origin verification at `41b64b52a9146669eb26dcf25a86523a35219b8d`: passed.
- Stage 8A summary, cache manifest, and KDA01 tape hash verification: passed.
- `/opt/testerdonch/.venv/bin/python -m py_compile tools/qlmg_kda01_v2.py tools/build_kda01_v2_prerun_freeze.py unit_tests/test_kda01_v2_prerun_freeze.py`: passed.
- Focused Stage 8B unit suite: `23 passed`.
- Relevant regression suite: `199 passed`.
- `/tmp/validate_stage8b.py`: passed all shard, row-reconciliation, identity, branch-direction, time-boundary, definition, attempt, outcome-column, and source-hash assertions.
- Final authoritative generator used `/usr/bin/time -v` and `--tg-auto-chat`: exit `0`, `24m18.11s`, `1,928,216 KiB` max RSS, no swap.
- Cache-only deterministic replay: parent/event Parquet hashes matched byte-for-byte.
- No economic runner, outcomes, PnL, funding outcomes, controls, protected payload, acquisition, Capital.com, capture, private endpoint, or order action was invoked.

Fail-closed attempts and their exact logs are preserved under `attempts/`; none supplied decision-bearing evidence.
- Active/task documentation-link check: passed across 35 files; archived received-source links were excluded and left unchanged.
- `git diff --check`: passed.
- Secret-pattern scan: passed.
