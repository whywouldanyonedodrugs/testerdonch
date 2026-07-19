# Commands and results

- Verified Git root, worktrees, dirty state, exact base and sanitized remotes; created and hash-validated the external recovery bundle.
- Verified the approval attachment SHA-256 and Stage-14 packet, manifest, search, resource, and local-tape-manifest hashes.
- Ran `PYTHONPATH=. /opt/stage14-phase1-venv/bin/python tools/validate_stage14_closure.py`: 13 checks passed; 228 cells reconciled.
- Ran `PYTHONPATH=. /opt/stage14-phase1-venv/bin/python -m unittest unit_tests.test_qlmg_research_campaign -v`: 14 tests passed.
- Ran the combined campaign plus Stage-14 derivatives suite: 23 tests passed.
- Compiled the changed engine and test module with `py_compile`: passed.
- Ran `sha256sum -c /opt/testerdonch-stage15-dirty-recovery-20260720/RECOVERY_HASHES.sha256`: all entries passed.
- Ran `git diff --check`: passed.
- Independent first review: rejected three validator bypasses and two packet-semantic blockers.
- Independent re-review: validator findings closed and repair accepted; economic launch remained rejected on the two packet-semantic blockers.
- Economic run, funding extension, Telegram send, protected access, Capital.com access, and live action: not run.
