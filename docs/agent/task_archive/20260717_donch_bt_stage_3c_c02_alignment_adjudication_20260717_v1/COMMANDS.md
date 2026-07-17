# Commands

- `git merge-base --is-ancestor 64c65e2... 408b9d9...` - pass; handoff discrepancy resolved as implementation ancestor versus final archive commit.
- `./.venv/bin/python -m unittest ...` - focused pre-run suite passed.
- `/usr/bin/time -v ./.venv/bin/python tools/adjudicate_kraken_c02_alignment.py` - pass; 32,686 frozen events reconciled.
- Full replay to `/tmp/donch_stage3c_c02_deterministic_replay_20260717` - pass; all generated hashes byte-identical.
- Final compile and 75-test relevant suite - pass.
- Artifact hash, schema, protected-boundary, ZIP integrity, secret, and Drive round-trip checks are recorded in the manifest and handoff record.
