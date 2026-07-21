# Commands and results

All commands used the repository-root module path through the task-local
`.venv`; no `PYTHONPATH` override was used.

## Focused implementation tests

```text
.venv/bin/python -m unittest unit_tests.test_core_liquid_campaign -v
Ran 18 tests in 22.213s
OK
```

Coverage includes all 271 typed axis levels, active/inactive branches, identity
and invalid combinations, raw family dispatches, A2 exact parents, KDA raw
predicates, all accounting exits, favourable/adverse funding, PIT/platform
firewalls, all 20 control classes, explicit empty folds, plateau/refinement,
selection, cache/authority bindings, PID churn, resource excursion, worker
retry, bound stop and idempotent recovery.

## Deterministic compiler replay

```text
compile_deterministic(...)
validate_compiled(...)
independent_replay(...)
```

Result: PASS; 9,088 registered rows, 9,083 unique execution rows, 800 unique
controls, 2,270 A2 counterpart rows, zero marginal/pairwise failures, zero
unrepresented valid regions, and 37 replayed files with zero byte mismatch.

## Repository-wide baseline

```text
.venv/bin/python -m unittest discover -s unit_tests -v
Ran 1245 tests in 43.735s
FAILED (failures=2, errors=33)
```

The Stage-22 module's 18 tests all passed within that run. The remaining
repository-wide failures are pre-existing/out-of-scope environment and fixture
issues: optional packages absent from `requirements.txt` (`requests`,
`tabulate`, `yaml` in some loaders), result roots intentionally absent from the
isolated worktree, current governance expectations differing from old tests,
and unrelated legacy fixtures. No reported failure referenced a changed
Stage-22 file or its generated candidate.

## Static checks

```text
python -m compileall tools/core_liquid_campaign unit_tests/test_core_liquid_campaign.py
git diff --check
changed-file credential/private-key pattern scan
```

Result: PASS; no syntax failure, whitespace error or credential-like match.

## Candidate-build serialization regression

The first `v02` build stopped before review because strict JSON correctly
rejected an internal negative-infinity empty-fold sentinel. After adding an
explicit JSON-safe unavailable representation, the focused suite was repeated:

```text
.venv/bin/python -m unittest unit_tests.test_core_liquid_campaign -v
Ran 18 tests in 22.140s
OK
```

The failed `v02` root was preserved and is not a review candidate. The repaired
candidate is built in a new versioned root.

## V03 review and resumed verification

```text
V03 review target SHA-256:
d573d4c6982eb4d6f5d434345c03714bc70fd524ae6c3c464563902ba908b779

V03 independent review SHA-256:
2af69d48e55d31dbe9b29bf7cf6c8459141eb2c0304368ee2d447773e94b80a0

V03 verdict: BLOCK (12 consolidated findings)
```

After the second bounded repair and the resumed-session threshold/funding
audit:

```text
.venv/bin/python -m unittest -v unit_tests.test_core_liquid_campaign
Ran 25 tests in 38.275s
OK

.venv/bin/python -m unittest -v \
  unit_tests.test_core_liquid_campaign.EngineAccountingControlTests \
  unit_tests.test_core_liquid_campaign.RuntimeAuthorityAndReviewTests.test_cache_authority_binds_physical_artifact_and_frame_content
Ran 10 tests in 13.069s
OK
```

No economic outcome reader, protected row, Capital.com payload, order,
deployment or live-trading action was invoked.
