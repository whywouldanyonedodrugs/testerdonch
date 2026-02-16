# JT-005 Intrabar Replay Parity Harness

## Purpose

Validate deterministic first-touch behavior for SL/TP resolution on 1m data, including ambiguous bars where both levels are inside the same minute.

## Artifacts

- Harness script: `tools/intrabar_replay_parity_harness.py`
- Fixture pack: `unit_tests/fixtures/intrabar_replay_fixtures.json`

## Run

```bash
cd /opt/testerdonch
./.venv/bin/python tools/intrabar_replay_parity_harness.py
```

Optional:

```bash
./.venv/bin/python tools/intrabar_replay_parity_harness.py \
  --fixtures unit_tests/fixtures/intrabar_replay_fixtures.json \
  --outdir results/intrabar_replay_parity \
  --run-id manual_check
```

## Outputs

Per run:
- `results/intrabar_replay_parity/<run_id>/summary.json`
- `results/intrabar_replay_parity/<run_id>/results.csv`
- `results/intrabar_replay_parity/<run_id>/report.md`

## Acceptance

Pass when:
- `summary.status == "ok"`
- `checks_fail == 0`
- `determinism_failures == 0`

Notes:
- `resolver_vs_legacy_diff_checks` can be non-zero on ambiguous parent bars; this is diagnostic and expected.

