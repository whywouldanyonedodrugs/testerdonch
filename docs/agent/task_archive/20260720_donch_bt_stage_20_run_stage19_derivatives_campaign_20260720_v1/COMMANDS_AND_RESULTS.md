# Commands and results

## Authority and repository preflight

- ZIP structural test: pass.
- ZIP and all three member SHA-256 checks: pass; see `PACKAGE_VERIFICATION.json`.
- `git fetch origin`: pass.
- canonical `main`, `origin/main`, and isolated-worktree base: `245b375b00167f1b4a81f6a4449e7de1d1db83a2`.
- canonical checkout cleanliness: pass.
- continuity pointer sequence 1 and referenced snapshot size/SHA-256 round trip: pass.
- Stage 19 Drive archive size/SHA-256 and all 18 bound dependency hashes: pass.
- free-resource gate: 4 CPUs, more than 5 GiB available RAM, more than 5 GiB free disk: pass.

## Validation commands

```text
/opt/testerdonch/.venv/bin/python tools/run_stage20_campaign.py preflight \
  --approval <received exact approval JSON> \
  --run-root /opt/testerdonch/results/rebaseline/phase_kraken_derivatives_campaign_stage20_20260720_v01
```

Result: pass; 187 funding symbols; Stage 19 launch validator and synthetic funding canary pass.

```text
/opt/testerdonch/.venv/bin/python -m unittest \
  unit_tests.test_stage20_campaign \
  unit_tests.test_qlmg_stage19_funding \
  unit_tests.test_validate_stage19_campaign_packet -v
```

Result: 15 focused tests passed before the final supervisor-order guard; the Stage 20-only suite subsequently passed 9/9 after that guard was added.

The four sealed-slice tests cannot find their ignored seal fixture inside an isolated worktree. The same exact tests were rerun in `/opt/testerdonch`, where the fixture exists, and passed 4/4. No protected payload was opened by those tests.

```text
/opt/testerdonch/.venv/bin/python tools/stage20_phase2_5_canary.py \
  --output <run-root>/preflight/SYNTHETIC_PHASE2_5_CANARY.json
```

Result: pass; synthetic only; zero real economic outcomes; deterministic maximum-five beam; outer-result isolation; family stop; zero protected or Capital.com access.

## Pre-outcome event construction

Primary and frozen-threshold replay builds use `tools/run_stage20_campaign.py build-events`. They read only causal feature/event identity inputs and keep the economic outcome reader closed. Their terminal manifests and hash-only replay comparison are retained under the local run root.

## Economic and closure commands

The remediation suite passed 21/21 focused tests. The reviewed source manifest
canonical hash was `a7d7dc862f72148d11bd5c2acd00cab8d9eb12c47c9d0036ceaa079660b463e7`.
Bound replay (`836d133d...`), synthetic supervisor canary (`cd4bc932...`),
repeated independent review (`eb506bfa...`), secure Telegram preflight, and
final launch authority (`30a8420d...`) all passed.

The final atomic launch audit reverified the approval, 18 dependency hashes,
funding package and 187-symbol runtime, 187 event partitions, source/gate
hashes, repository authority, and resource limits. The campaign launched and
then reached a genuine global stop at `2026-07-20T15:10:26Z`:

```text
ProcessLookupError: [Errno 3] No such process
```

The Telegram global-stop notification succeeded. Metadata-only closure found
6,459 valid completed markers, 3,193 fully hash-reconciled artifacts, 126
registered cell identities represented, no remaining workers or temporary
files, zero protected rows, and no Capital.com access. The campaign is not
complete and no result or route is claimed.
