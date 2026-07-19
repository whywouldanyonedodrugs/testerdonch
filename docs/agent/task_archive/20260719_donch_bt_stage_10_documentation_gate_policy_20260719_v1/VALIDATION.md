# Validation

Status: pass for documentation, registry, authority, protected-boundary, and package integrity. This is not economic or statistical validation.

## Authority and immutable decisions

- Starting authority: `3ea0d320d71716a5c0890f4c924ed924224beda2`.
- KDA01 remains exactly `KDA01_level3_repaired_no_primary_pass_stop`.
- KDA02A remains exactly `KDA02_level3_no_primary_pass_stop`.
- Stage 8C1 finalized manifest SHA-256: `6c5bbaafb53cb4bf3dd8b10d5e1ae2b6964b61fdc5c82fc648d6b87fc596f382`.
- Stage 9 finalized manifest SHA-256: `2f19fc2e6d87d78706215d425ece94cd3629c05559e06a88d94da6e8d0b289f2`.
- Stage 9 compact ZIP SHA-256: `195ab09fbcd5f5cb25ca4889be95c183f9e76ee476aaa7d8bef1045e927507e0`.
- Stage 7C analytics manifest content hash: `f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d`.
- `git diff` reports no change under the finalized Stage 8C1 or Stage 9 roots.

## Source package

- Received ZIP SHA-256: `8096498ef634dbea0520cce1dbfb0c341cb8ccdfe67b8b84ece7c67468aef241`.
- Received package-manifest SHA-256: `bb72cac270224bb057d9aaf6c3b76be015d9e1695ac92d07a34412868c5545f5`.
- 13/13 declared files matched exact bytes and SHA-256; 14 total received files including the package manifest are retained.
- Active source-map paths: 13/13 required policy/continuity/registry paths resolved.

## Policy invariants

- Policy JSON version: `1.0`; SHA-256 `c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa`.
- Payoff archetypes: 6; routing statuses: 9; hard integrity gates: 8; routing diagnostics: 8.
- All six exact contract fields appear in the active evidence manual.
- The KDA02 negative six-hour row carries exact `post_hoc_context_hypothesis` and separate `conditional_context_candidate_unvalidated` tokens.
- Every candidate-route token in the gate matrix is an approved JSON routing status; year concentration uses the exact `_unvalidated` status.
- Controls require separate task authorization and cannot validate a candidate by themselves.
- Same-sample rescue, calendar-year context, retroactive promotion, and protected selection remain prohibited.

## Mechanical checks

- Active JSON parse: pass.
- Active CSV parse and uniform width: 7/7 pass.
- Changed-document local links: pass, zero missing.
- Source-map required paths: 13/13 present.
- Secret scan: zero findings.
- `git -c core.whitespace=cr-at-eol diff --cached --check`: pass; the exact received CRLF source bytes remain fully diffable.
- Focused dependency-free unit test: 5/5 pass.
- Full repository-supported pytest command: unavailable because `pytest` is not installed; sealed-slice unittest import also lacks `pandas`. No dependency installation was attempted for this documentation-only task.

## Boundaries

```text
strategy_code_changed: no
loaders_or_simulators_changed: no
data_or_run_roots_changed: no
economic_outputs_computed: no
candidate_returns_opened: no
protected_rows_opened: no
Capitalcom_payload_opened: no
controls_executed: no
```

## Publication and handoff

- Application commit: `91ba2ab07630556984bcdf6b2c650fa6b84fcf7f`; pushed non-force and verified at `origin/main`.
- Closed ZIP: `/opt/qlmg-stage10-policy-handoff-20260719/qlmg_stage10-conditional-alpha-policy_20260719_v01.zip`.
- ZIP bytes/SHA-256: `116191` / `bd746471bc5a008429ef7301dbd7b8727b4fe7706295435fef1dd0f59e98718c`.
- Drive path: `qlmg_sweep_drive:20260719_donch_bt_stage_10_documentation_gate_policy_20260719_v1_v01/`.
- Drive folder: `https://drive.google.com/drive/folders/1U_KaAb4OOvP0rpdlTxJXAUc65r9yytVk`.
- Exactly five direct files were uploaded without overwrite and round-trip downloaded; all sizes and SHA-256 values matched.
