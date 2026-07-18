# Multi-Platform Rankable Source Authority Contract and Synthetic Adapter Boundary v1

task_id: donch_multiplatform_source_authority_synthetic_adapter_20260718_r1
date_utc: 2026-07-18
target_environment: backtesting Codex
mode: direct_apply in an isolated worktree/branch; local reviewed commit authorized; no push or merge
drive_handoff: approved_default

## Objective

Implement one minimal non-economic platform boundary: a source-neutral rankable manifest guard, unchanged Kraken compatibility, and a synthetic-fixture Capital.com bid/ask adapter contract.

## Governing approval

The human approved exactly on 2026-07-18:

1. Donch research scope expands from Kraken-only research to two research platforms: Kraken derivatives and Capital.com instruments present in a verified acquisition manifest for the configured account and legal environment.
2. The common rankable interval remains [2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z).
3. Data from 2026-01-01 onward remain protected on both platforms. July 2026 Kraken capture remains execution_calibration_only. Capital.com data from 2026 onward default to data_engineering_only until another purpose is explicitly approved. Pre-2023 data may support context and mechanism priors but may not enter rankable results.
4. Paid historical vendor data remain prohibited.
5. No live trading, demo or real orders, private-account actions, risk changes, or production deployment are authorized on either platform.
6. Kraken and Capital.com mechanics remain platform-specific. Funding or financing, bid/ask prices, spreads, calendars, expiries, corporate actions, lifecycle, margin, currency conversion, trading rules, and fills must not be collapsed into one generic execution model.
7. Cross-platform research requires an explicit directed source-to-target contract. Capital.com -> Kraken and Kraken -> Capital.com are separate hypotheses. Every economic run requires separate approval of its frozen contract.
8. Existing Kraken identities, hashes, family decisions, negative findings, run roots, and evidence lineage must remain unchanged. New platform-aware identities may wrap existing Kraken identities but may not rehash or rewrite them.
9. “All Capital.com instruments” means acquisition and inventory scope. It does not authorize an immediate all-instrument economic search. Every research universe must be declared, point-in-time valid, and appropriate to the hypothesis.
10. The bounded non-economic task “Multi-Platform Rankable Source Authority Contract and Synthetic Adapter Boundary v1” is authorized, including an isolated branch/worktree, exact authority-document updates, source-neutral pre-open manifest guard, Kraken compatibility call, synthetic Capital.com bid/ask adapter tests, task archive, Drive handoff, and a local reviewed commit. Real Capital.com import, economics, protected outcomes, Kraken identity changes, push, merge, deployment, order endpoints, and account actions remain prohibited.
11. Preparation of revised Donch source files and a source-replacement manifest is authorized. Uploading, replacing, or deleting live Donch project sources is not authorized by this approval.


## State to reverify

The prior preflight verified `/opt/testerdonch`, clean `main`, commit `9df9c4f319cceffce84fc4567ddcaec9333c2bb7`, synchronized with `origin/main`, root `AGENTS.md`, and 49/49 corrected baseline tests. Treat these as prior evidence, not current state. Recheck root, instruction chain, branch, commit, remotes, worktree, commands and Drive target before change.

## Authority order

1. Current machine-enforced contracts and finalized manifests.
2. Current repository roots, hashes, ledgers and tests.
3. The exact human approval above and attached multi-platform contract.
4. Current repository manuals and the attached preflight design.

If a higher machine contract makes the approved patch unsafe or contradictory, stop and report the exact conflict. Do not run economics to resolve it.

## Allowed files

Inspect repository files as needed. Change only:

- `AGENTS.md`
- `docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md`
- `docs/agent/REPOSITORY_MAP.md`
- `docs/QLMG_PERP_PROJECT_STATE.md`
- new `docs/CAPITALCOM_DATA_CONTRACT.md`
- new `docs/CROSS_PLATFORM_RESEARCH_CONTRACT.md`
- new `tools/qlmg_rankable_source_contract.py`
- `tools/run_kraken_family_engine_aggregate_first_sweep.py`, only the smallest compatibility call/import required
- new `tools/capitalcom_data_adapter.py`
- new `unit_tests/test_rankable_source_contract.py`
- new `unit_tests/test_capitalcom_data_adapter.py`
- the task archive under the repository’s established convention
- registry/continuity files only when the task’s verified implementation facts require an update and the exact change is documented

Do not modify existing Kraken family modules, run roots, candidate/control IDs, output ledgers or data files.

## Data allowed

- repository source and documentation;
- existing non-economic unit-test fixtures;
- new synthetic fixtures created for this task;
- metadata-only inspection needed to verify paths and contracts.

## Data excluded

- all real Capital.com market payloads;
- Capital.com API credentials, sessions or account responses;
- protected strategy outcomes;
- any economic result table or return-bearing 2026 data;
- any order/private endpoint.

## Implementation requirements

### 1. Source-neutral manifest authority guard

Create a small module that validates before a supplied payload-reader callback can run. The contract must require at least:

```text
platform
source_dataset_id
purpose
minimum_event_time_utc
maximum_event_time_utc or exclusive interval end
schema_hash
content_sha256
```

It must fail closed for missing/unrecognized platform, wrong selected adapter, unknown/mixed purpose, protected/holdout/calibration/engineering-only purpose in rankable mode, unprovable cutoff, pre-2023-only or mixed intervals, missing hashes, and files reaching 2026+ unless an exclusive end at the cutoff is proved.

Do not design a large framework. Match repository style and expose only the functions needed by the compatibility call and tests.

### 2. Kraken compatibility

Replace no existing Kraken identity or output. Route the current rankable pre-open checks through the new contract using the smallest call possible. Preserve existing Kraken/PF behavior, exact-funding rules and downstream row filtering. Do not generalize unrelated runner code.

### 3. Synthetic Capital.com adapter

The adapter accepts only manifest-authorized synthetic rows and requires:

```text
platform = Capital.com
platform_epic
instrument_type
instrument_name
currency
contract_form
bid_open/high/low/close
ask_open/high/low/close
bar_start_utc
bar_end_utc
availability_utc
calendar_id
metadata_snapshot_hash
expiry_or_undated
market_status
volume_semantic_status
financing_status
corporate_action_status
```

Requirements:

- preserve bid and ask separately;
- never label rows as exchange trades;
- default hypothetical buy execution to ask and sell execution to bid;
- no midpoint fill mode by default;
- fail closed on missing epic/type/calendar/metadata hash or invalid bid/ask ordering;
- keep volume semantics `unverified` unless explicitly verified by fixture authority;
- make financing/corporate-action state explicit known/unknown fields, not zero;
- support a deterministic first-executable-target helper over synthetic calendar/market-status fixtures;
- import no API/order/account client.

### 4. Repository documentation

Update only the approved boundary language. Preserve Kraken-specific manuals as Kraken-specific. State that real Capital.com import remains blocked pending a verified acquisition package, and economics require a separately approved frozen contract.

### 5. Tests

Add focused tests proving:

- protected, mixed, unknown-purpose and hash-unprovable files fail before reader invocation;
- wrong-platform files fail before reader invocation;
- pre-2023 and 2026+ boundary cases fail as required;
- valid Kraken fixture reaches the compatibility reader and existing outputs remain unchanged;
- pre-2023/wrong-platform rows do not reach downstream spies;
- Capital.com bid/ask remain separate;
- buy uses ask and sell uses bid;
- midpoint is not the default;
- missing epic/type/calendar/metadata/expiry status fails closed;
- closed target maps to first executable quote after reopening;
- directed source-to-target identity differs from reverse direction;
- no credential, account or order endpoint dependency exists.

Run the exact current repository-supported baseline, including the previously verified 49 tests, plus all new tests. Record commands, counts, exit codes and failures.

## Milestones

1. Reverify state and create isolated target. Acceptance: safe branch/worktree, current authority and commands recorded. Failure: stop without change.
2. Freeze plan and task archive. Acceptance: objective, files, tests, rollback and approval recorded.
3. Implement source guard and Kraken compatibility. Acceptance: pre-open spy tests and existing Kraken parity pass. Failure: revert compatibility call and leave Capital.com disabled.
4. Implement synthetic adapter and tests. Acceptance: bid/ask, required fields and clock tests pass with no real data. Failure: remove adapter/fixtures.
5. Update bounded docs. Acceptance: no existing Kraken lineage rewritten and no economic authorization implied.
6. Independent review actual diff/tests/artifacts. Acceptance: pass or pass-with-notes; otherwise repair or stop.
7. Create local reviewed commit. Use a descriptive message; do not push.
8. Close archive and verified Drive handoff under `DONCH_BACKTESTING_HANDOFFS`.

## Stop conditions

Stop if current Git state is unsafe, a higher machine contract cannot be reconciled with the approved boundary, real Capital.com data or credentials would be required, protected outcomes may be exposed, existing Kraken IDs/hashes would change, tests require economics, or verification/rollback fails.

## Rollback

Revert only this task’s commit or remove the additive modules/tests/docs and compatibility import/call. Verify the full baseline against the pre-change commit. Do not rewrite or delete historical roots.

## Archive context

- Approval date: 2026-07-18.
- Preflight repository commit: `9df9c4f319cceffce84fc4567ddcaec9333c2bb7`.
- Preflight closed archive: `3f3adc45590cbf7c7f1e9e26f7baf06f066b729345451a65973a88a3eb2ec2f8`.
- Preflight review: `pass_with_external_data_authority_blocker`.
- Existing Kraken decisions and same-sample prohibitions remain binding.
- Real Capital.com import and economics remain blocked.

## Final response

Report status, starting and ending repository identity, branch/worktree, commit, exact diff, commands/test counts, economic/protected/real-data status, artifacts and SHA-256, local task archive, verified Drive folder/package, registries/continuity updates, review, unverified items, prohibited actions not performed, rollback and next bounded task.
