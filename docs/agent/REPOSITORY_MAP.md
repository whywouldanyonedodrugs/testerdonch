# Repository Map and Preflight

Status: repository-specific map amended for the approved 2026-07-19 Stage 10 prospective gate-routing policy.

## Verified 2026-07-19 Stage 10 Starting State

- Stage 10 verified `origin/main` at `3ea0d320d71716a5c0890f4c924ed924224beda2`, the supplied Stage 9 authority.
- KDA01 remains `KDA01_level3_repaired_no_primary_pass_stop` and KDA02A remains `KDA02_level3_no_primary_pass_stop`.
- The active policy separates research routing from historical machine decisions and is prospective only.
- Stage 10 is documentation/registry work; it authorizes no economic run, candidate-return access, protected outcome, Capital.com payload, or data acquisition.

## Historical 2026-07-16 Governance Baseline

- Repository root: `/opt/testerdonch`.
- Original branch at task start: `main`.
- Original commit at task start: `404cd207085406e2dc9e19bbae7558392c750c95` (`Cleanup`).
- Isolated governance worktree: `/opt/testerdonch-agent-governance-20260716`.
- Isolated branch: `agent-governance/backtesting-harness-20260716`.
- Base commit: `404cd207085406e2dc9e19bbae7558392c750c95`.
- Sanitized origin: `git@github.com:whywouldanyonedodrugs/testerdonch.git`.
- Submodules: none observed by `git submodule status --recursive`.
- Sparse checkout: unset.
- Applicable repository `AGENTS.md` chain at base: none. This integration adds the root `AGENTS.md`.

The original checkout was dirty at that task's start: 139 staged entries, 4 unstaged entries, 96 untracked entries, and no unmerged entries. Recovery records were produced outside the dirty checkout before that worktree was created. These facts describe the 2026-07-16 governance baseline, not the current repository state.

## Verified 2026-07-18 Multi-Platform Integration State

- Multi-platform source-boundary base commit: `9df9c4f319cceffce84fc4567ddcaec9333c2bb7`.
- Reviewed implementation commit: `73aeb419b43ee8f92423982bd47abd627451d362` on `agent/multiplatform-source-authority-20260718`.
- The reviewed implementation adds the source-neutral authority guard and synthetic Capital.com adapter boundary without changing existing Kraken identities or results.
- The historical 2026-07-16 worktree and commit details above remain provenance only; they are not assertions about the active checkout after the 2026-07-18 integration.

## Current Authority Paths

Use current repository authority in this order:

1. machine-enforced contracts, finalized manifests, and hashes;
2. authoritative run roots, ledgers, registries, and audit artifacts;
3. current continuity brief when present;
4. root `AGENTS.md` and `docs/agent/`;
5. active docs under `docs/`;
6. research reports and hypothesis catalogues as priors;
7. superseded docs, Donch material, and legacy reports as provenance only.

Current active docs:

- `AGENTS.md`
- `docs/agent/DIRTY_REPOSITORY_RECOVERY.md`
- `docs/agent/RUN_AND_ARTIFACT_CONTRACT.md`
- `docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md`
- `docs/agent/EXECUTION_PLAN_TEMPLATE.md`
- `docs/agent/CODE_REVIEW.md`
- `docs/agent/KNOWN_FAILURE_PATTERNS.md`
- `docs/agent/SOURCE_MAP.md`
- `docs/agent/DYNAMIC_CONTINUITY.md`
- `docs/agent/CURRENT_CONTINUITY.md`
- `docs/agent/CURRENT_RESEARCH_DECISIONS.md`
- `docs/agent/RESEARCH_GATE_ROUTING_POLICY.md`
- `docs/agent/RESEARCH_GATE_ROUTING_POLICY.json`
- `docs/agent/HYPOTHESIS_DEVELOPMENT_PROTOCOL.json`
- `docs/agent/ACTIVE_RESEARCH_METHODS.json`
- `docs/agent/FAMILY_REDEFINITION_POLICY.json`
- `docs/agent/RESEARCH_CAMPAIGN_PROTOCOL.json`
- `docs/agent/research_campaigns/kraken_research_campaign_001_readiness/`
- `docs/agent/KRAKEN_DERIVATIVES_LEARNINGS.md`
- `docs/agent/TEST_AND_EVIDENCE_STANDARDS.md`
- `docs/agent/BACKTEST_CLAIMS_AND_REVIEW.md`
- `docs/agent/BACKTESTING_AGENT_OPERATING_AND_ARCHIVE_INSTRUCTIONS.md`
- `docs/agent/CURRENT_FAMILY_HYPOTHESIS_REGISTRY.csv`
- `docs/agent/CURRENT_ATTEMPT_MULTIPLICITY_REGISTRY.csv`
- `docs/agent/CURRENT_RUN_EVIDENCE_REGISTRY.csv`
- `docs/agent/CURRENT_DATA_CAPABILITY_REGISTRY.csv`
- `docs/agent/CURRENT_DEFECT_REPAIR_REGISTRY.csv`
- `docs/agent/KDA_LEARNING_CONTEXT_REGISTER.csv`
- `docs/agent/GATE_ROUTING_MATRIX.csv`
- `docs/QLMG_PERP_PROJECT_STATE.md`
- `docs/QLMG_PERP_BACKTESTING_MANUAL.md`
- `docs/QLMG_PERP_DATA_CONTRACT.md`
- `docs/QLMG_PERP_VALIDATION_PROTOCOL.md`
- `docs/CAPITALCOM_DATA_CONTRACT.md`
- `docs/CROSS_PLATFORM_RESEARCH_CONTRACT.md`

Superseded pre-governance docs are preserved under `docs/agent/superseded/20260716_pre_governance/`.

## Binding Rules

- Approved research platforms: Kraken derivatives and manifest-authorized Capital.com instruments.
- Rankable interval: `2023-01-01T00:00:00Z` inclusive through `2026-01-01T00:00:00Z` exclusive.
- Protected period: `2026-01-01T00:00:00Z` onward.
- July 2026 Kraken capture: strategy-agnostic execution calibration only. Capital.com data from 2026 onward: data-engineering-only by default.
- Real Capital.com import and all economic contracts remain blocked pending their separately verified and approved packages.
- Paid historical vendor data: prohibited.
- Economic runs, protected-outcome inspection, live actions, pushes, merges, destructive Git operations, and remote overwrites require exact task authorization.
- Future gate routing is governed by `RESEARCH_GATE_ROUTING_POLICY.json`. It cannot supersede a historical run contract or terminal decision.

## Repository Commands

Stage 13 outcome-free campaign checks:

```bash
python3 -m unittest unit_tests.test_qlmg_research_campaign -v
python3 tools/build_research_campaign_readiness.py --output <readiness-directory>
```

Discovered from the repository root:

- Dependency pins: `requirements.txt`.
- Unit tests live under `unit_tests/` and `tests/`.
- No `.github/workflows`, `pyproject.toml`, `pytest.ini`, `tox.ini`, `Makefile`, or `.pre-commit-config.yaml` was present at base.

Supported non-economic command pattern for this documentation integration:

```bash
python3 -m pytest unit_tests/test_project_deep_cleanup_20260624.py unit_tests/test_sealed_slice_guard.py
```

Additional documentation and governance checks may be run with repository-local Python one-liners or scripts when they are read-only, non-economic, and recorded in the task archive.

Dynamic continuity checks:

```bash
python3 -m unittest unit_tests.test_donch_continuity -v
python3 tools/donch_continuity.py validate-files --snapshot <snapshot> --pointer <pointer> [--event <event>]
```

The publisher uses the standard library plus the approved environment's `rclone`; it reads no market payload.

Platform-boundary implementation:

- `tools/qlmg_rankable_source_contract.py`: strict source manifest guard and directed identity helper.
- `tools/capitalcom_data_adapter.py`: synthetic bid/ask adapter contract; no API, account, or order client.
- `tools/run_kraken_family_engine_aggregate_first_sweep.py`: explicit legacy Kraken compatibility call preserving existing behavior.

## Archive Conventions

Historical run and review roots are stored under `results/rebaseline/`. For documentation-only governance tasks that should be tracked in source, use:

```text
docs/agent/task_archive/<YYYYMMDD>_<task-id>/
```

Keep large dirty-checkout recovery bundles, binary patches, result roots, parquet data, and remote handoff ZIPs out of the source tree unless a task explicitly authorizes tracking them.

## Drive Handoff

The active approved default Drive target is:

```text
folder name: DONCH_BACKTESTING_HANDOFFS
folder ID: 1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl
remote label: qlmg_sweep_drive:
collision policy: unique task folders only, no overwrite
```

Remote writes require exact task authorization, collision checking, a closed local ZIP, manifest/hash verification, and remote verification by hash or documented round-trip download.
