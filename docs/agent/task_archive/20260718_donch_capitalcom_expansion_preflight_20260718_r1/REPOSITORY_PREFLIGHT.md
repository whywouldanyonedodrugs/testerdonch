# Repository Preflight

## Verified identity

- Root: `/opt/testerdonch`
- Branch: `main`
- HEAD: `9df9c4f319cceffce84fc4567ddcaec9333c2bb7`
- Subject: `Record Stage 7C running handoff`
- Upstream: `origin/main`
- Ahead/behind: `0/0`
- Origin: `git@github.com:whywouldanyonedodrugs/testerdonch.git`
- Working tree at start: clean
- Submodules: none
- Sparse checkout: disabled
- Applicable instruction chain: root `AGENTS.md`

A separate old governance worktree exists at `/opt/testerdonch-agent-governance-20260716`; it was not modified.

## Current authority

Current machine/repository contracts retain:

- active venue: Kraken only;
- rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`;
- protected period: `2026-01-01T00:00:00Z` onward;
- 2026 capture: execution calibration only;
- paid historical vendor data: prohibited.

## Local Capital.com state

Only the planning ZIP was found. No `/opt/parquet/capitalcom` or equivalent Capital.com data root, dataset manifest, schema, instrument inventory, or downloader output is present on this host.

## Commands and archive conventions

- Python: repository `.venv/bin/python`
- Unit tests: `python -m unittest ...`
- No repository CI/config framework supersedes this command evidence.
- Durable archive convention: `docs/agent/task_archive/<YYYYMMDD>_<task-id>/`
- Approved Drive destination: `DONCH_BACKTESTING_HANDOFFS` using non-secret label `qlmg_sweep_drive:` and root folder ID `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`.
