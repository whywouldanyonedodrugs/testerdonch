# Commands and Inspections

All commands were read-only except creation of this task archive and the authorized Drive handoff performed after closure.

| Command/inspection | Result |
|---|---|
| `git rev-parse --show-toplevel; git status --porcelain=v2 --branch; git worktree list --porcelain; git submodule status --recursive` | Pass; clean `main`, HEAD/upstream `9df9c4f...`, no submodules |
| `sha256sum research_inputs/donch_capitalcom_expansion_preflight_20260718_v01.zip; unzip -l ...` | Pass; 125075 bytes, SHA-256 `75dd...0d18`, 18 archive entries |
| Python `zipfile` validation of `INPUT_MANIFEST.json` | Pass; 17/17 size and SHA-256 records |
| `find ... AGENTS.md`, authority/document discovery | Pass; root instruction chain resolved |
| `rclone lsf qlmg_sweep_drive: --drive-root-folder-id ...` | Pass; approved root readable |
| `rg`/`nl` inspection of loader, evidence, lifecycle, C01/C02, controls, run-manifest and reporting code | Pass; no economic execution |
| Metadata-only `find` of `/opt/parquet` and Capital.com path candidates | Pass; Kraken roots present; no Capital.com dataset root found |
| `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest unit_tests.test_rankable_loader_boundary unit_tests.test_kraken_readiness_repair unit_tests.test_qlmg_evidence_contracts` | Command error: 20 valid tests passed, nonexistent `unit_tests.test_qlmg_evidence_contracts` caused one discovery error |
| Corrected two-module unittest command | Pass; 20/20 |
| `unit_tests.test_qlmg_signal_state_contract unit_tests.test_qlmg_mechanical_qa_evidence_contract` | Pass; 29/29 |
