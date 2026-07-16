# Commands And Results

All commands were run on 2026-07-16 UTC.

## Repository And Package Discovery

- `git rev-parse --show-toplevel` in `/opt/testerdonch`: exit 0, root `/opt/testerdonch`.
- `git rev-parse --abbrev-ref HEAD`: exit 0, branch `main`.
- `git rev-parse HEAD`: exit 0, commit `404cd207085406e2dc9e19bbae7558392c750c95`.
- `git status --short --branch --untracked-files=all`: exit 0, dirty checkout observed.
- `git worktree list --porcelain`: exit 0, only original worktree before isolation.
- `git submodule status --recursive`: exit 0, no submodules reported.
- `git config --get core.sparseCheckout`: exit 1, sparse checkout unset.
- `rclone version`: exit 0, `rclone v1.60.1-DEV`.
- `rclone listremotes`: exit 0, remote label `qlmg_sweep_drive:`.
- `rclone lsf qlmg_sweep_drive: --drive-root-folder-id 1eWdmin0E7QKYJ5AvFcpRmvvDG3YiWlBz --format psh --files-only`: exit 0, approved ZIP and transfer manifest found.
- `rclone copy ...`: exit 0, approved transfer files downloaded to `/tmp/codex_governance_20260716/drive_download`.
- `stat` and `sha256sum` on the ZIP: exit 0, bytes `106041`, SHA-256 `fc06159e5caee52e3d5a0521601d31c9fb9fc8c0300a63ecab052b17cd4fbae2`.
- Embedded package manifest verification script: exit 0, 30 package entries verified by size and SHA-256.

## Dirty Checkout Recovery

- Recovery script: exit 0, bundle path `/tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z`.
- Recovery manifest SHA-256: `4298c44952057de78cacd4125a4f8e8b557c8340153fa4b58294a90e4489bbc6`.
- Original status hash after recovery: `704aecb0a4457fbafac6dcf6d927e61dcba24de34b70095664c926b359307251`, matching the archived pre-recovery status hash.
- `git bundle verify committed_refs.bundle`: exit 0.
- Disposable clean-clone recovery patch check:
  - staged patch `git apply --check`: exit 0;
  - staged patch `git apply`: exit 0 with whitespace warnings in recovered user files;
  - unstaged patch `git apply --check`: exit 0;
  - unstaged patch `git apply`: exit 0.

## Isolation And Integration

- `git worktree add -b agent-governance/backtesting-harness-20260716 /opt/testerdonch-agent-governance-20260716 404cd207085406e2dc9e19bbae7558392c750c95`: exit 0.
- Governance files copied from the verified extracted package into the isolated worktree.
- Active docs revised in the isolated worktree only.

## Validation

- Structured Python validation for package hashes, markdown links, skill frontmatter, skill eval counts, active boundary terms, secret patterns, and recovery manifest hash: exit 0.
- `python3 -m json.tool` on copied transfer, package, and recovery manifests: exit 0.
- Forbidden active-authority scan for Bybit-primary or relaxed paid-vendor language: exit 0; only superseded provenance copy matched old Bybit text.
- Original checkout status hash after integration: `704aecb0a4457fbafac6dcf6d927e61dcba24de34b70095664c926b359307251`, still unchanged.
- `python3 -m pytest unit_tests/test_project_deep_cleanup_20260624.py unit_tests/test_sealed_slice_guard.py`: exit 1, `No module named pytest`; no dependency installation was performed.
- `which pytest`: exit 0 with empty output, confirming pytest unavailable on PATH.
- Full `git diff --cached --check`: exit 2 due trailing whitespace in exact received/provenance files.
- Scoped `git diff --cached --check` excluding exact received task/source copies and copied original recovery validation: exit 0.
- Final governance patch `git apply --check` in disposable clone: exit 0.
- Final governance patch `git apply` in disposable clone: exit 0 with whitespace warnings limited to exact provenance files.

## Remote Handoff

- Local ZIP v01: `/tmp/codex_governance_20260716/final_archives/qlmg_backtesting_governance_integration_task_archive_20260716_v01.zip`.
- `unzip -t` on local ZIP v01: exit 0.
- Local ZIP v01 bytes: `223850`.
- Local ZIP v01 SHA-256: `30f6af307097cf9b84a04dc5f567d1e6e27528812015710e5744f694d7e5ad71`.
- Collision check for `qlmg_backtesting_governance_integration_task_archive_20260716_v01.zip`: exit 0, no matching remote file before upload.
- `rclone copy ... qlmg_sweep_drive: --drive-root-folder-id 1eWdmin0E7QKYJ5AvFcpRmvvDG3YiWlBz`: exit 0.
- Remote listing after upload: exit 0, remote file present with bytes `223850`.
- Round-trip download to `/tmp/codex_governance_20260716/roundtrip_download/`: exit 0.
- Local and round-trip SHA-256 matched: `30f6af307097cf9b84a04dc5f567d1e6e27528812015710e5744f694d7e5ad71`.
- `cmp -s` local versus round-trip ZIP: exit 0.
- `unzip -t` on round-trip ZIP: exit 0.

Because this record was written after v01 upload, a final v02 archive was prepared for source-complete handoff without overwriting v01. The final response records v02 upload verification.
