# Remote Handoff

Approved destination:

- Remote label: `qlmg_sweep_drive:`.
- Folder ID: `1eWdmin0E7QKYJ5AvFcpRmvvDG3YiWlBz`.
- Collision policy: unique names only, no overwrite.
- Tool: `rclone v1.60.1-DEV`.
- Non-secret identity label: configured `qlmg_sweep_drive:` remote. Credential contents were not inspected or copied.

Verified uploaded archive v01:

- Local path: `/tmp/codex_governance_20260716/final_archives/qlmg_backtesting_governance_integration_task_archive_20260716_v01.zip`.
- Remote path: `qlmg_sweep_drive:qlmg_backtesting_governance_integration_task_archive_20260716_v01.zip` under the approved folder ID.
- Bytes: `223850`.
- SHA-256: `30f6af307097cf9b84a04dc5f567d1e6e27528812015710e5744f694d7e5ad71`.
- Upload command: `rclone copy /tmp/codex_governance_20260716/final_archives/qlmg_backtesting_governance_integration_task_archive_20260716_v01.zip qlmg_sweep_drive: --drive-root-folder-id 1eWdmin0E7QKYJ5AvFcpRmvvDG3YiWlBz`.
- Upload result: exit 0.
- Remote listing after upload: present with bytes `223850`.
- Round-trip download path: `/tmp/codex_governance_20260716/roundtrip_download/qlmg_backtesting_governance_integration_task_archive_20260716_v01.zip`.
- Round-trip SHA-256: `30f6af307097cf9b84a04dc5f567d1e6e27528812015710e5744f694d7e5ad71`.
- Byte comparison: `cmp_exit=0`.
- Round-trip ZIP test: `unzip -t` exit 0.

Final source-complete handoff:

- This file was added after v01 verification, so a v02 archive was prepared and uploaded without overwriting v01.
- The final response records the v02 local path, remote path, size, SHA-256, and round-trip verification result.
