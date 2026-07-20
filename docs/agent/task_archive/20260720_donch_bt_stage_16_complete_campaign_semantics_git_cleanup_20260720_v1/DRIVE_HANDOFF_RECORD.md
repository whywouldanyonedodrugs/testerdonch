# Drive Handoff Verification Record

Status: pass. This is a post-closure record and is intentionally outside the already closed ZIP and its 33-file artifact manifest.

- UTC upload completion: `2026-07-20T06:26:11Z`
- Remote identity label: `qlmg_sweep_drive:`
- Approved root folder ID: `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`
- Drive path: `DONCH_BACKTESTING_HANDOFFS/20260720_donch_bt_stage_16_complete_campaign_semantics_git_cleanup_20260720_v1_v01/`
- Task folder ID: `1cRJSGLWTcaBNTon4effRpzRLHSpR8LCc`
- Task folder URL: `https://drive.google.com/drive/folders/1cRJSGLWTcaBNTon4effRpzRLHSpR8LCc`
- Collision check: proposed folder absent before write.
- Write mode: unique new folder, `--immutable`, no overwrite.
- Client: `rclone v1.60.1-DEV`.

Uploaded and round-trip verified:

| File | Bytes | SHA-256 |
|---|---:|---|
| `READ_FIRST.md` | 587 | `4fc24e167b8eb05996947a6bca9dd392aba667b5d7c401f7c5f59b4fb30817b5` |
| `REVIEW.md` | 268 | `a6ce6ef73ca0fdd99e2d66466f1e475aa7f2bf01e1195ba6489825c89677c411` |
| `TRANSFER_MANIFEST.json` | 1141 | `b6b4b615dcbd0490391491cdc6260e16962e0f10d2d18260dd59645ae663d428` |
| `VALIDATION.md` | 414 | `766915be2a720879758053eaeafc10f88370cee28c8794eb70b06fc4b361fe09` |
| `qlmg_stage16_complete_campaign_semantics_20260720_v01.zip` | 64268 | `9ce51afe35b930b69976fa08d327f34b0290bd52d7fda8662c326338d6cb2e08` |

Verification used a separate download root `/opt/stage16-drive-verify-PF7aRi`; every downloaded file matched its retained local original with both `cmp` and SHA-256. Local handoff files remain at `/opt/stage16-handoff-20260720-v01`.
