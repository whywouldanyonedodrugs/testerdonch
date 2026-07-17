# Approved Donch Backtesting Handoff Target

Status: active and round-trip verified on 2026-07-17 UTC.

## Permanent target

- Folder name: `DONCH_BACKTESTING_HANDOFFS`
- Folder URL: `https://drive.google.com/drive/folders/1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`
- Folder ID: `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`
- Non-secret remote/identity label: `qlmg_sweep_drive:`
- Repository-side addressing: `qlmg_sweep_drive:<task-folder>/` with `--drive-root-folder-id 1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`

This is the only standing default destination. The configured identity, remote configuration, tokens, and private account details must not be printed or archived. Do not create or reauthenticate a connection as part of a routine handoff.

## Task folders

Use a new direct child for every completed task:

```text
<YYYYMMDD>_<task-id>_vNN/
```

Before writing, list the exact target root and fail on an existing child with the proposed name. Never overwrite or delete a remote object. Increment `vNN` when a name is occupied; do not reuse, replace, synchronize, or purge an existing folder.

Each task folder may contain no more than five direct files:

- `READ_FIRST.md`
- `VALIDATION.md`
- `REVIEW.md` when review is required
- `TRANSFER_MANIFEST.json`
- one closed task-archive ZIP

Omit files that add no information. Detailed task records belong inside the ZIP. Retain the local ZIP through remote verification and handoff reporting.

## Verification procedure

1. Confirm the task requests the approved default Drive handoff and authorizes the package content.
2. Verify the exact root is readable through the recorded folder ID and non-secret remote label.
3. Select a non-colliding versioned task folder and create it without overwrite behavior.
4. Hash and size every local upload, then upload only the declared direct files.
5. Download every uploaded file to a separate local verification path.
6. Compare downloaded byte sizes and SHA-256 values with the local originals. Size equality alone is insufficient.
7. Record the task-folder ID/URL, exact Drive path, filenames, sizes, hashes, tool version, UTC time, remote label, verification result, retained local archive, and blockers.

Stop with `remote_handoff_blocked` before writing if the exact root, identity, collision state, local package, authorization, or round-trip method cannot be verified.

## Content boundary

Allowed content is a closed, task-scoped backtesting handoff package explicitly requested by the corresponding task, plus its concise read-first, validation, review, and transfer-manifest files.

Never upload credentials, tokens, private identity details, raw protected payloads, raw market datasets, capture payloads, unrelated repository contents, open worktrees, or unclosed archives. This standing authorization applies only to non-overwriting uploads to the exact root above when a task requests the approved default handoff. It does not authorize upload elsewhere, overwrite, deletion, economic work, protected-outcome access, or capture access.

## Setup verification

- Task folder: `20260717_donch_bt_drive_handoff_setup_v01/`
- Task-folder URL: `https://drive.google.com/drive/folders/10YTdexxvqS3qVxBpKLfdDrHSY_Cc5IPb`
- Verification file: `DRIVE_HANDOFF_SETUP_VERIFICATION.md`
- Bytes: `240`
- SHA-256: `f7d132db97b7c543a1119216756accdae158287fde2718bd2af51bfbeab0c7ec`
- Verification: separate round-trip download matched byte size and SHA-256.

## Required final response

Every successful handoff must report:

- exact task-folder URL;
- exact Drive path;
- uploaded filenames;
- byte sizes and SHA-256 values;
- remote verification result;
- non-secret identity label;
- whether the local archive was retained;
- any access or upload blocker.
