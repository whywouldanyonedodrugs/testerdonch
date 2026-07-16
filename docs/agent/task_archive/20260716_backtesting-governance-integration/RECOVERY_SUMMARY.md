# Recovery Summary

Original checkout:

- Root: `/opt/testerdonch`.
- Branch: `main`.
- Commit: `404cd207085406e2dc9e19bbae7558392c750c95`.
- Sanitized origin: `git@github.com:whywouldanyonedodrugs/testerdonch.git`.
- Staged entries: 139.
- Unstaged entries: 4.
- Untracked entries: 96.
- Unmerged entries: 0.

Recovery bundle:

- Path: `/tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z`.
- Size: 168 MB.
- Manifest: `RECOVERY_MANIFEST.json`.
- Manifest SHA-256: `4298c44952057de78cacd4125a4f8e8b557c8340153fa4b58294a90e4489bbc6`.
- Contains binary-capable staged and unstaged patches, untracked-file manifest and safe text copies, ignored-path metadata, status records, sanitized remotes, and `committed_refs.bundle`.

Repository archive records:

- `recovery_records/RECOVERY_MANIFEST.json`.
- `recovery_records/RECOVERY_VALIDATION.json`.
- `recovery_records/RECOVERY_VALIDATION.md`.
- `recovery_records/UNTRACKED_MANIFEST.json`.
- `recovery_records/git_status_porcelain_v2_before.txt`.
- `recovery_records/git_status_short_before.txt`.
- `recovery_records/git_staged_name_status.txt`.
- `recovery_records/git_unstaged_name_status.txt`.
- `recovery_records/git_remote_v_redacted.txt`.

Validation note:

- The original checkout status SHA-256 matched before and after recovery: `704aecb0a4457fbafac6dcf6d927e61dcba24de34b70095664c926b359307251`.
- The recovery Git bundle verified successfully.
- Patch apply was later checked in a disposable clean clone after applying staged then unstaged patches in order. Evidence: `recovery_records/CLEAN_CLONE_PATCH_APPLY_VALIDATION.md`.

Original checkout mutation status:

- Worktree files were not changed.
- No reset, restore, clean, stash, commit of unknown work, push, merge, or destructive operation was performed.
