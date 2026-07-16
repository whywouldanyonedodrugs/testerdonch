# Clean Clone Patch Apply Validation

Purpose: verify dirty-checkout recovery patches outside the original dirty checkout.

Disposable clone:

```text
/tmp/codex_governance_20260716/recovery_patch_check_20260716T1554
```

Commands and results:

```text
git clone --no-hardlinks /opt/testerdonch /tmp/codex_governance_20260716/recovery_patch_check_20260716T1554
exit 0

git apply --check /tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z/STAGED.patch
exit 0

git apply /tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z/STAGED.patch
exit 0
note: Git reported whitespace warnings in recovered user files, but no apply failure.

git apply --check /tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z/UNSTAGED.patch
exit 0

git apply /tmp/codex_governance_20260716/recovery/dirty_checkout_recovery_20260716T153648Z/UNSTAGED.patch
exit 0
```

Observed restored status:

```text
git status --short --untracked-files=all | wc -l
139

git diff --name-status | wc -l
3
```

Interpretation:

- The staged patch applies cleanly to a clean clone at the original base.
- The unstaged patch applies cleanly after the staged patch.
- Untracked files are not represented by Git patches and are preserved separately by `UNTRACKED_MANIFEST.json` and safe copied text files in the external recovery bundle.
- Earlier `git apply --check` failures recorded in `RECOVERY_VALIDATION.md` were caused by checking patches against the already-dirty original checkout, where added paths already existed.
