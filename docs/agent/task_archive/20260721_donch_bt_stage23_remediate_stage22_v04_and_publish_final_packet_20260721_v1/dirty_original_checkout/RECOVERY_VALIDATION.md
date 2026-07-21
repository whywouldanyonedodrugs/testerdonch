# Recovery validation

The binding task prohibits copying or mutating `/opt/testerdonch/code`.
Recovery therefore consists of preservation in place plus binary-safe identity
metadata. The path, mode, size, mtime, working SHA-256, expected Git-content
SHA-256, Git blob ID and porcelain status were independently recorded.

The separate Stage-23 worktree was clean at the exact reviewed candidate
commit before task files were added. No implementation or future service path
depends on the dirty binary.
