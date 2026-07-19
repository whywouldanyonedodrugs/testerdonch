# Commands and Results

- Compilation and focused synthetic tests: passed.
- Exact authority/hash preflight and schedule reconstruction: passed `204272/183744/20528`.
- First invocation: failed on worktree-relative funding root after price scoring and before funding/gates; preserved.
- Repaired authoritative invocation: exit 0 in 4m24.69s; peak RSS 2,463,012 KiB; no swap.
- Independent report/bootstrap recomputation: passed.
- Deterministic 100-row exact-open source check: passed.
- Funding boundary duplicate/protected checks: zero/zero.
- Final relevant regression suite: 171/171 passed after linking two repository-ignored metadata-only sealed fixtures.
- Compilation, task Markdown links, manifest checks, secret scan, and `git diff --check`: passed.
