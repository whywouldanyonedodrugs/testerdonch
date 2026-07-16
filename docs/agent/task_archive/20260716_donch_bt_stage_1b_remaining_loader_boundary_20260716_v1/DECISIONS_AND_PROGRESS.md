# Decisions and Progress

- 2026-07-16: Verified branch `fix/rankable-loader-boundary-20260716` at `fc1113d61683e41e7cf9aa76b75b87933a70897c`; local and remote main both `992e7928d0dd948c0bb3f3fc3c74b1095648df1b`.
- 2026-07-16: Scope frozen to one production file, one focused test file, and this archive.
- 2026-07-16: Existing manifest authority will be bound locally by `data_paths()`; no catalog or partition abstraction will be introduced.
- 2026-07-16: Pre-patch focused tests reproduced 8 assertion failures and 5 errors across the three readers plus missing metadata binding.
- 2026-07-16: Focused, owning-module, guard, compile, metadata-only real-manifest, and independent AST/diff checks passed.
- 2026-07-16: Readiness rerun closed the loader blocker and selected an outcome-free U2 lifecycle preflight as the smallest next task.
