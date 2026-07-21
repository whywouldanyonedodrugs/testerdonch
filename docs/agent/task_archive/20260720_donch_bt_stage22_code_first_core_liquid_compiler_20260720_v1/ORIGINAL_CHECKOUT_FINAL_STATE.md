# Original checkout final state

The original checkout was verified clean at task start at `main` and
`origin/main` commit `56059c1a2e86c91c32087bd4e4da9257b632fbd9`.

At final verification, the refs remained at that exact commit but the tracked
binary `code` was modified in `/opt/testerdonch`:

- Git status: `.M code`;
- working-tree SHA-256:
  `d24aad2612fb79bb0893e13b9cac2592539ac9c783ad95c3b00fafc64bb37b1b`;
- tracked/task-worktree SHA-256:
  `3e6dc67b93eacbd316e9d91b0f2e23195d6a78cf75b5deb27970b260e4ffe297`;
- both files are 32,732,320-byte ELF executables;
- working-tree mtime: `2026-07-20T22:59:50.419175457Z`;
- isolated task-worktree mtime: `2026-07-20T21:51:14.965289002Z`.

No Stage-22 command edited or installed this binary, and its ownership is
unknown. It was preserved exactly: no restore, checkout, deletion, staging or
commit was performed. The isolated Stage-22 task worktree is clean. The
original checkout cannot truthfully be reported clean until the binary change
is attributed and resolved by its owner.
