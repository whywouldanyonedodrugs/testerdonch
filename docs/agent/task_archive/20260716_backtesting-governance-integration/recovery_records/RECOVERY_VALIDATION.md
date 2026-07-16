# Recovery Validation

- `git apply --check STAGED.patch`: rc=1 checked against dirty original; may fail if patch context already applied in worktree
- `git apply --check UNSTAGED.patch`: rc=1 checked against dirty original; may fail if patch context already applied in worktree
- `git bundle verify committed_refs.bundle`: rc=0 
- `manifest`: rc=n/a 
