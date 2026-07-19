# Failed isolated-worktree test attempt

The first broad relevant-suite invocation produced four environment errors because the repository-ignored sealed metadata fixture was absent from the isolated worktree. No protected payload was opened. The existing metadata-only fixture was linked read-only from the main checkout and the final expanded relevant suite then passed 222/222.
