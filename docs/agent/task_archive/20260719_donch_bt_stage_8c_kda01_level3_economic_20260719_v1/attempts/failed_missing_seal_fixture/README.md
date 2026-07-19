# Missing ignored seal-metadata fixture

The first relevant-suite invocation had four environment errors because the isolated worktree lacked the repository-ignored seal manifest. No assertion failed and no protected payload was opened. The existing metadata-only fixture was linked read-only from the main checkout before repeating the identical suite.

The first repeat then identified the companion ignored `data_manifest_hashes.json`; it was linked under the same metadata-only rule before the final repeat.
