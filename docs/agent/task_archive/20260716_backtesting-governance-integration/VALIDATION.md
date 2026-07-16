# Validation

## Passed

- Approved ZIP byte count and SHA-256 matched the task and transfer manifest.
- All 30 embedded package entries matched the package manifest by size and SHA-256.
- Dirty checkout recovery bundle was created without changing the original checkout.
- Original checkout status hash matched before and after recovery and after integration.
- Recovery Git bundle verified.
- Staged and unstaged recovery patches applied in order on a disposable clean clone.
- Isolated worktree was created from the verified base commit.
- Structured validation passed for adapted markdown links, skill frontmatter, skill eval counts, active authority terms, JSON manifests, package hashes, recovery manifest hash, and high-risk secret patterns.
- Active docs consistently encode Kraken-only authority and the protected interval cutoff.
- Scoped whitespace validation passed for adapted repository docs and task-authored records.
- Full governance patch applied successfully in a disposable clean clone.

## Accepted Provenance Exceptions

- Exact received-source copies under `received/` contain original relative links to files not included in the approved package. These files are preserved as provenance and excluded from repository-relative link validation. See `ARCHIVED_SOURCE_LINK_POLICY.md`.
- Exact received-source copies and the exact task specification contain trailing whitespace. Full `git diff --check` reports those lines; scoped validation of adapted files passes. The whitespace was retained to preserve provenance.
- Exact Drive IDs are retained in task/provenance/handoff records because the task required exact transfer preservation and remote verification. See `DRIVE_IDENTIFIER_RETENTION.md`.

## Unavailable

- Focused pytest command could not run because `pytest` is not installed in the environment. No dependency installation was performed during this governance task.

## Not Run

- No economic backtest, alpha screen, parameter sweep, outcome-bearing validation, protected-period strategy inspection, live capture, private endpoint action, order placement, or risk change.
- No push, merge, rebase, reset, restore, clean, stash, or remote overwrite.
