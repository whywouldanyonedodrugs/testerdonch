# Run and Artifact Contract

This document governs research runs and review-package artifacts. It does not authorize an economic run.

## Authorization gate

Before any economic run, require a task that names the exact frozen hypothesis or mechanism, scope, authority paths, train interval, controls, costs, outputs, hard gates, and final decision vocabulary. If the task only asks for code, tests, documentation, audit, or packaging, do not launch a screen.

Keep `2026-01-01T00:00:00Z` onward sealed from strategy outcomes. Treat 2026 data as strategy-agnostic execution calibration only.

## Frozen run contract

Before outcomes, record and hash where applicable:

- objective, hypothesis, and mechanism;
- definition set and parameter budget;
- Kraken-only venue and `[2023-01-01, 2026-01-01)` train bounds;
- point-in-time data and lifecycle authority;
- candidate economic-address contract;
- control classes and control economic-address contract;
- signal-state contract version;
- costs, slippage scenarios, funding policy, and boundary policy;
- code commit and code hash;
- config, data, universe, and funding hashes;
- expected artifacts, gates, and decision vocabulary.

For rankable event runners, preserve the current signal-state requirements unless a newer machine contract supersedes them:

```text
parent-neutral raw signal tape
-> immutable freeze and hashes
-> PIT parent-policy projections
-> definition-local chronological simulation
-> non-overlap using actual executable exit_ts
-> complete skip ledger
-> accepted-trade freeze
-> control freeze
-> outcome analysis
```

Do not share position state across definitions. Do not preblock with a nominal maximum hold.

## Required artifact properties

- Write to a new versioned run root. Never mutate a finalized source root.
- Use atomic temporary-to-final writes where the repository supports them.
- Preserve a machine-readable manifest with relative paths, byte sizes, and SHA-256 values.
- Record command, environment, start/finish UTC, exit status, peak memory if available, and resume behavior.
- Record candidate/control duplicate-address checks, protected-period counts, boundary reconciliation, PIT checks, funding partitions, and deterministic replay.
- Record tests and failures as numbers. Do not leave them blank while claiming test evidence.
- Keep decision-bearing event and control ledgers separate from summaries.
- Preserve missing fields and blockers explicitly. Do not use a successful hash check as evidence that the package is release-ready.

## Routine task archive

Every substantive repository task must preserve the exact received specification and its `archive_context`, plus the plan, progress/decision logs, changed-file record, commands and results, validation evidence, artifact manifest, review findings, completion report, and next-action record. Use the repository's verified archive convention. If none exists, propose and review a stable path under `docs/agent/task_archive/<YYYYMMDD>_<task-id>/`.

The backtesting agent owns this filing and updates affected registries or continuity documents. The human receives a concise decision, trading sanity-check questions, material risks, approval requests, and paths; the human should not be asked to reconstruct or file the technical record.

## Dirty-repository preservation artifact

When a task begins from a dirty or uncertain checkout, the task archive must link a separately validated recovery bundle created under [DIRTY_REPOSITORY_RECOVERY.md](DIRTY_REPOSITORY_RECOVERY.md). Record staged, unstaged, untracked, conflicted, submodule, excluded-sensitive, and large-file counts. Preserve index and working-tree patches plus an untracked-file manifest, and use an additional recovery mechanism when patches cannot represent the complete state. Record the isolated worktree path, branch, and base commit. The original checkout must remain unchanged.

## Review bundle standard

Create a closed ZIP for human and remote handoff:

```text
qlmg_<specific-content-slug>_<YYYYMMDD>_vNN.zip
```

Rules:

- `YYYYMMDD` is the UTC bundle-creation date.
- `<specific-content-slug>` identifies the actual scope, such as `all-tested-family-review` or `signal-state-repair-evidence`.
- Increment `vNN` instead of overwriting an existing bundle.
- Do not use names such as `review.zip`, `latest.zip`, or `bundle-final.zip`.
- The ZIP must contain a read-first file, package manifest, authority and supersession metadata, verification status, known gaps, and a secret-scan result.
- A full archive may also use another format locally, but the handoff artifact required by this contract is a ZIP.
- Generate the archive hash after closure and record it in a sidecar or handoff ledger. An archive cannot truthfully contain its own final hash.

The 2026-07-16 extracted external-review package is precedent for structure, not proof of release readiness. Its recorded status was `blocked_by_protocol_issue`; raw event-window trade, mark, index/spot, and exact-funding extracts were absent, test/failure counts were blank, one source snapshot was missing, and five family lineages lacked reproducibility hashes.

## Google Drive handoff gate

Remote handoff is expected when an authorized bundle task supplies all prerequisites. Before using `rclone` or another supported client, verify:

1. the task explicitly authorizes upload;
2. the exact remote and folder are named and readable;
3. the authorized write identity is confirmed by the operator or approved environment configuration;
4. credentials remain hidden;
5. the local ZIP is closed, uniquely named, and hash-verified;
6. overwrite and collision behavior is explicit;
7. the client and remote support a documented content-verification method.

If any prerequisite is missing, stop before the first remote write and report `remote_handoff_blocked` with the missing items. Do not reuse a similarly named remote. In July 2026, `qlmg_sweep_drive:` was readable on one audited machine, while the expected `qlmg_gdrive:` remote was not configured; neither fact authorizes a future write.

After an authorized upload:

- verify remote size and content hash, or perform a documented round-trip download and local hash comparison;
- record remote path, local path, UTC time, archive size and hash, tool/version, and non-secret identity label;
- retain the local bundle until remote verification and manifest recording pass;
- do not delete or overwrite remote files without separate approval.

Once a stable Drive archive root and write identity receive one-time approval, task specifications may authorize routine uploads to that exact root without asking the human to copy files. The agent still must use a unique path, verify content, record the transfer, and stop on a collision or configuration mismatch.

## Final run record

Report exact commands and results, artifact paths and hashes, local and remote archive paths, registry/continuity updates, test counts, package status, protected-period row count, secret-scan result, remote handoff status, and every unresolved gap. State explicitly whether a new economic run occurred and whether protected outcomes were inspected.
