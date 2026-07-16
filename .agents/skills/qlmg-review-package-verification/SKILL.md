---
name: qlmg-review-package-verification
description: Build, rebuild, validate, archive, or hand off a QLMG backtesting review package. Use when work involves package manifests, authority and supersession registries, recomputation, verification extracts, ZIP naming, secret scans, release-readiness gates, or an authorized Google Drive upload through rclone or equivalent tooling.
---

# QLMG Review Package Verification

## Establish package scope

Read [RUN_AND_ARTIFACT_CONTRACT.md](../../../docs/agent/RUN_AND_ARTIFACT_CONTRACT.md), [REPOSITORY_MAP.md](../../../docs/agent/REPOSITORY_MAP.md), and current package-builder contracts. Resolve authoritative roots and superseded lineages before copying evidence.

Package work does not authorize a new economic screen, strategy re-optimization, protected-outcome inspection, or mutation of finalized source roots.

## Build a truthful local package

1. Write to a new versioned package root.
2. Include a read-first file, authority registry, supersession map, method and strategy cards as required, engineering evidence, schemas, and explicit known gaps.
3. Keep decision-bearing ledgers separate from summaries. Do not treat pooled definition rows as a portfolio.
4. Generate a relative-path manifest with byte sizes and SHA-256 values.
5. Run repository-supported authority, schema, protected-period, recomputation, deterministic replay, hash, and secret checks.
6. Record actual test and failure counts. Do not replace missing numbers with vague source references.
7. Preserve package status independently from hash status. A package with valid hashes may remain blocked.

Require raw event-window verification extracts when the current package contract requires them. Missing trade, mark, index/spot, or exact-funding extracts must remain visible blockers rather than being inferred from derived ledgers.

## Create the handoff ZIP

Name the closed archive:

```text
qlmg_<specific-content-slug>_<YYYYMMDD>_vNN.zip
```

Use UTC date, describe the contents, and increment the version instead of overwriting. Reject ambiguous names such as `review.zip`, `latest.zip`, or `final-bundle.zip`. Generate the ZIP hash after closure and record it outside the archive or in the handoff ledger.

## Gate Google Drive writes

Do not write remotely until all prerequisites are verified:

- the task explicitly authorizes this upload;
- the exact remote and folder are named;
- the authorized write identity is confirmed without exposing credentials;
- the local ZIP and manifest pass;
- collision and overwrite behavior is explicit;
- the selected client supports a documented remote content check.

Do not infer authorization from a configured or readable remote. Do not guess between similarly named remotes. If a prerequisite is missing, return `remote_handoff_blocked` and list the missing items while retaining the local ZIP.

For an authorized upload, use the repository- or environment-supported `rclone` or equivalent workflow. Verify remote content by hash or a documented round-trip download and local comparison; size equality alone is insufficient. Record the remote path, local path, UTC time, size, SHA-256, tool/version, and non-secret identity label. Do not delete or overwrite remote bundles without separate approval.

## Review release readiness

Confirm:

- every intended authoritative lineage is present once;
- superseded economics are not current;
- recomputed metrics match or mismatches are explicit;
- protected-period rows and secret findings are zero;
- required raw verification extracts exist;
- test counts, source snapshots, and recoverable reproducibility hashes are present;
- package and archive hashes validate;
- open gaps and evidence caps remain visible.

The 2026-07-16 extracted package recorded 386 manifest rows and passing hash validation, yet remained `blocked_by_protocol_issue`. Use this as a negative release-readiness test, not as current repository state.

## Report

List source roots, package paths, archive name, sizes and hashes, commands and counts, package status, upload authorization, remote verification, economic-run status, protected-outcome status, and blockers.

## Boundaries

Do not use this skill for:

- ordinary unit-test artifacts that are not a review package;
- generic file compression unrelated to QLMG evidence;
- strategy performance interpretation or threshold selection;
- a Google Drive task that does not involve a QLMG review bundle.
