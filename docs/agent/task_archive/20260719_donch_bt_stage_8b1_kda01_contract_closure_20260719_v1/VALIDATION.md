# Validation

Status: pass; decision `ready_for_human_KDA01_Level3_run_approval`.

## Authority and package closure

- Stage 8B source commit: `2a3d38545600eb39f70f91180fb237bc436a1ece`.
- Stage 8B manifest content hash: `569bb1c53ba58767091c42dbf96ead63b3df5a45ae7cad1b4f6d4425e5ef61d5`; recomputation passed.
- Stage 8B manifest file SHA-256: `ee7db729eac30363c6147984658777ed92dc06f1d436527a56efae5bb997f669`.
- Reconciled source objects: `52` archive files plus `748` cache files; missing/mismatched: `0/0`.
- Source parent/event tape SHA-256 values: `ae470d06c7da049d145ac8e083d0c4711c9c2dd6e4ec32fedf79224e8885ddbd` / `7c9e682380681aa6fc83161ae29dafab9a047bdb19fe1fa8ade3ba5a4eb9c2e5`.
- Exact source manifest snapshot included at `source_stage8b/ARTIFACT_MANIFEST.json`.

## Independent clusters

- Event rows reconciled: `102,136/102,136`; duplicate event IDs: `0`.
- Market-day clusters: `1,928` (`960` primary; `968` robustness).
- Six-hour sensitivity clusters: `7,440` (`3,699` primary; `3,741` robustness).
- Primary market-day counts by year: `240 / 356 / 364` for 2023/2024/2025.
- Robustness market-day counts by year: `240 / 364 / 364`.
- Same attempt and UTC date always maps to exactly one market-day identity; primary and robustness never share an identity.
- Full cluster-ID recomputation passed for every distinct onset identity.

## Timestamp-only execution

- Definition-event records: `204,272` (`102,136` events x two frozen timeouts).
- Accepted: `183,744`.
- Rejected/skipped: `20,528`: `20,473` actual-position overlaps and `55` missing exit bars.
- Entry-delay exceeded / exit-delay exceeded / missing entry: `0 / 0 / 0`.
- Definition-event duplicates: `0`; protected timestamps: `0`.
- Reader-spy tests prove only `time`, `venue_symbol`, `resolution`, `rankable_pre_holdout`, and `contains_protected_period` are opened. Price fields are not requested.
- All eight primary definitions pass events, annual events, symbols, symbol-share, duplicate, and protected-row gates after filtering and actual-exit non-overlap.

## Contract and replay

- Frozen Level-3 v2 contract hash: `d2112d8fb4f0666890761b70f7fa71c9e8cb4f43c704a04cdf9470e949a177b3`.
- Definitions: `16`; controls: `7`, unchanged and unexecuted.
- Primary inference unit: `market_day_cluster_id`; parent episode and six-hour cluster are sensitivity-only.
- Complete core replay matched byte-for-byte for the cluster tape, cluster summary, execution counts, rejection ledger, definition register, and contract JSON.
- Integrity validator passed: `102,136` events, `183,744` accepted records, `20,528` rejected/skipped records, `0` protected rows, `0` economic outputs.

## Tests and resources

- Compilation: passed.
- Focused Stage 8B1 tests: `12 passed`.
- Relevant Stage 8A/8B, loader, sealed, lifecycle, C01/C02, analytics, and Stage 8B1 suite: `222 passed` in the final expanded invocation.
- First broad test invocation had four environment errors because ignored seal-metadata fixtures were absent from the isolated worktree; the existing repository fixtures were linked without payload inspection and the complete final suite then passed `222/222`.
- Authoritative build: `2m11.55s`; `/usr/bin/time` peak RSS `1,240,416 KiB`; no swap.
- Protected rows opened: `0`; economic outputs computed: `0`.
