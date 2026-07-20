# Stage 19 Execution Plan

Task: `donch_bt_stage_19_local_official_funding_export_packet_20260720_v2`

Starting authority: clean `origin/main` at `3f5e94eb8e6b8becb4dfaa5457742682ac31f7e9`.

## Scope and safety

- Read only the human-transferred `/opt/testerdonch/research_inputs/exports.zip`.
- Route protected funding rows by timestamp and identity only; do not parse protected rate values numerically or join protected strategy prices/returns.
- Parse rankable rates as `Decimal`; fail closed on integrity, schema, unit, coverage, or determinism failures.
- Regenerate a non-authorizing packet. Do not run Phase 2+ economics, send Telegram messages, inspect Capital.com, or acquire additional bulk data.

## Milestones

1. Build and test ZIP security validation, raw-line partitioning, member/schema/coverage manifests, and campaign-symbol reconciliation.
2. Verify PF units, implement deterministic dual alignment and rankable-only gap allowances, regenerate funding-dependent campaign authority, and run synthetic canaries.
3. Audit official Kraken historical-data metadata and update durable registries/continuity records.
4. Run focused and repository-supported validation, independent review gate, secret scan, archive manifest, and completion records.
5. Make reviewed commits, non-force push, perform approved Drive handoff with round-trip verification, and leave both checkouts clean.

## Rollback

All work is isolated on `agent/stage19-local-official-funding-export-packet-20260720`. Large partitions remain outside Git under a new immutable result root. Prior Stage 16–18 artifacts remain unchanged.
