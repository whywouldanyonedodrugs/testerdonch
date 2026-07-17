# Execution Plan

## Objective

Determine whether free official sources support an auditable point-in-time C16 daily flow/creation authority for U.S.-listed spot-backed BTC and ETH products in 2024-2025.

## Non-goals and hard boundaries

- No Kraken price, funding, return, signal, control, ranking, or economic access.
- No observation dated on or after `2026-01-01` may be downloaded or opened.
- No mixed current/historical observation payload may be opened unless the source can enforce the protected cutoff before transfer.
- No third-party values, gap fills, paid data, capture access, strategy code, or parameter design.
- Existing first-wave decisions and roots remain immutable.

## Authority

- Starting commit: `e905a25a82582b8b6e436329c73c3a2117e793a6`.
- Task specification: `TASK_SPEC.md` and its preserved attachment path.
- Repository rules: `AGENTS.md`, `docs/agent/DATA_AND_PROTECTED_PERIOD_RULES.md`, and `docs/agent/RUN_AND_ARTIFACT_CONTRACT.md`.
- Source order: official issuer/administrator, official exchange/service, SEC/regulator, official prospectus/methodology.

## Milestones

1. **Preflight and freeze**
   - Verify clean main and accepted closure commit; create an isolated task branch.
   - Acceptance: exact commit match and task record exists.
   - Failure: stop without acquisition.
2. **Official-source inventory**
   - Discover official product, mechanics, historical-file, publication, revision, and terms surfaces without opening protected observation payloads.
   - Acceptance: every source has URL, access metadata, protected-payload classification, and rights conclusion.
   - Failure: record precise authority blocker; do not substitute third-party values.
3. **Bounded acquisition and panel build**
   - Acquire only sources whose server-side scope or dated artifact proves pre-2026 content; retain hashes and raw evidence locally.
   - Build first-published and latest-revised panels only from authoritative dated values. Empty schema-valid panels are permitted only when the decision is authority-unavailable.
   - Acceptance: deterministic manifests and all feasibility counts are explicit.
   - Failure: fail closed and preserve source metadata.
4. **Synthetic verification and independent review**
   - Test publication timing, revisions, measure semantics, lifecycle, coverage, protected/mixed rejection, and deterministic replay.
   - Acceptance: focused tests pass and independent review has no unresolved critical finding.
5. **Closure, integration, and handoff**
   - Update only factual continuity/data/multiplicity records, create reviewed commits, push non-force, package approved compact handoff, and round-trip verify.
   - Acceptance: local/remote hashes match and remote path is unique.
   - Failure: retain local package and report the exact publication or handoff blocker.

## Expected repository changes

- One bounded C16 source/panel builder and synthetic unit tests if needed.
- This task archive and required C16 authority outputs.
- Factual first-wave continuity/data/source/multiplicity records only.
- Compact handoff metadata and ZIP; large raw evidence and Parquet remain local.

## Risks and responses

- Official sites may expose only current mutable values: classify unavailable; do not infer history.
- Historical files may lack publication/revision versions: exclude from first-published rankable authority.
- Terms may be ambiguous: stop with exact non-economic remedy.
- SEC filings may be authoritative but too sparse: inventory truthfully; do not relabel periodic AUM as daily flow.
- Protected observations may coexist in an endpoint: reject before body transfer/open.

## Rollback

All work is additive on a task branch. Existing roots are never modified. Remote objects use a new collision-checked folder and are never overwritten or deleted.
