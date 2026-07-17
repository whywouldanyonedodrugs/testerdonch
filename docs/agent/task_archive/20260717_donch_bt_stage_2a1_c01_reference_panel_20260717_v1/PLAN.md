# Stage 2A1 Execution Plan

## Objective And Non-Goals

Build the bounded `kraken_c01_reference_panel_v1` authority and close the missing
2025-12-31 trade/mark metadata interval for `PF_XBTUSD` and `PF_ETHUSD`. This task
does not implement C01 features, candidates, controls, returns, funding, a
candidate universe, or any economic evaluation.

## Frozen Inputs And Boundaries

- Base commit: `2e9630646299757a8b5465f7ee1bd63fb5fa5d58`.
- Exact candle requests: `{trade,mark} x {PF_XBTUSD,PF_ETHUSD}` only.
- Request interval: `from=1767139200`, `to=1767225599`.
- Protected start: `1767225600000` milliseconds; any returned row at or after it fails before normalization.
- Terminal source: official cumulative Kraken derivatives suspensions/delistings page.
- Existing authority: accepted Stage 2A source snapshots and download-manifest metadata only.
- No fallback to the known mixed 2025/2026 chunk.

## Expected Changes

- `tools/build_kraken_c01_reference_panel_authority.py`
- `unit_tests/test_kraken_c01_reference_panel_authority.py`
- this task archive, including retained source provenance, normalized bounded slices,
  required authority outputs, validation/review records, manifest, and closed handoff.

## Milestones And Failure Responses

1. Acquire the four exact slices and official lifecycle page. Accept only HTTP 200,
   unambiguous schema/identity, `more_candles=false`, strict timestamp order, no
   duplicates, and no protected row. Otherwise preserve provenance and stop blocked.
2. Normalize under the existing Kraken candle columns and report, never fill, any
   five-minute gap. Trade and mark remain separate.
3. Parse the cumulative terminal-event ledger. Absence from that ledger is only
   negative terminal-event evidence as of access time, never a no-outage claim.
4. Emit the two-member factor/reference panel with `continuous_tradeability_claim=no`
   and `survivorship_free_claim=no`.
5. Run synthetic focused tests, compilation, repository guards, hash validation,
   secret scan, and diff review. Any substantive defect blocks integration.
6. Commit once, fast-forward `main`, non-force push `origin/main`, create a closed
   package that excludes raw market rows from Drive, and round-trip verify the
   approved non-overwriting Drive handoff.

## Rollback And Retention

The task branch isolates all repository changes. No prior artifact is modified or
deleted. Raw official responses and local normalized authority slices remain in the
repository task archive; the Drive package omits raw market payloads under the
standing handoff prohibition.
