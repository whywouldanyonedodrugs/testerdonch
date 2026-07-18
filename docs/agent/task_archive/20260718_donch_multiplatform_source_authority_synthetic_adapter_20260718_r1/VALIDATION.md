# Validation

## Result

`pass_with_preexisting_suite_failures`

The task's required corrected baseline and new tests pass 65/65. Complete repository unit-test discovery was also executed. It passes 1,162 of 1,165 tests; all three remaining failures reproduce unchanged on the starting `main` commit and do not touch the bounded task diff.

## Contract Evidence

- Input authority package: 17/17 manifest records pass byte-count and SHA-256 verification.
- Strict source manifest guard rejects missing/unrecognized platform, wrong adapter, unrankable purpose, unprovable interval, pre-2023/mixed interval, protected interval, missing hashes, and non-exact funding before the reader callback.
- Generic row filtering prevents pre-2023 and wrong-platform rows from reaching the downstream spy and rejects protected payload rows.
- Existing Kraken loader/funding tests pass through the explicit legacy compatibility validator without changing row filtering or output fixtures.
- Capital.com tests preserve bid/ask, apply buy-at-ask and sell-at-bid, provide no midpoint mode, validate identity/calendar/metadata/expiry and bid/ask ordering, preserve explicit volume/financing/corporate-action semantics, and map closed periods to the first executable quote.
- Directed cross-platform identity is asymmetric.
- New operational modules import no API, credential, account, or order client.

## Safety Evidence

- Real Capital.com payloads accessed: no.
- Economic research or strategy outcomes read: no.
- Protected outcomes read: no.
- Kraken identities or results modified: no.
- Push, merge, deployment, order endpoint, or account action: no.

Detailed command output is retained under `validation/`.
