# Decisions And Progress

## Intake

- Starting commit: `9df9c4f319cceffce84fc4567ddcaec9333c2bb7`.
- Isolated branch: `agent/multiplatform-source-authority-20260718`.
- Input ZIP SHA-256: `58991dad07d1e3563f101ef784a9ac5b50fb2ce95fc52126584d29787b34d89b`.
- `INPUT_MANIFEST.json`: 17 records verified, 0 failures.
- The first local validator expected a `files` key; the authority schema uses `records`. A later archive-report script also initially omitted the ZIP's top-level path prefix. Both verifier bugs were corrected without changing the package; the retained machine report resolves the prefix and all 17 declared byte counts and SHA-256 values pass.

## Design Decisions

- The new generic rankable-source guard is strict and requires the approved platform, purpose, interval, schema hash, and content hash fields before invoking a payload reader.
- Existing Kraken authority objects remain supported only through an explicit legacy-Kraken compatibility function so current identities and loader behavior do not change.
- The Capital.com adapter contains no API, authentication, account, order, or real-data acquisition code. Tests use synthetic fixtures and reader spies only.

## Verification Status

- Required corrected baseline plus new tests: 65/65 pass.
- Complete unit-test discovery: 1,162/1,165 pass. The three failures reproduce unchanged on the starting `main` commit and are outside this task's files.
- Independent staged-diff review: pass with notes; no task-scope correctness, leakage, credential, account/order, or protected-data finding.
