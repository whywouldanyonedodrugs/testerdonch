# Proposed Implementation Task Sequence

Each future task has one observable outcome. None is authorized by this preflight.

## 1. Apply multi-platform authority and source-manifest contract

Outcome: one approved repository contract plus a source-neutral manifest validator and synthetic tests.

Acceptance: protected/mixed/unprovable files fail before reader invocation for both source labels; pre-2023/wrong-platform rows never reach downstream spies; all existing Kraken loader tests pass unchanged.

Failure response: stop without enabling Capital.com paths. Rollback: remove the new module and compatibility call; verify Kraken test parity. Drive handoff: code/test report only.

## 2. Intake Capital.com acquisition metadata

Outcome: verified instrument/dataset/schema/coverage manifests, no market-payload import.

Acceptance: every file has path, size, SHA-256, interval, purpose, epic, resolution, schema, legal-environment label and protected classification; credentials absent; 2026+ physically/manifest-separated.

Failure response: reject the handoff. Rollback: remove only the new intake archive. Drive handoff: metadata package only.

## 3. Implement Capital.com bid/ask adapter

Outcome: adapter normalizes synthetic and explicitly authorized small schema fixtures without scoring.

Acceptance: bid/ask preserved; correct side selected for hypothetical entry/exit helper tests; calendars/availability enforced; missing financing/corporate-action semantics remain blockers; peak outputs contain no economics.

Failure response: fail closed before normalized data publication. Rollback: remove adapter and its fixtures. Drive handoff: tests/schema report.

## 4. Freeze instrument identity and directed relationship contract

Outcome: small reviewed mapping for named research relationships only.

Acceptance: no all-pairs generation; effective intervals and provenance required; existing Kraken IDs unchanged; direction reversal produces a distinct contract ID.

Failure response: leave cross-platform mode disabled. Rollback: remove additive map version. Drive handoff: manifest and validation.

## 5. Run a non-economic clock/execution mechanical canary

Outcome: prove source availability, target reopening, first executable bid/ask quote, and protected-boundary behavior on fixtures or explicitly authorized non-outcome slices.

Acceptance: no returns, ranking or parameter selection; deterministic source-to-target timestamps; all boundary spies pass.

Failure response: repair timing only. Rollback: discard canary artifacts. Drive handoff: mechanical evidence.

Only after these tasks and a separately approved frozen hypothesis may a bounded Capital.com or directed cross-platform economic task be proposed. A broad all-instrument screen is explicitly excluded.
