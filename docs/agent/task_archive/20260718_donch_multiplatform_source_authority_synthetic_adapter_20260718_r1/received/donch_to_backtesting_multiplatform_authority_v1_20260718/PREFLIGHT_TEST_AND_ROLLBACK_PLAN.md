# Test and Rollback Plan

## Required implementation tests

- Protected, mixed, unknown-purpose and hash-unprovable files rejected before payload reader invocation.
- Pre-2023 and wrong-platform rows removed before feature/downstream spies.
- 2026+ rows never opened by rankable loaders when manifest separation permits pre-open rejection.
- Equivalent funding/financing guards.
- Capital bid and ask fields remain separate; long/short execution selects the adverse executable side.
- Missing epic/type/calendar/expiry/metadata hash fails closed.
- Closed-market source events map to first executable target quote after reopening.
- Directed `source -> target` identity differs from reverse direction.
- Platform namespace prevents Kraken/Capital collisions while existing Kraken IDs and fixture outputs remain unchanged.
- No credentials or account/order endpoints imported.

## Existing regression baseline

- `unit_tests.test_rankable_loader_boundary`
- `unit_tests.test_kraken_readiness_repair`
- `unit_tests.test_qlmg_signal_state_contract`
- `unit_tests.test_qlmg_mechanical_qa_evidence_contract`

## Stop conditions

Stop before implementation/data import if:

- applied authority still says Kraken-only;
- the Capital.com manifest/schema package is absent or conflicting;
- files mix 2026+ and rankable purposes;
- historical financing, corporate-action or lifecycle fields are silently inferred;
- a patch requires rewriting existing Kraken identities/results;
- tests would need protected payloads or economic outcomes.

## Rollback

Keep changes additive and in an isolated task branch. Remove the new source-contract/adapter modules and their tests, revert only explicit compatibility imports, and verify all Kraken tests and hashes against the pre-change commit. Never rewrite or delete historical run roots.
