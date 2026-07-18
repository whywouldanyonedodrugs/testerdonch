# Execution Plan

Task: `donch_multiplatform_source_authority_synthetic_adapter_20260718_r1`

## Frozen Scope

1. Verify the supplied ZIP and every `INPUT_MANIFEST.json` record.
2. Add a source-neutral rankable manifest guard with an explicit legacy Kraken compatibility path.
3. Add a synthetic-only Capital.com bid/ask adapter contract and directed cross-platform identity helper.
4. Add focused synthetic tests and preserve existing Kraken tests and identities.
5. Update only the bounded governance and contract documentation named by the task.
6. Run the full unit-test suite, review the actual diff, create one local commit, and publish the closed review package to the approved Drive handoff root.

## Safety Boundaries

- No real Capital.com payload access or import.
- No economic screen, strategy scoring, protected-outcome access, or capture access.
- No changes to existing Kraken identities, results, strategy semantics, or funding behavior.
- No push, merge, deploy, order endpoint, or account action.
- Generic source manifests fail closed; existing Kraken manifests use only the explicit compatibility route.

## Verification

- Focused source-contract and Capital.com adapter tests.
- Existing rankable-loader and contract regression tests.
- Complete `unit_tests` discovery suite.
- `py_compile`, `git diff --check`, prohibited-import scan, and explicit diff review.
- Review-package manifest and ZIP hash verification followed by non-overwriting Drive round-trip verification.
