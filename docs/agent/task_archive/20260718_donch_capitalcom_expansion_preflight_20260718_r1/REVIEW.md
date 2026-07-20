# Review

Result: `pass_with_external_data_authority_blocker`.

## Findings

1. No Capital.com acquired-data manifest or schema is available locally. Import and economics are blocked.
2. Current rankable loader guards are materially useful but embedded in a large Kraken runner and hard-code Kraken/PF identities. A small extraction is justified; broad runner refactoring is not.
3. Existing event/control IDs are not uniformly platform-namespaced. Rehashing Kraken history would break lineage. New outer platform-aware identities are safer.
4. Capital.com financing, corporate actions, lifecycle, calendar and historical trading rules cannot be inferred from current metadata or planning prose.
5. The proposed shared core plus two adapters is feasible if “shared” is limited to authority, identity, time, evidence and reproducibility contracts.

No strategy, threshold, family status or economic conclusion was changed.
