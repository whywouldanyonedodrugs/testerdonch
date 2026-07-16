# Code Review

Review consequential changes separately from implementation when possible. Review evidence, not the author's confidence.

## Establish scope

1. Read the task plan and applicable instruction chain.
2. Confirm the exact diff and identify unrelated pre-existing changes.
3. Confirm whether the task was documentation, non-economic code, rankable research, data work, or package work.
4. Confirm the authority files and commit used by the change.

Do not use review as authorization to run an economic screen, inspect protected outcomes, alter remote state, or repair unrelated problems.

## Review order

1. Safety and authorization boundaries.
2. Protected-period and point-in-time separation.
3. Candidate and control identity, freezes, and deterministic replay.
4. Fill, mark, index, funding, lifecycle, boundary, and non-overlap semantics.
5. Provenance, manifests, atomic output behavior, and supersession.
6. Tests, fixtures, failure paths, and documented commands.
7. Scope discipline, readability, and repository conventions.

## Required checks by risk

### Any code change

- The change is limited to approved files.
- Error handling fails closed where evidence could be misclassified.
- No credential, token, private endpoint, or sensitive path is exposed.
- A defect fix has a reproducing test when feasible.
- The exact supported commands were run, or the reason they could not run is recorded.

### Candidate or event generation

- `feature_available_ts <= decision_ts` is enforceable.
- No pre-listing signal and no current-roster historical backfill.
- Parent-neutral raw signals are frozen before parent-policy projections and outcomes.
- Maximum-hold preblocking does not suppress later valid signals.
- Eligible, accepted, skipped, and excluded rows reconcile.

### Controls and outcomes

- Controls are real, mechanism-relevant, and uniquely addressed.
- Candidate and control identities are hashed and frozen before outcomes.
- Summary, pooled, and projected rows cannot enter trade-level calculations.
- Last/trade, mark, index, and funding roles remain distinct.
- Imputed funding cannot activate a signal.
- Boundary-crossing positions are dropped or censored under a declared rule, not artificially closed.

### Rankable and package outputs

- No pre-2023, Bybit, or protected-period row reaches active output.
- Event sampling and caps are absent.
- Config, code, data, universe, funding, and output provenance is recorded.
- Source roots remain immutable and supersession is explicit.
- Package status, gaps, test evidence, hashes, and secret scan are truthful.

## Finding format

For each actionable finding, state:

```text
severity: blocking | high | medium | low
path and line:
observed behavior:
required contract:
consequence:
smallest repair:
verification:
```

Do not inflate style preferences into defects. If no actionable issue is found, state the remaining evidence limits and unrun checks.

## Completion gate

A review passes only when blocking findings are repaired, relevant repository-supported tests pass, and the diff is rechecked. A package can be structurally valid and still remain blocked by a protocol or evidence gap.
