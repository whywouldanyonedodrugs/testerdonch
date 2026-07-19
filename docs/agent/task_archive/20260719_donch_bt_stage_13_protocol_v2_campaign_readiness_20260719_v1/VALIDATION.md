# Validation

## Passed

- Exact policy hashes: protocol v2 `4effe3e9608377876486b1583854a7e1479c8e93d6de1661f13d35842bbc6a73`; family policy `d07b0d290e59d17f7d6a587e2a31e6e550468f9ad403250cd6183c549e685335`; campaign protocol `7091156df9bf815a001423b63d673d2d4217f2616fd40e76a261f218a2d614df`.
- Route policy remains `c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa`; evidence tags remain `06d6705cf6c81f60a82fc3f7936974bd187025d96a28ca931efe24281573ef97`.
- Original columns and values for every pre-existing family and multiplicity registry row match the starting commit.
- Manifest rejects missing fields, protected folds, self-authorization, unknown identities, and invalid exposure classes.
- DAG represents hypothesis, fold, and phase dependencies; outer/later fold influence on an earlier freeze fails closed.
- Unregistered or duplicate explored cells and excess beam entries fail closed.
- Phase 2 is denied by the readiness manifest; family and global stop scopes remain isolated.
- Phases 2–7 require a separate human approval artifact bound to the actual request packet, campaign manifest, authority/cost hashes, hypotheses, and phases; state and hypothesis identity are mandatory.
- State commits accept exactly one dependency-valid DAG transition or one registered stop transition, preventing direct forged progress.
- State initialization/resume is manifest-hash idempotent; generation-checked writes use an exclusive lock and atomic replace.
- Resource excess and artifact hash mismatch fail closed; heartbeat is explicit.
- Builder imports no pandas, pyarrow, polars, numpy, or outcome loader; two independent builds were byte-identical.
- All folds end no later than `2026-01-01T00:00:00Z`; all are labelled programme-exposed and non-independent.
- Explored-cell and beam registries are empty; future packet is non-authorizing and excludes C17.

## Environment limitation

The existing sealed-slice test could not import because `pandas` is absent. No dependency was installed. Equivalent Stage 13 boundary assertions run in the dependency-free test and validation scripts.
