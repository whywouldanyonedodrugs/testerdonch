# Independent Review

Decision: `approve` after repair and final validation.

## Scope reviewed

- New analytics semantic and identity module.
- New bounded Stage 8A generator.
- Synthetic regression module.
- Factual project/data-authority documentation amendments.
- Generated manifests, event schemas, source hashes, protected-boundary counts, and resource telemetry.

## Findings repaired before approval

1. **Blocking PIT normalization defect.** The first implementation mapped an entire current UTC day's aggregate onto each intraday row. Later same-day values could therefore affect an earlier event. It was replaced with row-level scoring against a distribution ending on the prior UTC day, with a same-day future-invariance regression.
2. **Blocking incomplete-horizon defect.** OI, basis, and liquidation lagged features could bridge a missing five-minute interval. Exact contiguous-horizon masks now fail those rows closed, with a missing-window regression.
3. **High cache-lineage/identity defect.** Event shards did not bind an explicit generator-contract hash, and the robustness OI-vacuum attempt duplicated the primary OI-vacuum mechanics. Event rows and shard manifests now bind the frozen generator hash; the duplicate branch remains in the attempt registry as killed and emits no tape.
4. **Medium completeness defect.** Initial old-family overlap output omitted C01 and C02. Safe identity-only mappings were added without projecting economic columns.

All affected pre-repair outputs and cache shards remain preserved and explicitly non-authoritative.

## Post-repair review result

No remaining blocking or high finding was identified. The final diff is task-scoped, uses bounded partitioned reads, keeps raw field semantics distinct, enforces train/protected boundaries, produces no outcome columns, and preserves exact source/feature/event lineage. No strategy logic, economic output, capture path, Capital.com payload, private endpoint, or historical result root is modified.

## Remaining evidence limits

- Analytics units remain `inferred_authoritative_v1`, not exchange-certified semantics.
- The cohort is current-roster capped with known lifecycle invalidations; it is not proven survivorship-free or continuously tradable.
- OI starts later than basis/liquidation history, truncating exact intersections in early 2023.
- Liquidation direction is only a price-inferred proxy; native long/short assignment is unavailable.
- One-minute diagnostics are BTC/ETH aggregation checks only and are not an alternate rankable event tape.
- Older-family exact overlap is unavailable where no safe causal identity tape exists.
- Stage 8A is outcome-free and cannot support an economic conclusion.
