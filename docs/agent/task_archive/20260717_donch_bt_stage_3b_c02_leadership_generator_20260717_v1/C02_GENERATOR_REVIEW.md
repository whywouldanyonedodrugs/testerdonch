# C02 Generator Review

Decision: `approve_mechanics_alignment_fragile`.

The generator uses only manifest-authorized Kraken PF trade/mark shards, Stage 3A sparse official USD spot bars, the Stage 2C lagged daily Top-100 panel, and known terminal lifecycle invalidations. It emits outcome-free identities and diagnostics. Reader boundaries, timestamp availability, sparse intersections, prior-day scaling, first-onset selection, completed trade-and-mark failures, and deterministic identities passed review.

The initial crossing implementation was rejected during review because it treated an already-active boundary row as a crossing. The final implementation requires a consecutive previous bar below the follower threshold. The final tape contains 567 primary ambiguous events and is covered by a regression test.

The alignment contract does not pass: 72.1073% of exact events retain the same episode and direction under both shifts, and 4.6884% of those retain the same leadership state. Required levels are 80% and 70%. No threshold, alignment, or label was changed after observing this result.

No economic output, candidate return, post-decision path, protected outcome, capture payload, or new market data was read or computed.
