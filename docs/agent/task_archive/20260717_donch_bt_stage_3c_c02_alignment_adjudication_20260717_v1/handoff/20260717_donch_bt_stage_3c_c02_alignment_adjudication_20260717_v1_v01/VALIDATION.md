# Validation

Status: **pass** for outcome-free resolution-aware contract review.

- Source Stage 3B contract/event/failure hashes: pass.
- Stage 3A spot manifest hash and protected-row gate: pass.
- Frozen Stage 3B events reconstructed: 32,686 of 32,686; original 15m/30m label mismatches: 0.
- Resolution-aware identities: duplicate event IDs 0; duplicate economic addresses 0.
- Deterministic full replay: all four generated Parquet/CSV hashes byte-identical.
- Primary states: 1,017 resolved spot-led; 609 resolved perp-led; 31,060 coincident/unresolved.
- Retained completed failures: 264, all attached only to resolved perp-led events.
- Protected rows/events: 0. Prohibited outcome fields: 0. Economic outputs: 0.
- Resolved spot-led aggregate feasibility: pass (`82.6942%` same leader at 30m).
- Resolved perp-led aggregate feasibility: fail (`67.6519%` same leader at 30m).
- Authoritative run: 231.19 seconds; peak RSS 1,080,340 KiB; exit 0.
- Deterministic replay: 231.82 seconds; peak RSS 1,086,196 KiB; exit 0.
- Compile and focused Stage 3A/3B, protected-loader, lifecycle, authority, and Stage 3C tests: 75 passed; 0 failed; 0 errors.
