# Validation

Status: **pass mechanics; frozen alignment gate fails**.

- Frozen contract SHA-256: `25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb`.
- Stage 3A spot manifest content hash: `3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046`.
- Stage 2C cohort hash: `768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15`.
- Coverage: 204 mapped spot/PF symbols audited; 223,584 symbol-days; 27,871 eligible-before-scale symbol-days.
- Tape: 32,686 onsets across 87 symbols; 1,212 completed failures; 13,734 canonical episodes.
- Primary leadership: 24,805 simultaneous, 4,499 spot-led, 2,815 perp-led, 567 ambiguous.
- Direction: 16,691 positive and 15,995 negative.
- Years: 4,255 in 2023; 12,046 in 2024; 16,385 in 2025.
- Identities: event duplicates `0`; economic-address duplicates `0`; failure duplicates `0`; deterministic event/episode mismatches `0`.
- Temporal gates: decision-input leaks `0`; protected rows/events `0`; sparse fills `0`; failure-to-episode misses `0`.
- Schema gate: prohibited outcome fields `0`; economic outputs `0`; protected outcomes opened `0`.
- Alignment: same episode/direction under both shifts `0.721073`; same leadership state conditional on both `0.046884`; status `alignment_fragile_requires_review`.
- Runtime: 450.59 seconds; peak RSS 983,384 KiB.
- Compile: pass. Focused plus spot/PF loader, lifecycle, protected-boundary, and authority tests: 65 passed, 0 failed, 0 errors.
