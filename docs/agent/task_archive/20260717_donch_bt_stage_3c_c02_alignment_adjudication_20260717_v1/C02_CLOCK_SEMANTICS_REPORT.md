# C02 Clock Semantics Report

## Verified semantics

- Official Kraken spot trades are floored to UTC five-minute interval opens in `tools/run_kraken_c02_spot_reference_authority.py:243-244`.
- Normalized spot bars set `source_close_ts = timestamp + 5 minutes` and `feature_available_ts = source_close_ts` in `tools/run_kraken_c02_spot_reference_authority.py:321-327`.
- Authorized PF payloads are required to declare `resolution == 5m`; their millisecond `time` values are normalized to UTC interval-open timestamps by `read_pf_bars()` in `tools/build_kraken_c02_leadership_generator.py`.
- `align_exact()` performs one-to-one joins on identical spot, PF-trade, and PF-mark interval opens and assigns availability at interval open plus five minutes.
- Therefore the exact UTC grid is authoritative. Each crossing is observed only within its five-minute bar interval and is available at that interval's close.
- The `-5m` and `+5m` spot shifts are perturbation diagnostics. They are not alternative alignments and never create the resolution-aware event population.
- Sparse spot intervals remain absent. The loaders and exact inner join do not fill, interpolate, or forward-fill them; a non-consecutive predecessor cannot establish a crossing.

## Resolution implication

A same-bar or five-minute bar-open ordering does not prove a bounded lead after accounting for the crossing intervals. A bar-open separation of at least ten minutes produces non-overlapping five-minute intervals and guarantees at least a bounded five-minute lead. This is a data-resolution rule, not a performance threshold.
