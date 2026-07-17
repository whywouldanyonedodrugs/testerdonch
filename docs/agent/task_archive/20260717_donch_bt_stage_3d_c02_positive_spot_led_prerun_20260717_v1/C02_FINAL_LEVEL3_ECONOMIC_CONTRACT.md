# C02 Final Level-3 Economic Contract

Lineage: `C02_positive_resolution_aware_spot_led_continuation_v1`.

This document freezes a later, separately authorized test of positive resolved spot-led continuation only. Stage 3B one-bar alignment remains failed. Negative spot-led, perp-led, completed failure, shifted clocks, alternate thresholds, and alternate horizons are excluded.

Primary identities: `7dbdb3763b9131480f712f60c2e7a4d0822f65a276b4ed5c5c00bdb804e3c42c` (489 source events). Robustness identities: `f3284aaf54da7c2f53d6a3561eab8e92cc639c40c7b9c025ed1991ac63bf7ca1` (425 source events).

Definitions are exactly `c02_l3_primary_all_1h`, `c02_l3_primary_all_6h`, `c02_l3_30m_agreement_1h`, and `c02_l3_30m_agreement_6h`. Agreement definitions are robustness-only and cannot rescue primary failure.

Decision is the Stage 3C onset-bar availability time. Entry is the first executable Kraken PF five-minute trade-bar open strictly after decision. Exit is the first executable PF five-minute trade-bar open at or after entry plus one or six hours. Exposure is fixed notional; exits are timeout-only. Each definition applies symbol-local non-overlap using its actual timeout exit.

Base cost is 5 bps taker per side plus 4 bps round-trip slippage (14 bps total). Stress cost is 10 bps per side plus 12 bps round-trip slippage (32 bps total). Primary gates use base net bps excluding funding. Funding is partitioned separately as fully exact, mixed, fully imputed, or zero-boundary and cannot rescue a definition.

The frozen Level-3 gates and seed are machine-readable in `C02_LEVEL3_DECISION_RULES.json`. Passing permits later controls only; it is not validation or promotion. No economic run is authorized by this contract freeze.
