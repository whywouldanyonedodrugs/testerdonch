# Liquidity Proxy Contract

Field: `close_based_usd_volume_proxy`. Five-minute value is `trade_close_5m * verified_base_volume_5m`; daily value is its UTC-day sum. Membership is top 100 by the prior 30 calendar days' median, requiring at least 20 valid prior days and ranking once daily using data through the prior UTC day.

This is a causal cohort-hygiene proxy. It is not exact quote volume, not traded USD notional, not capacity, not spread/depth evidence, and not executable-liquidity or slippage evidence.
