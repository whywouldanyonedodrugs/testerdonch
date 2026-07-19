# KDA Shared Feature Contract

- Feature contract hash: `4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4`.
- Feature version: `kda_shared_causal_features_v1_20260719`.
- Completed five-minute grid with exact trade/mark/analytics timestamp intersection.
- Lagged and rolling features require contiguous five-minute horizons and fail closed across gaps.
- Daily robust normalization uses distributions from the prior 60 UTC calendar days only, with at least 30 valid days and 70% of expected days. Each intraday row is scored using its own current value; later same-day rows cannot alter it.
- Zero MAD, non-finite ratios, missing exact inputs, lifecycle-invalid intervals, and non-cohort alt rows fail closed.
- BTC/ETH parent returns are causal context fields only.
- Raw basis, OI tuple, and liquidation strings remain addressable.
- No outcome field is produced.
