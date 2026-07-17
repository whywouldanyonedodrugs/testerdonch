# C01 Feature Contract

Family: `C01_debetaed_residual_shock_path_bifurcation`

Feature version: `c01_residual_path_features_v1_20260717`
Feature-contract hash: `c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb`

Five-minute source timestamps are candle opens. A source row becomes available at `source_open_ts + 5m`; that completed close is `decision_ts`. The daily OLS used on UTC day D is fitted from aligned valid observations in `[D-30d,D)`. Valid observations require candidate and required-factor trade returns plus distinct mark rows. At least `6048` of `8640` expected observations are required.

The 6h shock uses 72 consecutive completed residuals. Scale is the sample standard deviation of valid UTC-anchored, non-overlapping 6h residual sums whose block ends are on or before the current shock-window start and within its prior 30 calendar days. At least 80 blocks are required. This removes overlap between the current shock and scale inputs.

Mark closes are eligibility/quality inputs only and are never substituted for trade returns. Parent returns/RV are diagnostics, not gates. No funding, OI, basis, spread, session, catalyst, prior-high, relative-strength, forward return, exit, or PnL field is computed.

Candidate cohort is exactly `current_roster_bar_existence_cohort`. It is current-roster capped and is not survivorship-free. Official opening dates and complete event-time bars fail closed; continuous tradeability is not claimed. Known official terminal intervals are masked from the settlement date through the end of a later documented resumption date, or through the protected cutoff when no later resumption exists. Date-only lifecycle authority is conservatively interpreted as whole UTC dates.
