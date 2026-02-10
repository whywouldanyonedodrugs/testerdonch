# Executive Status Summary (2026-02-09)

Audience: non-quant stakeholders with trading familiarity.

## 1) What happened

1. We discovered that backtest results were overstating performance.
2. Main cause: a leakage path and sizing behavior made historical results look safer and better than reality.
3. Symptom: many losing trades were tiny while winners stayed large, producing unrealistic profit curves.

## 2) What we fixed

1. Disabled replay-feature leakage in backtesting by default.
2. Switched daily regime logic to past-only probability estimation (no future-looking smoothing in offline path).
3. Added tooling to run a full offline release pipeline and produce a strict parity handoff package for live team.
4. Added formal live parity manual and go/no-go checks.

## 3) What current evidence says

1. The old in-sample “winner” was too optimistic.
2. On newer out-of-sample data, performance is still positive in some modes, but much smaller and more sensitive to sizing.
3. Conclusion: there may be edge, but it is not yet proven robust enough for immediate full-risk deployment.

## 4) Strategy in plain language

The strategy is a long-only breakout system:

1. It looks for assets breaking above a multi-day range.
2. It waits for a pullback/retest behavior before entering.
3. It uses additional quality checks (volume, trend context, etc).
4. Exits are controlled with stop-loss and take-profit rules based on volatility (ATR).

## 5) Risk management in plain language

1. Every trade starts from a base risk budget.
2. A model score can scale position size up or down.
3. In “risk-off” market states, size is forcibly capped to a small probe amount.
4. Portfolio-level controls limit overexposure and excessive clustering.

## 6) Why live and offline must match exactly

If live mechanics differ from backtest mechanics, we cannot trust expected outcomes.

Critical parity items:

1. same model bundle and checksum
2. same feature schema
3. same gate logic
4. same size mapping
5. same regime logic (past-only)

## 7) What we are doing next

1. Complete corrected offline release run:
- source backtest
- enriched clean trades
- retrain and export new model bundle
- generate parity fixture package
2. Run corrected mini-sweep (pre-registered set) to pick deployable policy.
3. Hand package to live team for strict parity checks.
4. Start with canary risk only.
5. Move to full size only if parity and canary checks pass.

## 8) Immediate operating stance

1. Do not use contaminated sweep ranking as final truth.
2. Do not force full-size live based on old winner.
3. If live currently takes no trades, adjust to a parity-safe temporary profile (no hidden vetoes, probe risk) while release process completes.

## 9) Bottom line

We are not abandoning the strategy.  
We are moving from “possibly inflated backtest confidence” to a controlled, auditable, parity-validated deployment path.

