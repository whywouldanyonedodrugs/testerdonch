# Kraken Candle Volume Validation

Authority hash: `c33f620828c3d0e21865c269a5cacf02002335ead29eddbd05d80d268fd27f1f`.

All 12 frozen five-minute intervals matched complete public execution quantity exactly. The public execution interval is half-open and every execution UID was unique. The observed PF contract specification rows express base currency and minimum lot in base units. Historical semantic rows are versioned by archived official source; events preceding a symbol's first observed official version or with inconsistent base/min-lot observations must fail closed.

`volume` is authorized as base quantity only for listed PF symbol/semantic intervals. Exact quote volume is unavailable. Current 2026 calibration was not joined to alpha, strategy outcomes, parameters, or ranks.
