# JT-012: Regime-Sliced Attribution and Failure-Bucket Audit

## Scope
Implemented in:
- `tools/donch_autopar_compare.py`
- `tools/donch_autopar_daily.py` (no-signals fallback)

## What Was Added
1. Per-regime attribution from backtest trades:
- `risk_on` vs `risk_off`
- `eth_hist_sign` (ETH MACD histogram sign)
- `btc_vol_regime` (high/low via `BTC_VOL_HI`)

Metrics:
- `n_trades`
- `hit_rate`
- `pnl_R_mean`
- `pnl_R_sum`
- `max_drawdown_R`

2. Canonical failure buckets for live skipped decisions:
- `no_signals`
- `schema_fail`
- `scope_fail`
- `below_pstar`
- `gate_fail`
- `other`

3. Monthly health outputs combining regime and failure diagnostics.

## Output Artifacts
Produced under compare `outdir`:
- `summary.json`
- `failure_bucket_audit.csv`
- `failure_bucket_rows.csv`
- `regime_attribution.csv`
- `monthly_regime_health.csv`
- `monthly_regime_health_report.md`
- `report.html`

## Acceptance Mapping
1. Report separation of likely causes:
- Pipeline-heavy months are marked when `pipeline_failure_share` is elevated.
- Negative months with low pipeline share are tagged as `unfavorable_regime_or_edge_decay`.

2. Canonical assignment coverage:
- `summary.json -> checks.failure_bucket_assigned_rate_live`
- Target: `>= 0.95`

## Smoke Command
```bash
./.venv/bin/python tools/donch_autopar_compare.py \
  --live-decisions /tmp/donch_autopar_smoke/live_decisions.csv \
  --bt-decisions results/donch_autopar/smoke_orch/reference/backtest/signal_decisions.csv \
  --bt-trades results/trades.csv \
  --outdir results/donch_autopar/smoke_jt012_compare_with_trades \
  --min-overlap-rate 0.0 \
  --min-enter-agreement 0.0
```
