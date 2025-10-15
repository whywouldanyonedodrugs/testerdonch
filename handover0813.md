

# Donch Project: Owner's Manual 📖


## What the project is

**Donch** is a long-only crypto strategy that ranks coins by weekly relative strength, waits for a daily Donchian breakout (upper channel) with volume confirmation, then seeks a pullback/stall pattern before entering. Trades are gated by market regime (ETH 4h MACD histogram $\\ge 0$), and exits use ATR-based stops/targets on a higher timeframe (e.g., 1h ATR), plus time stop and deterministic intrabar fill rules. On top, a **LightGBM** meta-model predicts the probability each entry will be profitable; we threshold those probabilities to maximize EV, then re-backtest only the filtered signals. (Donchian: breakout bands; MACD histogram: distance MACD–signal; PR-AUC & Brier evaluate ranking & calibration; PBO/CSCV & PSR/DSR check robustness.) (*Sources: Investopedia, ChartSchool, Scikit-learn, SSRN, davidhbailey.com*)

-----

## Data model & folders 📁

  * `parquet/` — Primary OHLCV bars (5-minute) per symbol; must include a `timestamp` column (UTC) or `DatetimeIndex` (UTC).
  * `parquet_1m/` — Optional 1-minute bars for intrabar SL/TP sequencing.
  * `signals/` — Generated entry candidates (`signals_*.parquet`).
  * `results/` — Backtests (`trades.csv|parquet`, `equity.csv|parquet`, aggregates) and meta-model outputs (`results/meta/*`).

**Timestamps**: Everything written by the code is UTC. Start/end date filters in `config.py` are plain strings:

```python
START_DATE = "2025-01-01"   # or None
END_DATE   = "2025-08-10"   # or None
```

-----

## Strategy logic (operational spec) ⚙️

### Universe & strength filter

Compute weekly returns (from 5m data resampled to 1w), derive a relative-strength percentile, keep top X% and drop illiquid symbols by median 24h turnover threshold.

### Regime gate

Build ETH 4h bars from resampled 5m. Compute MACD line (fast EMA − slow EMA), signal (EMA of MACD), and histogram (MACD − signal). Trade only when hist \> 0 (or both MACD\>signal and hist\>0 if configured). (*Source: ChartSchool*)

### Entry signal (daily timeframe)

  * **Donchian breakout**: Price takes out the N-day highest high (e.g., N=20/55). (*Source: Investopedia*)
  * **Volume confirmation**: Either multiple of rolling median/mean (e.g., $\\ge 3\\times$) with a bars cap (so we don’t ask for 8k-bar medians), or top quantile (e.g., $\\ge 95$th pct) over a bounded window.
  * **Pullback/stall**: Rule variants like “retest of the breakout” or “close\_above\_break” with small consolidation.

### Position sizing & exits

  * **ATR** computed on a higher timeframe (e.g., 1h) from 5m data via resample, then forward-filled to decision bars in 5m. SL/TP at entry $\\pm$ k $\\times$ ATR. (ATR timeframe/length are configurable.)
  * **Time stop**: Exit after `TIME_EXIT_HOURS` if neither SL nor TP hit.
  * **Intrabar sequencing**: If both SL & TP lie inside a bar, tie-break using deterministic rules; use 1m when available to resolve the first touch.

### Throughput guards

Per-symbol cooldown, max trades per variant, min equity floor guard, min ATR as % of price to avoid micro-stops. (Optional max trades/day and notional caps if enabled in `config.py`.)

-----

## The backtesting & meta-modeling loop 🔄

### 1\. Scouting (`scout.py`)

Resamples, computes Donchian & volume filters, applies regime gate + RS/liq constraints, with vectorized helpers in `indicators.py`. Writes `signals/signals_*.parquet` containing:
`timestamp (UTC)`, `symbol`, `entry`, `atr (1h)`, `don_break_len`, `don_break_level`, `vol_flag/vol_mult`, `pullback_type`, `entry_rule`, `rs_pct`, `regime_up`, …

### 2\. Backtest (`backtester.py`)

Loads signals, aligns each to the next tradable 5m bar, sizes positions from risk % and ATR stop distance, simulates exit path using 1m (if present) or deterministic 5m rules, applies fees, and writes:

  * `results/trades.csv|parquet` (per-trade P\&L, R, exit reason)
  * `results/equity.csv|parquet` (equity, timestamp)

### 3\. Meta-model (`meta_model.py` & `auto_meta_pipeline.py`)

Train **LightGBM** classifier (sklearn API) on entry-time features. Use Combinatorially Purged / Embargoed Cross-Validation to avoid look-ahead/leakage; evaluate **PR-AUC** (ranking quality) and **Brier score** (calibration). Save OOS probabilities per entry, plus permutation and **SHAP** importances (OOS). (*Sources: LightGBM Documentation, shap.readthedocs.io, SSRN*)

Build an EV curve by thresholding probabilities (e.g., $t \\in [0.50 \\dots 0.95]$) and averaging realized R conditional on $p \\ge t$. Baseline AP equals the positive rate, so PR-AUC above prevalence indicates lift; Brier is mean-squared error of probabilities vs outcomes (lower is better). (*Source: Scikit-learn*)

Filter `signals.parquet` at the selected $p^\*$ and re-backtest those only.

### 4\. Robustness (`reporting.py`)

  * **CSCV / CPCV** $\\rightarrow$ estimate Probability of Backtest Overfitting (PBO).
  * **PSR / DSR** $\\rightarrow$ Sharpe significance under fat tails & multiple testing. (Bailey & López de Prado.) (*Sources: SSRN, davidhbailey.com*)

-----

## How to run (end-to-end) 🚀

#### 0\) One-time installs

```bash
pip install pandas numpy pyarrow tqdm lightgbm shap scikit-learn
```

#### 1\) Parameter sweeps (guarded)

Generates daily-TF signals with volume filters and runs a grid of SL/TP/time-exit & entry/pullback variants, with guardrails (cooldown, equity floor, max-trades). Produces `results/leaderboard_guarded.csv`.

```bash
python sweep_donch_params_guarded.py --start 2025-01-01 --end 2025-08-10
```

#### 2\) Pick the baseline tag

Open `results/leaderboard_guarded.csv`, choose a top tag (e.g., `DON_DAYS-55_VOL-quantile0.95_ENTRY-rebreak_high_PB-retest_SL-1.5_TP-3.0_TE-8.0`).

#### 3\) Train meta + filter + re-backtest

`auto_meta_pipeline.py` re-builds `results/trades.csv` from the chosen signals file (to ensure alignment), trains the meta-model, prints fold PR-AUC/Brier, computes EV by threshold, dedups nearby signals, filters at chosen $p^\*$, and re-runs the backtest over the filtered set.

Outputs (under `results/meta/`):
`oos_predictions.parquet`, `metrics.csv`, `perm_importance_oos.csv`, `shap_importance_oos.csv`, `ev_curve.csv`, `ev_by_threshold.json`.

-----

## File-by-file (what each does) 🗂️

*If I’m unsure, I flag it explicitly so your reviewer can double-check.*

### Core pipeline

  * `config.py` — All knobs: start/end dates (strings), directories, Donch lookbacks in days, volume filter mode/params, ATR length/timeframe (e.g., 14 on 1h), regime MACD params (ETH 4h), fees, risk %, cooldowns, guards.
  * `scout.py` — Generates signals (RS filter → regime gate → daily Donch breakout + volume confirmation + pullback rule). Parallel per symbol, bounded volume windows to avoid huge moving medians.
  * `indicators.py` — Vectorized helpers: resampling, Donchian channels, ATR with selectable timeframe, MACD (for regime), volume spike metrics (multiple & quantile modes) with window caps.
  * `bt_gates.py` — On/off entry gates; now mostly delegated to `scout.py`. (Small utility file.)
  * `bt_intrabar.py` — Deterministic 1m sequencing (`resolve_first_touch_1m`) for SL/TP ties and time-exit edge cases.
  * `backtester.py` — Simulation engine. Caches 5m (and optional 1m), computes SL/TP from ATR(1h) at entry, fees, P\&L, guards (max trades, equity floor, cooldown), and writes results.
  * `manager.py` — Orchestrates a single run (signals → backtest → aggregates). Useful for baseline checks.
  * `reporting.py` — Roll-ups and robustness: PBO/CSCV, PSR/DSR, plus significance utilities.
  * `meta_model.py` — The modeling core: LightGBM classifier, purged/embargoed CPCV, metrics, OOS predictions, permutation & SHAP importances. (*Sources: LightGBM Documentation, shap.readthedocs.io*)

### Sweep & automation

  * `sweep_donch_params_guarded.py` — Daily-TF Donch sweep driver with guardrails; writes per-variant results and `leaderboard_guarded.csv`.
  * `sweep_donch_params.py` — Earlier (pre-guard) sweeper; keep for reference or delete if redundant.
  * `auto_meta_pipeline.py` — “One-button” meta flow on the current baseline signals: train → EV curve → choose p\* → filter → re-backtest. (Handles UTC merges and de-dups.)

### Diagnostics & utilities

  * `diag_breakout_stages.py` — Fast smoke-test on a small subset: shows symbols that pass RS/liquidity and produce $\\ge 1$ Donch+volume event with current config (helps catch “no signals” issues).
  * `diag_subset_lift.py` — Compares realized performance for a probability subset (e.g., $p \\ge 0.8$) vs ALL to verify lift.
  * `check_header.py` — Validates required columns & dtypes for CSV/Parquet.
  * `shared_utils.py` — I/O helpers (strict timestamp handling, drop last partial bar, bounded rolling windows, parallel RS, etc.).

### Data pulls / probes (verify contents locally)

  * `pull5.py`, `pullall.py`, `pull_trading.py` — **Data ingestion scripts** for 5m bars / bulk pulls (left from earlier pipeline). 
  * `etl.py`, `normalize_existing_csvs.py` — Clean-up / normalization utilities for downloaded CSVs to the Parquet schema.
  * `offline_eth_probe.py`, `offline_multi_asset_regime_probe.py` — Offline analyses of regime signals (ETH or multi-asset).
  * `bt_bt_analytics.py`, `analysis_runner.py`, `run_reporting.py` — Analysis macros; **I haven’t reviewed their current CLI args**—will need to check these at some point.
  * `meta_on_top_variants.py`, `run_meta_on_topk.py`, `run_meta_pipeline.py`, `run_meta_pipeline_no_limits.py`, `run_meta_pipeline_v2.py` — Legacy / alternative meta runners. You now use `auto_meta_pipeline.py`; keep these as references.
  * `regime_detector.py` — (Minimal) regime logic (ETH MACD hist $\\ge 0$); main implementation now sits in `backtester.py`’s `RegimeGate` for caching.
  * `master_symbol_list.txt`, `symbols.txt`, `symbols - full.txt`, `perplist.txt` — Universe list.

-----

## Testing principles baked in ✅

  * **Timeframe clarity**: Donch lookbacks in days, ATR on 1h (configurable), regime on ETH 4h MACD histogram. (Avoids hidden “5m-based 100-bar Donch” mistakes.)
  * **UTC discipline**: All timestamps UTC; merges coerce both sides to tz-aware UTC to prevent silent mismerges.
  * **No look-ahead**: Meta splits are purged & embargoed; signals are generated strictly from past bars; intrabar sequencing is deterministic.
  * **Guardrails**: Min ATR as % of price, equity floor, max trades per variant, cooldowns, optional day caps—not about “cheating,” but to avoid pathological churn.
  * **Robustness checks**: PBO/CSCV to estimate overfitting risk; PSR/DSR for Sharpe significance under fat tails & multiple testing. (*Sources: SSRN, davidhbailey.com*)

-----

## What the meta-model numbers mean (quick reads) 📊

  * **PR-AUC vs prevalence**: Prevalence (positive rate) is the baseline AP of a random model; PR-AUC meaningfully above it indicates ranking lift. Use the EV curve to pick $p^\*$ that best monetizes that lift. (*Source: Scikit-learn*)
  * **Brier score**: Mean squared error of predicted probabilities; lower is better. Consider Platt or isotonic calibration later if you want more accurate p’s (helps EV targeting). (*Source: Scikit-learn*)
  * **Permutation & SHAP (OOS)**: Which features matter out-of-sample; TreeSHAP is exact for tree ensembles. (*Source: shap.readthedocs.io*)

-----

## Next steps (shortlist) 🎯

  * Calibrate probabilities (isotonic/Platt) and re-run EV selection. (Improves $p^\*$ stability.) (*Source: Scikit-learn*)
  * Probability-weighted sizing (e.g., scale risk by a monotonic function of $p$, capped).
  * Per-variant meta: auto-train per top-K leaderboard tags (your `auto_meta_pipeline.py` is ready to loop).
  * Feature audit: confirm features are strictly entry-time available; expand with regime slope, cross-TF volatility, and volume-structure features.
  * Walk-forward OOS windows beyond CPCV; keep an untouched final OOS.
  * Data QA: keep rejecting files where we can’t infer timestamps; maintain symbol blacklist for chronically broken feeds.
  * Cost realism: stress slippage/fee scenarios (tiers, maker/taker) and verify intrabar sequencing with and without 1m.
  * Housekeeping: remove unused legacy runners after you’re fully on `auto_meta_pipeline.py`.

-----

## References (concepts) 📚

  * **Donchian channels**: Breakout bands built from N-period high/low; common N≈20 for daily. (*Source: Investopedia*)
  * **MACD histogram (regime proxy)**: Measures MACD–signal; above zero suggests bullish momentum. (*Source: ChartSchool*)
  * **PR-AUC**: Baseline = prevalence (random classifier); precision-recall guidance. (*Source: Scikit-learn*)
  * **Brier score**: Probability calibration metric. (*Source: Scikit-learn*)
  * **LightGBM (sklearn API) & SHAP TreeExplainer**. (*Sources: LightGBM Documentation, shap.readthedocs.io*)
  * **PBO/CSCV and PSR/DSR**: Anti-overfitting & Sharpe significance. (*Sources: SSRN, davidhbailey.com*)