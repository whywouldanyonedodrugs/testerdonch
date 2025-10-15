# Donchian Breakout + Meta-Model — Tester’s Technical Manual

This is your end-to-end manual for running, maintaining, extending, and upgrading the research stack and converting it into a reliable live-trading workflow. It’s opinionated, exact, and mirrors the codebase you’ve been using (`manager.py`, `scout.py`, `backtester.py`, `sweep_donch_params_guarded.py`, `meta_model.py`, `compute_ev_and_select_threshold.py`, `apply_meta_filter_and_rebacktest.py`, `reporting.py`, `config.py`).

---

## 0) Project Layout & Conventions

```
C:\testerdonch\
  config.py                      # All knobs: dates, regime, exits, risk, etc.
  scout.py                       # Signal generation (entry conditions, filters, features)
  backtester.py                  # Simulator (exits, partials, trail, fees, risk sizing)
  manager.py                     # Orchestration: scout → backtest → aggregate
  sweep_donch_params_guarded.py  # Grid sweep of exits; caches signals per entry config
  reporting.py                   # Robustness (CPCV/PBO/PSR/DSR), equity plots
  meta_model.py                  # Meta-label training (CPCV), export artifacts
  compute_ev_and_select_threshold.py  # Join OOS preds to trades, pick EV-optimal p*
  apply_meta_filter_and_rebacktest.py # Gate signals by meta_p and re-backtest

  signals\                       # Cached signals (parquet)
  results\                       # Trades, equity, variants, meta artifacts
  results\variants\...           # Per-sweep variant outputs
```

### Data assumptions

* All timestamps **UTC** (timezone-aware in trades, normalized in signals).
* 5-minute OHLCV history for backtest; 1h/4h/1d regime series for filters.
* Parquet and CSV are the interchange formats.

---

## 1) Installing & Upgrading the Environment

### Base setup (Windows PowerShell)

```powershell
cd C:\testerdonch
python --version            # (we’ve used 3.13)
# Optional venv:
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -U lightgbm scikit-learn pandas pyarrow joblib shap numpy matplotlib
```

### Version pinning (recommended)

Create `requirements.txt`:

```
lightgbm==4.3.*
scikit-learn==1.5.*
pandas==2.2.*
pyarrow==16.*
joblib==1.4.*
shap==0.45.*
numpy==2.0.*
```

Then:

```powershell
pip install -U -r requirements.txt
```

> **Note:** The code handles scikit-learn’s `OneHotEncoder` API differences (`sparse_output` vs `sparse`) via a try/except.

---

## 2) Strategy Definition (What We Trade)

### Hypothesis (entry model)

* **Donchian breakout** on *daily* lookback (`DON_N_DAYS=20`), **close-above-break** confirmation.
* Signal eligible only when:

  * **Regime filter** says UP (MACD-style or Markov regime, 4h/1d).
  * **Volume spike** present (mode **multiple** with `VOL_MULTIPLE=2.0` using 30-day baseline).
  * **Relative strength** percentile (`rs_pct`) ≥ threshold (usually 70–85).
  * **ATR sanity**: min ATR fraction of price (avoid dead books).
  * Per-symbol cooldown + per-day trade cap guardrails (throughput controls).

### Exit model (best zone found in sweep)

* **Stop**: `SL_ATR_MULT` in \*\*\[2.0, 2.5]\`.
* **Take profit**: `TP_ATR_MULT` in \*\*\[8, 12]\`.
* **Timed** exit: **72h** or **None** (both worked; 72h safer).
* **No trailing**; partials **off** by default (partials didn’t add much backtest EV).

### Live risk sizing

* Cash- or %-risk mode (`RISK_MODE="percent"`, `RISK_PCT=0.01` e.g. 1% risk).
* Notional/leverage caps enforced (`NOTIONAL_CAP_PCT_OF_EQUITY`, `MAX_LEVERAGE`).
* **Meta gate**: require `meta_p ≥ p*` (e.g. 0.60) before sizing.
* (Optional) Prob-scaled sizing via `risk_scale` or `_prob_size_multiplier(meta_p)`.

---

## 3) Daily Operation

### A) Refresh signals & backtest the current config

```powershell
python manager.py
# outputs:
#   results\trades.csv
#   results\equity.csv
#   signals\signals.parquet
#   results\trades_aggregated.csv
```

### B) Fast health check with live meta gate

Use yesterday’s model to gate and re-backtest:

```powershell
python apply_meta_filter_and_rebacktest.py ^
  --signals signals\signals.parquet ^
  --pred results\meta_export\oos_predictions.parquet ^
  --proba-col y_proba ^
  --pred-ts entry_ts --pred-sym symbol ^
  --dedup mean --round 5min --tol 10min ^
  --pstar 0.60 ^
  --rebt ^
  --out signals\signals_p060.parquet
```

Inspect console: trades, final equity, PF, DD vs baseline.

---

## 4) Weekly Retrain + Export for Live

### A) Train meta-model & export artifacts

Pick your label (we typically used **mfe\_k=2.0**):

```powershell
python meta_model.py ^
  --trades-csv results\trades.csv ^
  --signals-parquet signals\signals.parquet ^
  --target mfe_k --mfe-k 2.0 ^
  --outdir results\meta_export ^
  --pstar 0.60
```

Artifacts for deployment:

```
results\meta_export\
  donch_meta_lgbm.joblib     # model weights
  ohe.joblib                 # fitted OneHotEncoder for cats
  feature_names.json         # feature order for inference
  config_snapshot.json       # feature-building config parity
  pstar.txt                  # chosen threshold (optional)
  oos_predictions.parquet    # eval
  metrics.csv                # PR-AUC / Brier per fold
```

### B) Validate EV & confirm p\*

```powershell
python compute_ev_and_select_threshold.py ^
  --pred results\meta_export\oos_predictions.parquet ^
  --trades results\trades.csv ^
  --pred-ts entry_ts --trades-ts entry_ts ^
  --pred-sym symbol  --trades-sym symbol ^
  --proba-col y_proba ^
  --dedup mean ^
  --min-trades 150 ^
  --out results\meta_export\ev_curve.csv
```

* If EV curve suggests different p\*, re-run `meta_model.py` with `--pstar <value>` to record it in `pstar.txt`.

### C) (Optional) Re-backtest gated signals as a sanity check

```powershell
python apply_meta_filter_and_rebacktest.py ^
  --signals signals\signals.parquet ^
  --pred results\meta_export\oos_predictions.parquet ^
  --proba-col y_proba --pred-ts entry_ts --pred-sym symbol ^
  --dedup mean --round 5min --tol 10min ^
  --pstar 0.60 --rebt ^
  --out signals\signals_p060.parquet
```

---

## 5) Robustness & Parameter Sweeps

### A) Quick exit sweep (guarded)

```powershell
python sweep_donch_params_guarded.py --start 2025-01-01 --end 2025-08-10 --robust-topk 5
# Outputs: results\leaderboard_guarded.csv (+ per-variant subfolders)
```

Interpretation:

* Focus on `profit_factor`, `max_dd`, `cagr`, `mar` (CAGR/DD), Sharpe.
* Our best zone: **SL=2–2.5**, **TP=8–12**, **TE=72h/None**, **no trail**, **partials off**.

### B) Full robustness reporting

```powershell
python reporting.py --run-all --returns-col pnl_R ^
  --variant-cols pullback_type entry_rule don_break_len regime_up
```

---

## 6) Creating a New Strategy (Template)

You can treat **`scout.py`** as the “strategy module”. Typical workflow:

1. **Add config knobs** to `config.py`

   * Entry basis (e.g., channel length, thresholds)
   * Filters (RS, volume spike, regime)
   * Exit policy (SL/TP/time, partials, trail)
   * Risk, cooldowns, caps

2. **Implement signal logic** in `scout.py`

   * Produce one row per eligible entry with the minimum required columns:

     ```
     timestamp, symbol, entry, atr, don_break_len, don_break_level,
     pullback_type, entry_rule, rs_pct
     ```
   * Add extra features if useful (e.g. `atr_1h`, `rsi_1h`, `vol_mult`, `eth_macd_hist_4h`, …)
   * Ensure **UTC** timestamps and per-symbol dedup.

3. **Run a smoke test**

   ```powershell
   python manager.py
   # confirm signals\signals.parquet and results\trades.csv exist
   ```

4. **Sweep exits** (optional) to find a workable zone.

5. **Label and train meta-model**

   * Use `meta_model.py` (`--target mfe_k` or `--target pnl_pos`).

6. **EV threshold** selection and **meta gate** sanity re-backtest.

7. **Export** artifacts and **deploy** to live.

> **Tip:** Signals are cached in the sweep by a **hash of entry config** (see `_hash_entry_cfg()` in the sweep script). Change entry knobs to invalidate cache.

---

## 7) Live Integration (Bot)

In your bot’s entry handler:

1. **Build features** for the incoming signal:

   * Load `feature_names.json`, `ohe.joblib`, `config_snapshot.json`.
   * Build the numeric feature array (`num_cols` you trained with).
   * One-hot the categorical columns (using `ohe.joblib`).
   * **Order exactly as** `feature_names.json`.

2. **Predict**:

   ```python
   import joblib, numpy as np, json

   model = joblib.load("donch_meta_lgbm.joblib")
   ohe   = joblib.load("ohe.joblib")
   cols  = json.load(open("feature_names.json"))

   # x_num: np.array shape (1, n_num)
   # x_cat_df: DataFrame with same categorical columns used in training
   Xo = ohe.transform(x_cat_df.astype(str))
   x = np.hstack([x_num, Xo])
   proba = float(model.predict_proba(x)[:,1])
   ```

3. **Calibrate** (if `calibrator.joblib` is present):

   ```python
   cal = joblib.load("calibrator.joblib")     # IsotonicRegression
   proba = float(cal.predict([proba])[0])
   ```

4. **Gate**: `if proba >= pstar:` (load `pstar.txt`, default fallback)

5. **Risk size**: use the backtester’s sizing: `qty = cash_risk / (entry - sl_initial)`, observe notional/leverage caps.

6. **Trade orchestration**:

   * SL/TP/time implementation matches `backtester.py` logic.
   * Disable partials/trailing unless you decide otherwise.

7. **Logging**

   * Persist `(ts, symbol, features, meta_p, accepted/rejected, order ids)`.

8. **Parities**

   * Stop if live feature vector length ≠ `len(feature_names.json)`.
   * Assert UTC time alignment to the bar.

---

## 8) Scheduling & Maintenance

### Suggested schedule

* **Nightly:**

  * `manager.py` (refresh window).
  * `apply_meta_filter_and_rebacktest.py` with current model.
  * Compare PF/DD vs baseline; alert on degradation.

* **Weekly:**

  * `meta_model.py` retrain + export.
  * `compute_ev_and_select_threshold.py` → update `pstar.txt`.
  * Sanity re-backtest gated signals.
  * Deploy artifacts; keep previous version for rollback.

* **Monthly:**

  * `sweep_donch_params_guarded.py` for robustness.
  * `reporting.py --run-all` (CPCV/PBO/PSR/DSR).
  * Archive `results\meta_export\YYYYMMDD\`; keep last 3–6.
  * Update/lock `requirements.txt`.

---

## 9) Guardrails, Risk & Throughput Controls

* `MAX_TRADES_PER_DAY`
* `SYMBOL_COOLDOWN_MINUTES`
* `MIN_ATR_PCT_OF_PRICE`
* `NOTIONAL_CAP_PCT_OF_EQUITY`, `MAX_LEVERAGE`
* Abort variant if equity < `MIN_EQUITY_FRACTION_BEFORE_ABORT` in backtests
* Regime block when down (`REGIME_BLOCK_WHEN_DOWN`) or size-down alternative

---

## 10) Troubleshooting (Greatest Hits)

* **Duplicate index error (asfreq):** `cannot reindex on an axis with duplicate labels`
  Fix by `edf = edf.drop_duplicates(subset="timestamp", keep="last")` before `.asfreq("D")`.

* **LightGBM missing:** `NameError: LGBMClassifier is not defined`
  Ensure `from lightgbm import LGBMClassifier` at top; install `lightgbm`.

* **Feature name mismatch / shape errors:**
  Always build matrices as DataFrames with synthetic col names (`f0..fN`) both train/test; reindex test to train columns (missing→0), which we do already.

* **`MergeError: keys not unique` in EV script:**
  Use `--dedup mean` to reduce OOS preds to unique `(ts,symbol)`.

* **`Predictions must have proba`:**
  Pass `--proba-col y_proba` when needed.

* **Joining preds↔trades fails:**
  Supply explicit keys: `--pred-ts entry_ts --trades-ts entry_ts --pred-sym symbol --trades-sym symbol`.
  If bars don’t align, use `--round 5min --tol 10min` and per-symbol asof in `apply_meta_filter_and_rebacktest.py`.

* **`KeyError: 'timestamp'` in apply script:**
  Your OOS preds have `entry_ts`. Pass `--pred-ts entry_ts`.

* **`continue` not in loop (sizing):**
  Gates must live in the main backtest loop, **not** inside `_size_from_risk`. We already moved them.

* **SHAP warnings:** Safe to ignore; we degrade gracefully.

---

## 11) Extending Features for the Meta-Model

* Add columns in `scout.py` (e.g., `rsi_1h`, `vol_mult`, `markov_prob_up_4h`, `don_dist_atr`).
* `meta_model.py` picks up any column listed in `CAND_NUM`/cats if present.
* Keep everything **entry-time only** (no leakage from the future path).

---

## 12) Code Hygiene & Reproducibility

* **Seeds:** We seed CPCV folds (`random_state=seed+fold`) and final refit (`1337`).
* **`run_args.json`:** Every `meta_model.py` run writes inputs and split config.
* **Hash-cached signals** per entry config in the sweep prevent re-scouting.
* **Version control:** Tag artifact drops with date; commit `config_snapshot.json`.

---

## 13) Upgrades & Migration

* Upgrade one library at a time; rebuild the environment with pinned versions.
* Re-train meta model after upgrading major libs (LightGBM/sklearn).
* Regenerate `ohe.joblib` whenever cats change (new categories or new cat columns).
* Keep at least one prior artifact set for quick rollback.

---

## 14) Minimum Viable Settings (Reference)

**Entry**

* `DONCH_BASIS="days"`, `DON_N_DAYS=20`
* `ENTRY_RULE="close_above_break"`, `DON_CONFIRM_CLOSE_ABOVE=True`
* `VOL_SPIKE_MODE="multiple"`, `VOL_MULTIPLE=2.0`, `VOL_LOOKBACK_DAYS=30`
* `RS_MIN_PERCENTILE≈70–85`
* Regime filter: UP only (`REGIME_BLOCK_WHEN_DOWN=True`)

**Exit**

* `SL_ATR_MULT=2.0` (or 2.5)
* `TP_ATR_MULT=8.0` (or 12)
* `TIME_EXIT_HOURS=72` (or None)
* Partials/trail **off** by default

**Meta**

* Target: `mfe_k=2.0` (or `pnl_pos>=0.0`)
* p\*: **0.60** (re-check via EV)

**Risk**

* `RISK_MODE="percent"`, `RISK_PCT=0.01`
* Caps: `NOTIONAL_CAP_PCT_OF_EQUITY`, `MAX_LEVERAGE`

---

## 15) FAQ (Short)

* **Can I run partials/trailing?** Yes, they’re implemented; but sweeps showed little EV benefit. Keep off unless a new sweep says otherwise.
* **How do I add a new filter?** Add it to `scout.py` (compute), to `signals.parquet`, then (optionally) into `CAND_NUM`/cats for meta. Re-train.
* **What’s the quickest smoke test?** Run a tiny date range in `manager.py`; or `sweep_donch_params_guarded.py --single`.

---

## 16) Quick Reference — Core Commands

```powershell
# Refresh signals & backtest current config
python manager.py

# Sweep exits (guarded)
python sweep_donch_params_guarded.py --start 2025-01-01 --end 2025-08-10 --robust-topk 5

# Train + export meta-model (MFE target)
python meta_model.py --trades-csv results\trades.csv --signals-parquet signals\signals.parquet --target mfe_k --mfe-k 2.0 --outdir results\meta_export --pstar 0.60

# EV threshold selection
python compute_ev_and_select_threshold.py --pred results\meta_export\oos_predictions.parquet --trades results\trades.csv --pred-ts entry_ts --trades-ts entry_ts --pred-sym symbol --trades-sym symbol --proba-col y_proba --dedup mean --min-trades 150 --out results\meta_export\ev_curve.csv

# Apply meta gate & re-backtest
python apply_meta_filter_and_rebacktest.py --signals signals\signals.parquet --pred results\meta_export\oos_predictions.parquet --proba-col y_proba --pred-ts entry_ts --pred-sym symbol --dedup mean --round 5min --tol 10min --pstar 0.60 --rebt --out signals\signals_p060.parquet

# Full robustness report
python reporting.py --run-all --returns-col pnl_R --variant-cols pullback_type entry_rule don_break_len regime_up
```

---

## 17) Final Notes

* **Don’t skip the EV check.** The meta model is about *ranking*; choose p\* where **expected R** is best with enough trades.
* **Feature parity is sacred.** Always verify live features match `feature_names.json`. If in doubt, block trading and alert.
* **Keep it boring.** Weekly retrain, monthly sweeps, tightly pinned deps, predictable rollbacks.

You’re good to go.
