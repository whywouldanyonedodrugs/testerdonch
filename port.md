# Donch → Live Trading Hand-Over (Technical Playbook)

---

## 1) Strategy overview — what it does & why

### Hypothesis

1. **Relative strength (weekly)** surfaces coins with persistent demand.
2. **Daily Donchian breakout + volume spike** identifies regime-consistent momentum bursts.
3. **Pullback/stall entry** reduces chasing and improves fill quality.
4. **Market regime gate (ETH 4h MACD hist ≥ 0)** avoids trading against major crypto momentum.
5. **Risk/exit on higher TF volatility (ATR-1h)** standardizes stops/targets across symbols.
6. **Meta-model classifier** separates higher-EV entries from noise and controls throughput.

### Core flow (research & live)

```
Universe → RS/liq filter → Regime gate → Daily Donch breakout + Volume spike
→ Pullback/stall qualifies entry → Size by risk using ATR(1h) → Bracket: SL & TP (with optional partial & trail)
→ Time stop (optional) → Intrabar sequencing (deterministic; 1m if available)
→ Per-trade features logged → Meta-model (trained OOS) scores new entries → Threshold by p* → Execute
```

---

## 2) Data contracts & directories

```
parquet/          # 5-minute OHLCV per symbol (UTC); required columns: timestamp, open, high, low, close, volume
parquet_1m/       # optional 1-minute OHLCV per symbol (UTC) for intrabar resolution
signals/          # generated candidate entries (parquet)
results/          # backtests, equity, aggregates; meta outputs live under results/meta/*
```

**Timestamps**: tz-aware UTC. All merges/joins must coerce to UTC.

---

## 3) Configuration knobs (single source of truth)

In `config.py`, these govern both research and live.

### Universe, execution window, performance

* `START_DATE`, `END_DATE` — ISO strings or `None` (UTC window).
* `N_WORKERS`, `SCOUT_BACKEND` — parallelism controls.
* `IO_MEMORY_MAP` — speed for parquet.

### Relative Strength (weekly) & Liquidity

* `RS_ENABLED=True`
* `RS_MIN_PERCENTILE=70`  *(top 30%)*
* `RS_LIQ_MIN_USD_24H=500_000.0` *(median 24h turnover filter)*

### Donchian & Volume (entry)

* `DONCH_BASIS="days"`; `DON_N_DAYS=20` *(canonical)*
* `DON_CONFIRM_CLOSE_ABOVE=True` *(close > upper band on signal bar)*
* Volume spike gate:

  * `VOL_SPIKE_ENABLED=True`
  * `VOL_SPIKE_MODE="multiple"|"quantile"`
  * `VOL_LOOKBACK_DAYS=30`
  * `VOL_MULTIPLE=2.0` or `VOL_QUANTILE_Q=0.95`

### Pullback + Entry rule

* `PULLBACK_MODEL="retest"|"mean"`
* `ENTRY_RULE="close_above_break"|"rebreak_high"`
* Windows: `PULLBACK_WINDOW_BARS/HOURS`, `RETEST_EPS_PCT`, `MEAN_MA_LEN`, etc.

### Indicators (timeframes)

* `ATR_TIMEFRAME="1h"`, `ATR_LEN=14`
* **Regime filter (ETH 4h)**:

  * `REGIME_FILTER_ENABLED=True`
  * `REGIME_ASSET="ETHUSDT"`, `REGIME_TIMEFRAME="4h"`
  * MACD params: 12/26/9; `REGIME_REQUIRE_BOTH_POSITIVE=True` (MACD>signal **and** hist>0)
  * If not blocking: `REGIME_BLOCK_WHEN_DOWN=False`, `REGIME_SIZE_WHEN_DOWN=0.5`

### Risk, sizing & exits

* **Capital & fees**: `INITIAL_CAPITAL`, `FEE_RATE`
* **Sizing mode** (both live & backtest share):

  * Percent risk: `RISK_MODE="percent"`, `RISK_PCT=0.01` *(1% of equity)*
  * Fixed cash risk (optional): `RISK_MODE="cash"`, `FIXED_RISK_CASH=10.0`
* **Bracket**: `SL_ATR_MULT`, `TP_ATR_MULT`
* **Time exit**: `TIME_EXIT_HOURS` or `None`
* **Notional & leverage caps**:
  `NOTIONAL_CAP_PCT_OF_EQUITY`, `MAX_LEVERAGE`
* **Min ATR %**: `MIN_ATR_PCT_OF_PRICE=0.0001` to avoid micro-stops.

### Partials & trailing (optional)

* `PARTIAL_TP_ENABLED` *(0/1)*
* `PARTIAL_TP_RATIO=0.5`
* `PARTIAL_TP1_ATR_MULT=5.0`
* `MOVE_SL_TO_BE_ON_TP1=True`
* `TRAIL_AFTER_TP1` *(0/1)*
* `TRAIL_ATR_MULT=1.0`
* `TRAIL_USE_HIGH_WATERMARK=True`

### Throughput guards

* `DEDUP_BUSY_WINDOW_MIN=480` (8h) — de-dupe nearby signals
* `SYMBOL_COOLDOWN_MINUTES=120` — cool-down between trades on same symbol
* `MAX_TRADES_PER_DAY` — optional day cap
* Guards: `MAX_TRADES_PER_VARIANT`, `MIN_EQUITY_FRACTION_BEFORE_ABORT`

### Labeling mode (for meta data collection)

* `LABELING_MODE=True` to disable throughput caps (collect maximum trades/signals).

---

## 4) Signals: exactly what qualifies as an entry

**Per symbol (5m base bars):**

1. Resample to **1D** for Donchian upper on `DON_N_DAYS` (shifted by 1 day to avoid look-ahead); align back to 5m.
2. Volume spike on 5m bars using bounded window (30 days → bars).
3. Identify a **breakout bar** where:

   * `close > donch_upper` if `DON_CONFIRM_CLOSE_ABOVE=True`; else `high > donch_upper`.
4. Apply **pullback/stall** rule:

   * **"retest"**: after breakout, price re-visits/retests breakout area within small epsilon (`RETEST_EPS_PCT`) and consolidates briefly; entry triggers on **close above break** (or **rebreak high**), depending on `ENTRY_RULE`.
   * **"mean"**: pullback to MA ± band, then re-impulse.
5. **Regime gate** (ETH 4h MACD histogram):

   * Build ETH 4h from 5m. Compute MACD line, signal, histogram.
   * If `REGIME_REQUIRE_BOTH_POSITIVE=True`, require MACD>signal **and** hist>0; else require hist>0.
   * If `REGIME_BLOCK_WHEN_DOWN=True`, **skip** when regime down; else **size-down** (`REGIME_SIZE_WHEN_DOWN`).
6. **RS filter**: keep `rs_pct ≥ RS_MIN_PERCENTILE` and symbols passing liquidity threshold.
7. **De-dup**: drop busy duplicates within `DEDUP_BUSY_WINDOW_MIN`.
8. Write **signals/signals.parquet** with, at minimum:

   ```
   timestamp (UTC, 5m aligned), symbol, entry (float), don_break_len,
   vol_flag/vol_mult or vol_quant, rs_pct, regime_up (bool),
   pullback_type, entry_rule, atr_1h (at entry)  ← if available
   ```

   Plus enriched features if configured (see §7).

---

## 5) Sizing & order construction (live-ready)

### Position size (qty)

* Compute **initial stop**: `sl_initial = entry - SL_ATR_MULT * ATR_1h_at_entry`.
  (Long-only; invert for shorts if ever needed.)
* **Risk per unit** = `entry - sl_initial`.
* **Cash at risk**:

  * If `RISK_MODE="percent"`: `RISK_PCT * equity_for_sizing`
  * If `RISK_MODE="cash"`: `FIXED_RISK_CASH`
* **Qty** = `cash_risk / max(risk_per_unit, ε)`
* **Caps**: `qty` such that `qty*entry ≤ equity * NOTIONAL_CAP_PCT_OF_EQUITY * MAX_LEVERAGE`.

### Bracket (at submit)

* **TP**: `tp_final = entry + TP_ATR_MULT * ATR_1h_at_entry`
* **Time stop** (if set): exit at `entry_timestamp + TIME_EXIT_HOURS` if still open.

### Optional: partial + trailing behavior

* If `PARTIAL_TP_ENABLED`:

  * Take `PARTIAL_TP_RATIO` at `tp1 = entry + PARTIAL_TP1_ATR_MULT * ATR`.
  * **Move SL to BE** on TP1 if `MOVE_SL_TO_BE_ON_TP1=True`.
* If `TRAIL_AFTER_TP1`:

  * Maintain **high-watermark** (default) and trail by `TRAIL_ATR_MULT * ATR_at_entry`:
    `trail = HWM - (trail_mult*ATR)`. Exit if touched.

> **Deterministic intrabar sequencing**: If a 5m bar contains both SL and TP (or TP1 / trail), use 1m bars when available; otherwise apply a deterministic “first-touch” rule (configured tie-breaker, e.g., `TIE_BREAKER="sl_wins"`). This rule must be mirrored in live fill simulation (see §8).

---

## 6) Execution loop (live bot logic)

**Run every 5 minutes** (aligned to exchange bar close; ensure you account for exchange timestamping lag):

1. **Ingest** latest 5m OHLCV for all symbols (and 1m if using intrabar).
2. **Resample**: update ETH 4h (regime), 1h ATR, daily Donchian.
3. **Universe + RS/Liq**: refresh weekly RS (rebalance day anchored by `RS_REBALANCE_ANCHOR_WEEKDAY`, typically Monday).
4. **Find entries**: new **signals** (as in §4) at the *current completed bar close*.
5. **Score** with meta-model (see §7) → **probability p** for each signal.
6. **Gate**: keep `p ≥ p*` (e.g., `0.55–0.60` based on your EV).
7. **Throughput guards**:

   * Per-symbol **cooldown**
   * **Max trades per day** (optional)
   * **Min ATR%** filter
8. **Compute size** (percent or fixed cash risk).
9. **Submit** bracket:

   * Market/IOC entry (or limit with small epsilon if you prefer maker), plus **OCO** for SL/TP.
   * If partials/trailing: either emulate with **client-side** logic updating the stop/target as events occur, or use exchange native **reduceOnly** orders where supported.
10. **State machine** per open position:

    * On **TP1 fill**: reduce size, record partial pnl/fees, optionally **move SL to BE**.
    * If trailing enabled: update trail price with each new bar/hwm.
    * On **time stop**: close at market/limit.
11. **Bookkeeping/logging**:

    * Persist **trade journal** row on every exit; update **equity curve** after fees.
    * Persist **features at entry** (used by ongoing calibration).

---

## 7) Meta-model: training, scoring, thresholding

### Labels & targets

* For binary target:

  * `hit_tp` (did final TP get hit before SL/time).
* For *max-labeling* rich targets:

  * `mfe_k`: did **max favorable excursion** reach `k × ATR` (e.g., k=2.0) at any point.
* Labeling mode for training data: set `LABELING_MODE=True` to **disable throughput caps** and collect a large, diverse set of trades.

### Features (entry-time only; **no look-ahead**)

Recommended minimal panel (all computed at **entry\_ts** and aligned to 5m):

* **Price/vol**: ATR(1h), `atr_pct = ATR/price`, volume spike multiple or quantile flag/score, raw volume z-score if desired.
* **Trend/momentum**: RSI(1h), ADX(1h), Donchian length used, distance above breakout, ETH 4h **MACD histogram** numeric value.
* **Regime (Markov & daily)**:

  * 4h Markov up state & probability (`markov_state_up_4h`/`markov_prob_up_4h`)
  * 1D trend/vol regimes (`trend_regime_1d`, `vol_regime_1d`, `regime_code_1d`, `vol_prob_low_1d`)
* **Cross-sectional**: `rs_pct` (weekly relative strength percentile)
* **Safety**: min ATR% of price flag, liquidity proxy at entry.

> All of these are already being written into `trades.csv` in research; replicate the same computation online at signal time for live scoring.

### Training protocol (offline)

* CPCV / embargoed splits (temporal) to prevent leakage.
* **Model**: LightGBM classifier (sklearn API).
* **Metrics**: PR-AUC vs prevalence; **Brier**; reliability (ECE).
* **Artifacts**: save model + feature list + calibrator (Platt or isotonic) under `results/meta/*`.

### Threshold selection

* Compute **EV curve** of realized `R` vs probability threshold `t∈[0.3..0.9]`.
* Choose **p\*** where EV is stable and trade count is sufficient (e.g., min N=150).
* Your recent run suggested **p\*=0.55–0.60**; both backtested well.

### **Live scoring step (must-have)**

**Score every signal** in real time (avoid the “15.4% join coverage” artifact from research). Pipeline:

1. Load saved **model** and **calibration**.
2. Build **feature row** for each freshly emitted signal (same columns/order).
3. `proba = calibrator(model.predict_proba(X)[:,1])`
4. Keep **only** signals with `proba ≥ p*`.
5. Log `(timestamp, symbol, proba)` with the trade record.

*(In research you used `apply_meta_filter_and_rebacktest.py` to join OOF preds back to signals; for live you replace that with direct inference.)*

---

## 8) Fill simulation vs exchange realities

* **1m intrabar** is used when available to resolve SL/TP touch order. If both inside a 5m bar and **no 1m**, apply deterministic rule (e.g., `sl_wins`). **Your live bot must mirror this rule** in its PnL accounting to keep research/live consistent.
* **Fees**: apply on every fill (entry, TP1, TP2/final, SL, trail). Use **maker/taker** tier that matches your live account; research default is `FEE_RATE=0.00055`.
* **Slippage**: research currently uses deterministic fills; for live, include **limit-first with short timeout fallback to market** to manage slippage. Log slippage per trade for post-hoc realism adjustments.

---

## 9) Files & schemas (what your bot should read/write)

### `signals/signals.parquet` (inputs to execution)

Minimum:

```
timestamp (UTC 5m), symbol, entry, rs_pct, regime_up (bool),
pullback_type, entry_rule, don_break_len,
atr_1h (if precomputed), vol_mult or vol_flag/quant,   # volume context
# optional extra features if live-scoring here:
rsi_1h, adx_1h, atr_pct, macd_hist_4h,
markov_state_up_4h, markov_prob_up_4h,
trend_regime_1d, vol_regime_1d, vol_prob_low_1d, regime_code_1d
```

### Trade record (append one row on **final exit**)

`results/trades.csv`

```
trade_id, symbol, entry_ts, exit_ts, entry, exit, qty, side,
sl, tp, exit_reason, atr_at_entry,
regime_up, rs_pct, pullback_type, entry_rule, don_break_len,
fees, pnl, pnl_R, mae_over_atr, mfe_over_atr,
markov_state_4h, markov_prob_up_4h, trend_regime_1d, vol_regime_1d,
vol_prob_low_1d, regime_code_1d, regime_1d, markov_state_up_4h,
meta_p (if gated live), risk_mode, risk_budget_cash
```

### Equity curve

`results/equity.csv`

```
timestamp (UTC), equity
```

---

## 10) Research → Live parity checklist

* [ ] Bar alignment: **use completed 5m bars** (no partials).
* [ ] UTC everywhere; exchange time drift monitored.
* [ ] ETH 4h MACD computed via **5m→4h resample**; **shift** so the value at `t` uses info ≤ `t`.
* [ ] ATR(1h) computed via **5m→1h resample**; forward-fill to 5m decision bar; **no look-ahead**.
* [ ] Donchian **daily** upper uses **prior N days** (shifted by 1 day).
* [ ] Volume spike windows are **bounded** (e.g., 30 days).
* [ ] Same **tie-breaker** rule in live fill logic (`sl_wins` if both touched).
* [ ] Same **fees** and **order types** as live venue.
* [ ] Sizing mode parity: `"percent"` vs `"cash"`.
* [ ] Throughput guards mirrored: cooldown, max trades/day.
* [ ] Meta features built **only from info at entry**; **identical columns/order** to training.
* [ ] Calibrator (Platt/isotonic) applied at inference; **p\*** from EV study used unchanged.

---

## 11) Operating the research stack (reference CLI)

### Parameter sweep (guarded)

```
python sweep_donch_params_guarded.py --start 2025-01-01 --end 2025-08-10
# leaderboard: results/leaderboard_guarded.csv
# (with robustness columns if you ran the robust mode)
```

### Pick a baseline & run full backtest

```
python manager.py
# writes results/trades.csv and results/equity.csv
```

### Train meta (example: MFE≥2×ATR label)

```
python meta_model.py --target mfe_k --mfe-k 2.0 \
  --trades-csv results/trades.csv \
  --outdir results/meta_winner
# artifacts: oos_predictions.parquet, metrics.csv, importances, etc.
```

### EV curve & p\* selection

```
python compute_ev_and_select_threshold.py \
  --pred results/meta_winner/oos_predictions.parquet \
  --trades results/trades.csv \
  --pred-ts entry_ts --trades-ts entry_ts \
  --pred-sym symbol --trades-sym symbol \
  --proba-col y_proba \
  --dedup mean --round 5min --tol 10min \
  --out results/meta_winner/ev_curve.csv --min-trades 150
```

### Apply p\* & re-backtest the gated subset (research)

```
python apply_meta_filter_and_rebacktest.py \
  --signals signals/signals.parquet \
  --pred results/meta_winner/oos_predictions.parquet \
  --proba-col y_proba --pred-ts entry_ts --pred-sym symbol \
  --dedup mean --round 5min --tol 10min \
  --pstar 0.60 --out signals/signals_p060.parquet --rebt
```

> **Live**: replace the join approach with **direct inference** on all signals, then gate by p\*.

---

## 12) Live bot architecture (suggested but priority is for the working version of LIVEFADER - it is tested and robust)

**Processes**

1. **Data Collector**: 5m & 1m bars; retries; gap filling; writes to local store or in-mem cache.
2. **Feature Engine**: resamples 5m→1h/4h/1D; computes Donchian, MACD, ATR, RSI/ADX, volume baselines; ETH 4h MACD; Markov regimes; outputs a **feature snapshot** per symbol every 5m.
3. **Signal Engine**: detects Donch+volume+pullback candidates; writes `signals` stream.
4. **Meta Inference**: loads model & calibrator; builds **exact feature vector**; emits `meta_p`; filters by p\*.
5. **Trader**:

   * Sizing (percent or fixed cash risk)
   * Submits entry + OCO bracket (SL/TP)
   * Partial/Trail manager (client-side if exchange lacks native features)
   * Time stop scheduler
6. **Journal & Metrics**:

   * Append trade record on final exit; update equity
   * **Online calibration**: reliability bins, Brier rolling
   * Latency, slippage, reject/error rate

**Persistence**

* Lightweight DB (SQLite/Postgres) or append-only parquet for:

  * Open positions, pending orders, last HWM for trail, cooldown map, daily counters.

**Fault tolerance**

* Idempotent order submission (clientOrderId)
* Rebuild open position state from exchange on restart
* Heartbeats + alerting (missed bar, data gaps, order reject)

---

## 13) Monitoring & drift

**Trading KPIs**

* Max drawdown, MAR (CAGR/DD), daily Sharpe
* Hit rates: SL/TP/time/partial/trail distributions
* Slippage & fees per venue/symbol
* Capacity (notional utilization vs caps)

**Meta-model KPIs**

* Rolling **Brier**, **ECE** (calibration error), reliability plot
* **PR-AUC** by month (proxy), percentage of signals gated out
* EV realized vs expected by probability bands

**Data quality**

* Late/missing bars, time drift vs exchange, resample anomalies
* 1m availability rate for intrabar

---

## 15) Defaults grounded by your recent study (starting point)

* **Entry**: `DON_N_DAYS=20`, `VOL_SPIKE_MODE="multiple"`, `VOL_MULTIPLE=2.0`, `PULLBACK_MODEL="retest"`, `ENTRY_RULE="close_above_break"`, `DON_CONFIRM_CLOSE_ABOVE=1`.
* **Exits**: `SL_ATR_MULT=2.0–2.5`, `TP_ATR_MULT=8–12`, `TIME_EXIT_HOURS=72 or None` *(your sweep favored 72h or off)*.
* **Regime**: ETH 4h MACD require both positive; block when down.
* **Meta gate**: **p\* ≈ 0.55–0.60** (from EV & re-BT); revisit monthly.
* **Sizing**: start with `RISK_MODE="cash"` and a tiny `FIXED_RISK_CASH`, then graduate to % of equity.
* **Partials/trailing**: optional; your sweep showed **no strong need**—keep **off** initially for simplicity.

---

## 16) Common failure modes & fixes

* **Coverage gap** (research join): in live, **score all signals**—no join needed.
* **Indicator look-ahead**: ensure resamples use **closed=right, label=right** and values are **shifted/ffilled** so `value(t)` only uses data ≤ `t`.
* **Exchange latency**: only act on **fully closed** 5m bars; add 3–5s safety delay if needed.
* **Intrabar ambiguity**: if 1m feed goes down, your deterministic rule must match research (e.g., `sl_wins`).
* **Duplicate signals**: enforce `DEDUP_BUSY_WINDOW_MIN` and per-symbol `cooldown`.

---

## 17) Minimal pseudocode (live loop)

```python
while True:
    t = wait_for_completed_5m_bar()

    bars5 = get_5m_all_symbols(t)
    bars1 = get_1m_all_symbols(t)  # optional

    # Build higher TFs & features
    atr_1h = build_atr_1h(bars5)
    macd_eth_4h = build_eth_macd_4h(bars5["ETHUSDT"])
    markov_4h, prob_4h = markov_eth_4h(bars5["ETHUSDT"])
    regimes_1d = daily_regimes(bars5)

    # Universe filter
    rs = weekly_relative_strength(bars5)  # computed on resampled 1w
    liquid = pass_liquidity_threshold(bars5)
    universe = symbols[rs_pct >= RS_MIN & liquid]

    # Signals
    sigs = find_donch_volume_pullback(bars5[universe], atr_1h, macd_eth_4h, t)

    # Meta inference on ALL signals
    X = build_feature_matrix(sigs, atr_1h, macd_eth_4h, rsi/adx, rs, markov_4h, regimes_1d, ...)
    p = calibrator(model.predict_proba(X)[:,1])
    sigs["meta_p"] = p

    # Threshold & throughput guards
    tradable = sigs[(p >= P_STAR) & regime_gate & cooldown_ok & atr_pct_ok]

    for s in tradable.sort_by_time_symbol():
        qty = size_from_risk(s.entry, s.entry - SL_ATR*ATR_1h[s.symbol], equity, mode=RISK_MODE)
        submit_entry_and_bracket(s.symbol, s.entry, qty, sl, tp)

    manage_open_positions(trailing, partials, time_stops, bar=t)

    persist_journal_and_equity()
```

---

## 18) Final notes

* **Edge**: Your meta-gated re-BTs are **encouraging** (e.g., p≥0.55/0.60 subsets materially outperform). The key next step is **live scoring of every signal** with small risk, plus calibration monitoring.
* **Keep it simple initially**: run without partials/trailing and with `RISK_MODE="cash"` tiny size; verify stability; then add complexity (trail, partials) if they improve realized EV.
* **Document exact versions** of pandas/numpy/lightgbm used for reproducibility.
