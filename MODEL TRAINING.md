Short answer: the model does **not** update itself while the bot trades. It’s a fixed artifact that you loaded at startup. If you want it to improve with new data you must retrain/export a new pack and hot-swap it in. Below is a clean, reliable way to do that (manual or automated), wired to the code you already have.

---

# How to keep the meta-model up to date

## What the bot does today

* `WinProbScorer("results/meta_export")` loads a **static** artifact set:
  `donch_meta_lgbm.joblib`, `ohe.joblib`, `feature_names.json`, `pstar.txt`, (optional) `calibrator.joblib`.
* It will only change if you **replace the files** (or repoint a symlink) *and* call `reload()`/`reload_if_changed()` or restart the process.

---

## Recommended update loop (nightly / weekly)

This mirrors your research process and avoids selection bias.

### 0) Make sure you are logging everything

Keep (or add) a CSV/Parquet you append to in live:

* every **candidate signal** (even ones the meta gate rejects)
* the **feature row** you fed the model
* the model **probability** and the **decision** (pass/fail)
* realized **PnL/R** and **exit reason** once the trade closes

This is great for drift monitoring, but the **training set** should still come from the research pipeline (next step) to include *all* would-be signals, not only the ones you traded.

### 1) Extend the research dataset for the latest period

Use the exact research code (no live outcomes needed):

1. Generate signals with your Donch rules (RS, regime, volume, pullback, etc.) for the **new dates**.
2. Backtest to get realized labels (win/loss or R>0 net fees) on those signals.
3. Train the LightGBM meta-model again on **ALL** signals + labels (old + new).
4. Choose **p\*** from the new EV curve.
5. **Export** artifacts.

You already have the exporter; run something like:

```bash
# example date band; pick what fits your cadence
python auto_meta_pipeline.py \
  --start 2025-01-01 --end 2025-08-15 \
  --baseline-tag "DON_DAYS-20_VOL-mult2.0_ENTRY-close_above_break_PB-retest" \
  --outdir results/meta_export_2025-08-15 \
  --export  # ensures the joblib/ohe/feature_names/pstar/calibrator are written
```

(Use whatever CLI flags your repo expects; the goal is: train → compute EV → write artifacts.)

Artifacts expected in the new folder:

```
results/meta_export_2025-08-15/
  donch_meta_lgbm.joblib
  ohe.joblib
  feature_names.json
  pstar.txt
  calibrator.joblib        # optional
  config_snapshot.json
```

### 2) Atomically switch the bot to the new model

Keep your bot pointing at a **stable symlink** (e.g., `results/meta_export`), then swap the link:

```bash
ln -sfn results/meta_export_2025-08-15 results/meta_export
```

Now either:

* **Hot-reload** in process (no restart) if you wired this:

```python
# somewhere in your main loop / once per minute
self.winprob.reload_if_changed()
```

or:

* **Restart** the bot process (simplest).

Either way, the scorer loads the new `joblib/ohe/pstar` and you’re live on the updated model.

### 3) Verify & monitor

* Log a sample of `meta_p` before/after update; make sure distributions look sane.
* Confirm feature width alignment (no “X has N features, expected M” errors).
* Track calibration drift: bucket predicted p into deciles and plot average realized R per bucket.

---

## (Optional) Automate on a schedule

A tiny cron (UTC) that retrains nightly and swaps the symlink only if metrics pass:

```bash
# /etc/cron.d/donch_meta
15 04 * * * cd /root/apps/donch && . .venv/bin/activate && \
  python auto_meta_pipeline.py --start 2024-01-01 --end $(date -u +%F) \
    --outdir results/meta_export_$(date -u +%F) --export \
  && python scripts/check_meta_quality.py results/meta_export_$(date -u +%F) \
  && ln -sfn results/meta_export_$(date -u +%F) results/meta_export \
  && curl -s localhost:YOUR_BOT/trigger_winprob_reload  # or SIGHUP/restart
```

Where `check_meta_quality.py` asserts PR-AUC lift vs prevalence and Brier not worse than prior by >X%.

---

## About “online” updating (learning as it trades)

Not recommended here. While LightGBM can continue training with `init_model`/`model.booster_`, doing so:

* breaks your **purged/embargoed** validation guarantees,
* risks **concept drift miscalibration**,
* complicates rollback.

If you insist: use a **rolling window** (e.g., last 6–12 months), retrain from scratch on that window, recalibrate probabilities, and export a fresh pack. This achieves most of the benefit with fewer pitfalls.

---

## Exact commands you’ll actually use

1. **Retrain + export:**

```bash
python auto_meta_pipeline.py --start 2025-01-01 --end 2025-08-15 \
  --baseline-tag DON20_mult2_retest_closeabove \
  --outdir results/meta_export_2025-08-15 --export
```

2. **Swap live:**

```bash
ln -sfn results/meta_export_2025-08-15 results/meta_export
# either restart the bot or make it call:
# self.winprob.reload_if_changed()
```

3. **Pin libs** (to avoid the sklearn warning you saw):

```bash
pip install "lightgbm==4.5.*" "scikit-learn==1.7.0"  # match training env
```

---

## What if you want to use live outcomes for training?

Still run the **research pipeline** over the same live period and generate labels on **all** eligible signals. Then (optionally) blend in the subset of features you logged at live time for parity checks. Training only on traded signals introduces selection bias—avoid that.

---

### TL;DR

* The model **doesn’t auto-update**. You retrain/export periodically, then hot-swap artifacts (symlink + `reload_if_changed()` or restart).
* Keep the bot pointed at `results/meta_export/` so rotating to a new version is one atomic `ln -sfn`.
* Retrain on a rolling/expanding window via your existing `auto_meta_pipeline.py`; export the pack; swap; monitor.
