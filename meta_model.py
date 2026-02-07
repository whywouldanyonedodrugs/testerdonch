# meta_model.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score
from sklearn.preprocessing import OneHotEncoder

import pyarrow.dataset as ds
import pyarrow.parquet as pq

import lightgbm as lgb
import shap
import config as cfg
import warnings

warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray"
)
from sklearn.inspection import permutation_importance

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise RuntimeError("LightGBM not available. Install with: pip install lightgbm") from e

# --- Feature registry & toggles -------------------------------------------------
CAND_NUM = [
    "entry",
    "atr",
    "atr_1h",
    "atr_pct",
    "don_break_len",
    "don_break_level",
    "don_dist_atr",
    "rs_pct",
    "hour_sin",
    "hour_cos",
    "dow",
    "vol_spike_i",
    "rsi_1h",
    "adx_1h",
    "vol_mult",
    "vol_prob_low_1d",
    #"regime_code_1d",
    #"markov_state_4h",
    "markov_prob_up_4h",
    "markov_state_up_4h",
    "oi_level",
    "oi_notional_est",
    "oi_pct_1h",
    "oi_pct_4h",
    "oi_pct_1d",
    "oi_z_7d",
    "oi_chg_norm_vol_1h",
    "oi_price_div_1h",
    "funding_rate",
    "funding_abs",
    "funding_z_7d",
    "funding_rollsum_3d",
    "funding_oi_div",
    "eth_macd_line_4h",
    "eth_macd_signal_4h",
    "eth_macd_hist_4h",
    "eth_macd_both_pos_4h",
    "days_since_prev_break",
    "consolidation_range_atr",
    "prior_1d_ret",
    "rv_3d",
    "basis_pct",
    "funding_8h",
    "prior_breakout_count",
    "prior_breakout_fail_count",
    "prior_breakout_fail_rate",
    "rs_z",
    "turnover_z",
    "vwap_frac_in_band",
    "vwap_expansion_pct",
    "vwap_slope_pph",
    # --- Perps-wide sentiment (Phase 1) ---
    "sent_rets_1h_z",
    "sent_rets_1d_z",
    "sent_oi_chg_1h_z",
    "sent_oi_chg_1d_z",
    "sent_funding_mean_1d",
    "sent_funding_z_1d",
    "sent_beta_risk_on",
]

ENABLE_INTERACTIONS = False
INTERACTIONS: List[Tuple[str, str]] = [
    ("rs_pct", "eth_macd_both_pos_4h"),
    ("don_dist_atr", "eth_macd_both_pos_4h"),
]

TIME_DECAY_HALFLIFE_DAYS: int | None = 180

# ---- Base LGBM params used for CV + final refit ----
BASE_LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
    n_jobs=-1,
    verbose=-1,
)



def _add_interactions(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    if not pairs:
        return df
    out = df.copy()
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = out[a].astype(float) * out[b].astype(float)
    return out


def _time_decay_weights(ts: pd.Series, halflife_days: int | None) -> pd.Series:
    if halflife_days is None:
        return pd.Series(1.0, index=ts.index)
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    tmax = ts.max()
    age_days = (tmax - ts).dt.total_seconds() / 86400.0
    return pd.Series(np.power(0.5, age_days / float(halflife_days)), index=ts.index)


# =============================================================================
# I/O
# =============================================================================
def load_signals_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        table = ds.dataset(str(p), format="parquet").to_table()
    else:
        table = pq.read_table(str(p))
    df = table.to_pandas()
    # normalize types
    if "symbol" not in df.columns and "sym" in df.columns:
        df = df.rename(columns={"sym": "symbol"})
    df["symbol"] = df["symbol"].astype(str)
    # enforce tz-aware UTC and snap to bar grid
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("5min")
    return df

def export_artifacts(model, feature_names, outdir, cfg_snapshot: dict, calibrator=None, pstar: float | None=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) sklearn wrapper
    joblib.dump(model, out / "donch_meta_lgbm.joblib")

    # 2) raw LightGBM booster (optional but handy)
    try:
        model.booster_.save_model(str(out / "donch_meta_lgbm.txt"))
    except Exception:
        pass

    # 3) features used at train/infer (exact order for live)
    with open(out / "feature_names.json", "w") as f:
        json.dump(list(feature_names), f)

    # 4) snapshot of the feature-building config
    with open(out / "config_snapshot.json", "w") as f:
        json.dump(cfg_snapshot, f)

    # 5) optional probability calibrator
    if calibrator is not None:
        joblib.dump(calibrator, out / "calibrator.joblib")

    # 6) optional probability threshold
    if pstar is not None:
        with open(out / "pstar.txt", "w") as f:
            f.write(f"{float(pstar):.4f}\n")

def _p(path: Path | str) -> Path:
    return Path(path).resolve()

def load_trades(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = cfg.RESULTS_DIR / "trades.csv"
    path = _p(path)
    if not path.exists():
        raise FileNotFoundError(f"trades file not found: {path}")
    df = pd.read_csv(path)
    for c in ("entry_ts", "exit_ts"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df

def load_signals(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = cfg.SIGNALS_DIR / "signals.parquet"
    path = _p(path)
    if not path.exists():
        raise FileNotFoundError(f"signals parquet not found: {path}")
    tbl = pq.read_table(path)
    df = tbl.to_pandas()
    if "timestamp" not in df.columns or "symbol" not in df.columns:
        raise ValueError("signals.parquet must have ['timestamp','symbol']")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df

# =============================================================================
# Feature engineering (entry-time only — no leakage)
# =============================================================================
def _time_col(df: pd.DataFrame) -> pd.Series:
    if "entry_ts" in df.columns:
        return pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    raise KeyError("Neither 'entry_ts' nor 'timestamp' found in dataframe (feature builder).")

def _mk_features(signals_or_joined: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build numeric + categorical features from entry-time information only.
    Returns: features_df, num_cols, cat_cols
    """
    df = signals_or_joined.copy()

    # Base numerics derived if present
    if "atr" in df.columns and "entry" in df.columns:
        df["atr_pct"] = df["atr"] / df["entry"]
    atr_scale = df["atr_1h"] if "atr_1h" in df.columns else (df["atr"] if "atr" in df.columns else None)
    if {"entry", "don_break_level"}.issubset(df.columns) and atr_scale is not None:
        df["don_dist_atr"] = (df["entry"] - df["don_break_level"]) / atr_scale.replace(0, np.nan)

    # Time features (UTC)
    ts = _time_col(df)
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Binary from bools (robust)
    if "vol_spike" in df.columns:
        df["vol_spike_i"] = df["vol_spike"].astype("int8")
    if "regime_up" in df.columns:
        df["regime_up_i"] = df["regime_up"].astype("int8")

    # Clean RS
    if "rs_pct" in df.columns:
        df["rs_pct"] = df["rs_pct"].fillna(-1)

    # Candidate numerics (keep what's present)
    num_cols = [c for c in CAND_NUM if c in df.columns]

    # --- NEW: drop degenerate numeric features (all-NaN or near-constant) ---
    sane_num_cols: List[str] = []
    for c in num_cols:
        s = df[c]
        # 1) all-NaN → drop
        if not s.notna().any():
            continue
        # 2) near-constant → optional drop (threshold is very small)
        #    This will drop things like don_break_len when it's hard-coded to 20 everywhere.
        try:
            vals = s.dropna().to_numpy()
            if vals.size > 1 and np.nanstd(vals) < 1e-8:
                continue
        except Exception:
            # fallback: keep if we can't compute std
            pass
        sane_num_cols.append(c)
    num_cols = sane_num_cols

    if ENABLE_INTERACTIONS:
        df = _add_interactions(df, INTERACTIONS)
        for a, b in INTERACTIONS:
            col = f"{a}__x__{b}"
            if col in df.columns and col not in num_cols:
                num_cols.append(col)

    # Categorical feature list
    cat_cols = [c for c in ["pullback_type", "entry_rule", "regime_1d"] if c in df.columns]

    feats = df[["symbol"] + (["entry_ts"] if "entry_ts" in df.columns else ["timestamp"]) + num_cols + cat_cols].copy()
    feats.rename(columns={"entry_ts": "entry_ts", "timestamp": "entry_ts"}, inplace=True)
    return feats, num_cols, cat_cols


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)

def _fit_ohe(train_df: pd.DataFrame, cat_cols: List[str]) -> OneHotEncoder:
    ohe = _make_ohe()
    if not cat_cols:
        ohe.fit(np.empty((len(train_df),0)))
        return ohe
    ohe.fit(train_df[cat_cols].astype(str))
    return ohe

def _apply_ohe(ohe: OneHotEncoder, df: pd.DataFrame, cat_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    if not cat_cols:
        return np.zeros((len(df),0), dtype=np.float64), []
    Xo = ohe.transform(df[cat_cols].astype(str))
    names = list(ohe.get_feature_names_out(cat_cols))
    return np.asarray(Xo), names

def assemble_matrix(feats: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
                    train_idx: np.ndarray, test_idx: np.ndarray):
    """Return (X_train, X_test, cols_all, ohe) for one split."""
    ohe = _fit_ohe(feats.iloc[train_idx], cat_cols)
    Xn_train = feats.iloc[train_idx][num_cols].astype(np.float64).fillna(0.0).to_numpy()
    Xn_test  = feats.iloc[test_idx][num_cols].astype(np.float64).fillna(0.0).to_numpy()
    Xo_train, ohe_names = _apply_ohe(ohe, feats.iloc[train_idx], cat_cols)
    Xo_test, _          = _apply_ohe(ohe, feats.iloc[test_idx], cat_cols)
    X_train = np.hstack([Xn_train, Xo_train])
    X_test  = np.hstack([Xn_test, Xo_test])
    cols = num_cols + ohe_names
    return X_train, X_test, cols, ohe

# =============================================================================
# Targets
# =============================================================================
def compute_targets(df_joined: pd.DataFrame, mode: str = "pnl_pos", r_threshold: float = 0.0, mfe_k: float = 2.0) -> pd.Series:
    """
    Build binary meta-label y.

    mode:
      - "pnl_pos": label = 1 if pnl_R >= r_threshold else 0
      - "hit_tp":  label = 1 if exit_reason in {"tp_final","trail"} else 0
      - "mfe_k":   label = 1 if mfe_over_atr >= mfe_k else 0
    """
    mode = str(mode).lower()
    if mode == "pnl_pos":
        if "pnl_R" not in df_joined.columns:
            raise ValueError("pnl_pos target requires 'pnl_R' in trades.csv")
        x = df_joined["pnl_R"].astype(float).values
        y = (x >= float(r_threshold)).astype(np.int8)
        return pd.Series(y, index=df_joined.index, name="y")
    elif mode == "hit_tp":
        if "exit_reason" not in df_joined.columns:
            raise ValueError("hit_tp target requires 'exit_reason' in trades.csv")
        good = df_joined["exit_reason"].isin(["tp_final","trail"])
        return good.astype(np.int8).rename("y")
    elif mode == "mfe_k":
        if "mfe_over_atr" not in df_joined.columns:
            raise ValueError("mfe_k target requires 'mfe_over_atr' in trades.csv")
        return (df_joined["mfe_over_atr"].astype(float) >= float(mfe_k)).astype(np.int8).rename("y")
    else:
        raise ValueError(f"Unknown target mode: {mode}")

# =============================================================================
# Time splits: CPCV with purging + embargo
# =============================================================================
@dataclass
class CPCVConfig:
    blocks: int = 12
    k_test: int = 3
    embargo: int = 1
    max_splits: int = 30

def _blocks_for_dates(dates: pd.Series, n_blocks: int) -> List[np.ndarray]:
    n = len(dates)
    sizes = [n // n_blocks] * n_blocks
    for i in range(n % n_blocks):
        sizes[i] += 1
    masks = []
    start = 0
    for s in sizes:
        m = np.zeros(n, dtype=bool)
        m[start:start+s] = True
        masks.append(m)
        start += s
    return masks

def _purged_train_mask(n_rows: int, block_masks: List[np.ndarray], test_ids: List[int], embargo: int) -> Tuple[np.ndarray, np.ndarray]:
    test_mask = np.zeros(n_rows, dtype=bool)
    for j in range(len(block_masks)):
        if j in test_ids:
            test_mask |= block_masks[j]
    train_mask = ~test_mask
    for j in test_ids:
        for e in range(1, embargo+1):
            if j - e >= 0:
                train_mask &= ~block_masks[j - e]
            if j + e < len(block_masks):
                train_mask &= ~block_masks[j + e]
    return train_mask, test_mask

def generate_cpcv_masks(dates: pd.Series, cfgc: CPCVConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(dates, utc=True, errors="coerce").sort_values().reset_index(drop=True)
    blocks = _blocks_for_dates(dates, cfgc.blocks)
    from itertools import combinations
    combos = []
    for comb in combinations(range(cfgc.blocks), cfgc.k_test):
        combos.append(list(comb))
        if len(combos) >= cfgc.max_splits:
            break
    masks = []
    n = len(dates)
    for test_ids in combos:
        tr, te = _purged_train_mask(n, blocks, test_ids, cfgc.embargo)
        if tr.sum() >= 100 and te.sum() >= 50:
            masks.append((tr, te))
    return masks

# =============================================================================
# Training / Evaluation
# =============================================================================
def train_one_fold(
    X_tr,
    y_tr,
    X_te,
    y_te,
    seed: int,
    imbalance_weight: bool = True,
    feature_names: List[str] | None = None,
    sample_weight: Optional[np.ndarray] = None,
):
    """
    Train LGBM on a *DataFrame* with synthetic names (f0..fN), and align X_te to the same columns.
    This avoids sklearn name warnings and SHAP/PI shape mismatches.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, brier_score_loss
    import lightgbm as lgb

    # class weights for imbalance
    spw = 1.0
    if imbalance_weight:
        p = max(1, int((y_tr == 1).sum()))
        n = max(1, int((y_tr == 0).sum()))
        spw = n / p

    # --- Build train DataFrame with stable names ---
    tr_cols = [f"f{i}" for i in range(X_tr.shape[1])]
    X_tr_df = pd.DataFrame(X_tr, columns=tr_cols)

    # --- Build test DataFrame and align columns to train (add missing as 0, drop extras) ---
    te_cols_temp = [f"f{i}" for i in range(X_te.shape[1])]
    X_te_df = pd.DataFrame(X_te, columns=te_cols_temp)
    if X_te_df.shape[1] != len(tr_cols):
        X_te_df = X_te_df.reindex(columns=tr_cols, fill_value=0)
    else:
        X_te_df.columns = tr_cols

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=100,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_tr_df, y_tr, **fit_kwargs)          # fit with *named* columns
    proba = model.predict_proba(X_te_df)[:, 1]  # predict on *aligned* DF

    pr_auc = average_precision_score(y_te, proba)
    brier = brier_score_loss(y_te, proba)

    return model, proba, {"pr_auc": float(pr_auc), "brier": float(brier)}


def oos_permutation_importance(model, X_te, y_te, cols: List[str], seed: int = 42, repeats: int = 8):
    X_df = pd.DataFrame(X_te, columns=cols)
    pi = permutation_importance(
        model, X_df, y_te,
        scoring="average_precision", n_repeats=repeats, random_state=seed, n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": cols,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values("importance_mean", ascending=False)
    return imp

def oos_shap_importance(model, X_te, cols: List[str], nsamples_limit: int = 2000):
    # Align test to the model's train-time feature names
    train_cols = list(getattr(model, "feature_name_", [f"f{i}" for i in range(X_te.shape[1])]))
    _Xte_cols_tmp = [f"f{i}" for i in range(X_te.shape[1])]
    X_te_df = pd.DataFrame(X_te, columns=_Xte_cols_tmp)
    if X_te_df.shape[1] != len(train_cols):
        X_te_df = X_te_df.reindex(columns=train_cols, fill_value=0)
    else:
        X_te_df.columns = train_cols

    # Sample rows for SHAP if needed
    if X_te_df.shape[0] > nsamples_limit:
        sel = np.random.RandomState(0).choice(X_te_df.shape[0], nsamples_limit, replace=False)
        Xs = X_te_df.iloc[sel]
    else:
        Xs = X_te_df

    try:
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xs)
        if isinstance(sv, list):  # binary classifiers sometimes return [neg, pos]
            sv = sv[1] if len(sv) == 2 else sv[0]
        imp = pd.DataFrame({
            "feature": Xs.columns,
            "mean_abs_shap": np.mean(np.abs(sv), axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        return imp
    except Exception as e:
        print(f"[warn] SHAP failed: {e}")
        return pd.DataFrame(columns=["feature","mean_abs_shap"])


# =============================================================================
# Pipeline
# =============================================================================
def build_dataset(trades_csv: Optional[str], signals_parquet: Optional[str]):
    """
    Robust join of trades with signals.
    - Accepts signals as a single parquet file OR a directory of per-symbol parquet files.
    - Joins on (symbol, entry_ts ~= timestamp) with exact match first, then asof (nearest)
      using an inferred tolerance from the signal bar interval.
    - Falls back to a normalized symbol join (strips leading digits like '10000' from '10000COQUSDT').
    - Emits diagnostics showing why a join might be empty.
    """
    import re
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # ---- helpers ------------------------------------------------------------
    def _norm_sym(x: str) -> str:
        if pd.isna(x):
            return x
        s = str(x)
        s = re.sub(r'^\d+', '', s)  # drop leading digits (e.g., 10000COQUSDT -> COQUSDT)
        return s.upper()

    def _read_trades(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "entry_ts" not in df or "symbol" not in df:
            raise ValueError("trades.csv must contain columns ['entry_ts','symbol', ...]")
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
        if df["entry_ts"].isna().any():
            bad = df[df["entry_ts"].isna()].head(5)
            raise ValueError(f"Some trade 'entry_ts' could not be parsed to datetime. Sample:\n{bad}")
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["symbol_norm"] = df["symbol"].map(_norm_sym)
        # floor to seconds using lowercase 's' (avoid deprecated 'S')
        df["entry_ts_floor"] = df["entry_ts"].dt.floor("s")
        return df

    def _read_signals_any(path: str) -> pd.DataFrame:
        p = Path(path)
        if p.is_dir():
            files = sorted(list(p.glob("*.parquet")))
            if not files:
                raise ValueError(f"Signals directory has no .parquet files: {p}")
            parts = []
            for f in files:
                parts.append(pd.read_parquet(f))
            sig = pd.concat(parts, ignore_index=True)
        else:
            sig = pd.read_parquet(p)
        # try to locate time + symbol columns
        tcol = "timestamp" if "timestamp" in sig.columns else ("entry_ts" if "entry_ts" in sig.columns else None)
        if tcol is None:
            raise ValueError("signals.parquet must have 'timestamp' (preferred) or 'entry_ts'.")
        if "symbol" not in sig.columns:
            raise ValueError("signals.parquet must contain 'symbol'.")
        sig = sig.copy()
        sig[tcol] = pd.to_datetime(sig[tcol], utc=True, errors="coerce")
        if sig[tcol].isna().any():
            bad = sig[sig[tcol].isna()].head(5)
            raise ValueError(f"Some signal timestamps could not be parsed to datetime. Sample:\n{bad}")
        sig["symbol"] = sig["symbol"].astype(str).str.upper()
        sig["symbol_norm"] = sig["symbol"].map(_norm_sym)
        sig["sig_ts"] = sig[tcol].dt.floor("s")  # consistent floor w/ trades
        # de-dup on (symbol, sig_ts) if necessary
        sig = sig.sort_values(["symbol", "sig_ts"]).drop_duplicates(["symbol", "sig_ts"], keep="last")
        return sig

    def _infer_bar_tol_seconds(sig: pd.DataFrame) -> int:
        # infer per-symbol median spacing and then take global median
        g = sig.sort_values(["symbol", "sig_ts"]).groupby("symbol")["sig_ts"]
        diffs = g.diff().dropna().dt.total_seconds()
        if diffs.empty:
            # fallback 60s if no diffs found (shouldn’t happen unless only 1 row)
            return 60
        bar = int(np.median(diffs))
        # allow some slop (1.5x bar) to catch minor shift
        return max(1, int(round(bar * 1.5)))

    # --- load ---
    trades = load_trades(trades_csv)              # must have: symbol, entry_ts, ... targets/labels
    signals = load_signals(signals_parquet)       # must have: symbol, timestamp, entry, atr, rs_pct, ...

    # --- normalize time columns to tz-aware UTC ---
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")

    # --- pick required/optional signal columns (keep 'timestamp' name to avoid making a 2nd 'entry_ts') ---
    base_cols_req = [
        "timestamp", "symbol", "entry", "atr", "don_break_len", "don_break_level",
        "pullback_type", "entry_rule", "rs_pct",
    ]
    extras = [
        "atr_1h","rsi_1h","adx_1h","vol_mult","atr_pct","don_dist_atr",
        "eth_macd_hist_4h","eth_macd_line_4h","eth_macd_signal_4h","eth_macd_both_pos_4h",
        "days_since_prev_break","consolidation_range_atr","prior_1d_ret","rv_3d",
        "markov_state_4h","markov_prob_up_4h","vol_prob_low_1d","regime_code_1d",
        "markov_state_up_4h","regime_1d_NA_HIGH_VOL","regime_1d_NA_LOW_VOL",

        # --- OI + Funding (new) ---
        "oi_level","oi_notional_est","oi_pct_1h","oi_pct_4h","oi_pct_1d",
        "oi_z_7d","oi_chg_norm_vol_1h","oi_price_div_1h",
        "funding_rate","funding_abs","funding_z_7d","funding_rollsum_3d","funding_oi_div",
    ]

    missing = [c for c in base_cols_req if c not in signals.columns]
    if missing:
        raise ValueError(f"signals.parquet missing columns: {missing}")
    use_cols = base_cols_req + [c for c in extras if c in signals.columns]
    sig = signals[use_cols].copy()

    # --- exact inner join on (symbol, entry_ts == timestamp) ---
    df = pd.merge(
        trades,
        sig,
        left_on=["symbol", "entry_ts"],
        right_on=["symbol", "timestamp"],
        how="inner",
        suffixes=("", "_sig"),
        validate="one_to_one",  # be explicit; will raise if duplicates happen
    )

    # --- post-merge cleanup: drop the RHS timestamp, dedupe any accidental duped names ---
    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)

    if pd.Index(df.columns).duplicated().any():
        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    # Safety: ensure entry_ts is a Series, not a DataFrame (in case the input had duped labels)
    if isinstance(df["entry_ts"], pd.DataFrame):
        df["entry_ts"] = df["entry_ts"].iloc[:, 0]

    # --- optional: carry regime flag if present in trades ---
    if "regime_up" in df.columns:
        df["regime_up"] = df["regime_up"].astype(int)

    # --- build features (entry-time only) ---
    feats, num_cols, cat_cols = _mk_features(df)
    feats = feats.sort_values("entry_ts").reset_index(drop=True)

    return df, feats, num_cols, cat_cols


@dataclass
class CPCVArgs:
    blocks: int = 12
    k_test: int = 3
    embargo: int = 1
    max_splits: int = 30

def describe_meta_schema(
    trades_csv: Optional[str] = None,
    signals_parquet: Optional[str] = None,
    target: str = "pnl_pos",
    r_threshold: float = 0.0,
    mfe_k: float = 2.0,
    max_rows: int | None = 200_000,
    outdir: Optional[str] = None,
) -> None:
    """
    Build the meta-training frame once and print a compact schema report:
      - columns & dtypes
      - non-null / missing counts
      - basic stats for numeric features
      - Pearson correlation with the chosen target, where defined
      - simple snake_case naming sanity checks

    Optionally saves CSVs into outdir (default: cfg.RESULTS_DIR / "meta").
    """
    import re

    # --- load & build exactly like training does ---
    df_joined, feats, num_cols, cat_cols = build_dataset(trades_csv, signals_parquet)
    y = compute_targets(df_joined, mode=target, r_threshold=r_threshold, mfe_k=mfe_k)
    feats = feats.copy()
    feats["y"] = y.values

    if max_rows is not None and len(feats) > max_rows:
        # keep the most recent rows – usually more relevant for current behaviour
        feats = feats.iloc[-max_rows:].reset_index(drop=True)

    n_rows = len(feats)
    print(f"\n[meta_schema] rows in dataset: {n_rows}")

    # --- numeric feature summary ------------------------------------------------
    num_present = [c for c in num_cols if c in feats.columns]
    rows_num: list[dict] = []

    for col in num_present:
        s = feats[col]
        nn = int(s.notna().sum())
        na = int(s.isna().sum())
        na_frac = float(na / n_rows) if n_rows else float("nan")

        # stats; use float conversion but be robust to weird dtypes
        s_float = pd.to_numeric(s, errors="coerce")
        stats = {
            "mean": float(s_float.mean(skipna=True)) if nn else float("nan"),
            "std": float(s_float.std(skipna=True)) if nn > 1 else float("nan"),
            "min": float(s_float.min(skipna=True)) if nn else float("nan"),
            "max": float(s_float.max(skipna=True)) if nn else float("nan"),
        }

        # correlation vs target y (only if enough non-null)
        corr = float("nan")
        try:
            if "y" in feats.columns:
                df_corr = pd.DataFrame(
                    {"y": pd.to_numeric(feats["y"], errors="coerce"), "x": s_float}
                ).dropna()
                if len(df_corr) >= 30 and df_corr["x"].std() > 0:
                    corr = float(df_corr[["y", "x"]].corr().iloc[0, 1])
        except Exception:
            corr = float("nan")

        rows_num.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "n_non_null": nn,
                "n_null": na,
                "null_frac": na_frac,
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "corr_y": corr,
            }
        )

    df_num = pd.DataFrame(rows_num)
    if not df_num.empty:
        df_num = df_num.sort_values("corr_y", ascending=False)

    # --- categorical feature summary -------------------------------------------
    rows_cat: list[dict] = []
    for col in cat_cols:
        s = feats[col].astype("string")
        nn = int(s.notna().sum())
        na = int(s.isna().sum())
        na_frac = float(na / n_rows) if n_rows else float("nan")
        nunique = int(s.nunique(dropna=True))

        top = s.value_counts(dropna=True).head(5)
        top_repr = "; ".join(f"{idx} ({cnt})" for idx, cnt in top.items())

        # optional: mean target by category, for top categories only
        pos_rate_repr = ""
        if "y" in feats.columns and nn:
            try:
                grp = (
                    feats.dropna(subset=[col])
                    .groupby(col)["y"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(5)
                )
                pos_rate_repr = "; ".join(
                    f"{idx}: {rate:.3f}" for idx, rate in grp.items()
                )
            except Exception:
                pos_rate_repr = ""

        rows_cat.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "n_non_null": nn,
                "n_null": na,
                "null_frac": na_frac,
                "n_unique": nunique,
                "top_values": top_repr,
                "top_pos_rate": pos_rate_repr,
            }
        )

    df_cat = pd.DataFrame(rows_cat)

    # --- naming sanity checks ---------------------------------------------------
    bad_names = []
    snake = re.compile(r"^[a-z0-9_]+$")
    for col in num_present + cat_cols:
        if not snake.match(col):
            bad_names.append(col)

    if bad_names:
        print("\n[meta_schema] WARNING: non-snake_case feature names detected:")
        for c in bad_names:
            print(f"   - {c}")

    # --- print summaries --------------------------------------------------------
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)

    if not df_num.empty:
        print("\n=== Numeric features (sorted by corr_y) ===")
        print(
            df_num.to_string(
                index=False,
                float_format=lambda x: f"{x: .4f}" if isinstance(x, float) else str(x),
            )
        )
    else:
        print("\n[meta_schema] No numeric features found.")

    if not df_cat.empty:
        print("\n=== Categorical features ===")
        print(df_cat.to_string(index=False))
    else:
        print("\n[meta_schema] No categorical features found.")

    # --- optional CSV export ----------------------------------------------------
    out = _p(outdir or (cfg.RESULTS_DIR / "meta"))
    out.mkdir(parents=True, exist_ok=True)
    if not df_num.empty:
        df_num.to_csv(out / "schema_numeric.csv", index=False)
    if not df_cat.empty:
        df_cat.to_csv(out / "schema_categorical.csv", index=False)

    print(f"\n[meta_schema] Saved schema CSVs to: {out}")



def run_meta(
    trades_csv: Optional[str],
    signals_parquet: Optional[str],
    target: str = "pnl_pos",
    r_threshold: float = 0.0,
    mfe_k: float = 2.0,
    blocks: int = 12,
    k_test: int = 3,
    embargo: int = 1,
    max_splits: int = 30,
    outdir: Optional[str] = None,
    random_seed: int = 42,
    pstar: Optional[float] = None,
):
    outdir = _p(outdir or (cfg.RESULTS_DIR / "meta"))
    outdir.mkdir(parents=True, exist_ok=True)

    df_joined, feats, num_cols, cat_cols = build_dataset(trades_csv, signals_parquet)
    y = compute_targets(df_joined, mode=target, r_threshold=r_threshold, mfe_k=mfe_k)

    feats["y"] = y.values
    feats["symbol"] = feats["symbol"].astype(str)
    feats["sample_weight"] = _time_decay_weights(feats["entry_ts"], TIME_DECAY_HALFLIFE_DAYS).values

    masks = generate_cpcv_masks(feats["entry_ts"], CPCVConfig(blocks=blocks, k_test=k_test, embargo=embargo, max_splits=max_splits))
    if not masks:
        raise RuntimeError("No valid CPCV masks. Try reducing --blocks or --k-test, or ensure enough rows for splits.")

    all_metrics: List[Dict[str, float]] = []
    all_preds: List[pd.DataFrame] = []
    perm_fold_list: List[pd.DataFrame] = []
    shap_fold_list: List[pd.DataFrame] = []

    for fold, (tr_mask, te_mask) in enumerate(masks, start=1):
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        X_tr, X_te, cols_all, _ohe = assemble_matrix(feats, num_cols, cat_cols, tr_idx, te_idx)
        y_tr = feats.iloc[tr_idx]["y"].values.astype(int)
        y_te = feats.iloc[te_idx]["y"].values.astype(int)

        w_tr = feats.iloc[tr_idx]["sample_weight"].values
        model, proba, m = train_one_fold(
            X_tr,
            y_tr,
            X_te,
            y_te,
            seed=random_seed + fold,
            feature_names=cols_all,
            sample_weight=w_tr,
        )

        fold_preds = pd.DataFrame({
            "entry_ts": feats.iloc[te_idx]["entry_ts"].values,
            "symbol": feats.iloc[te_idx]["symbol"].values,
            "y_true": y_te,
            "y_proba": proba,
            "fold": fold
        })
        all_preds.append(fold_preds)

        # OOS importances (aligned to model's f-names)
        try:
            train_cols = list(getattr(model, "feature_name_", [f"f{i}" for i in range(X_te.shape[1])]))
            _Xte_cols_tmp = [f"f{i}" for i in range(X_te.shape[1])]
            X_te_df = pd.DataFrame(X_te, columns=_Xte_cols_tmp)
            if X_te_df.shape[1] != len(train_cols):
                X_te_df = X_te_df.reindex(columns=train_cols, fill_value=0)
            else:
                X_te_df.columns = train_cols

            perm_imp = permutation_importance(
                model, X_te_df, y_te,
                scoring="average_precision", n_repeats=8, random_state=random_seed + fold, n_jobs=-1,
            )
            perm_df = pd.DataFrame({
                "feature": train_cols,
                "importance_mean": perm_imp.importances_mean,
                "importance_std": perm_imp.importances_std,
                "fold": fold,
                "weight": len(te_idx),
            })
            perm_fold_list.append(perm_df)
        except Exception as e:
            print(f"[warn] permutation importance failed (fold {fold}): {e}")

        shap_imp = oos_shap_importance(model, X_te, cols_all)
        shap_imp["fold"] = fold
        shap_imp["weight"] = len(te_idx)
        shap_fold_list.append(shap_imp)

        m.update(dict(fold=fold, n_train=len(tr_idx), n_test=len(te_idx)))
        all_metrics.append(m)
        print(f"[fold {fold:02d}] PR-AUC={m['pr_auc']:.4f}  Brier={m['brier']:.4f}  (n_tr={m['n_train']}, n_te={m['n_test']})")

    preds = pd.concat(all_preds, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(preds), outdir / "oos_predictions.parquet")

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(outdir / "metrics.csv", index=False)

    features_to_drop: List[str] = []
    perm_summary = None
    if perm_fold_list:
        perm_all = pd.concat(perm_fold_list, ignore_index=True)

        def _agg_perm(d: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "importance_mean_weighted": float(np.average(d["importance_mean"], weights=d["weight"])),
                "importance_median": float(np.median(d["importance_mean"])),
            })

        perm_summary = perm_all.groupby("feature", as_index=False).apply(
            _agg_perm,
            include_groups=False,
        ).reset_index(drop=True)
        perm_summary.sort_values("importance_mean_weighted", ascending=False).to_csv(
            outdir / "perm_importance_oos.csv", index=False
        )

        features_to_drop = [
            f for f in perm_summary.loc[perm_summary["importance_median"] <= 0, "feature"].tolist()
            if f in num_cols
        ]
        if features_to_drop:
            print(f"[perm] pruning features with median ≤ 0: {features_to_drop}")

    if shap_fold_list:
        shap_all = pd.concat(shap_fold_list, ignore_index=True)
        g2 = shap_all.groupby("feature", as_index=False).apply(
            lambda d: pd.Series({"mean_abs_shap_weighted": float(np.average(d["mean_abs_shap"], weights=d["weight"]))}),
            include_groups=False,
        ).reset_index(drop=True)
        g2.sort_values("mean_abs_shap_weighted", ascending=False).to_csv(outdir / "shap_importance_oos.csv", index=False)

    print("\n=== OOS summary ===")
    print(metrics.agg({"pr_auc":["mean","median"], "brier":["mean","median"], "n_test":"sum"}).to_string())

    # Heuristic threshold suggestion
    grid = np.linspace(0.1, 0.9, 9)
    best_t, best_score = None, -1.0
    for t in grid:
        yhat = (preds["y_proba"].values >= t).astype(int)
        prec = precision_score(preds["y_true"].values, yhat, zero_division=0)
        if prec > best_score:
            best_t, best_score = t, prec
    print(f"Suggested probability threshold (heuristic): p* ≈ {best_t:.2f}")

    (outdir / "run_args.json").write_text(json.dumps(dict(
        trades_csv=str(trades_csv or cfg.RESULTS_DIR / "trades.csv"),
        signals_parquet=str(signals_parquet or cfg.SIGNALS_DIR / "signals.parquet"),
        target=target, r_threshold=r_threshold, mfe_k=mfe_k,
        blocks=blocks, k_test=k_test, embargo=embargo, max_splits=max_splits,
    ), indent=2))

    print(f"\nSaved to {outdir}:\n"
          f"- oos_predictions.parquet\n- metrics.csv\n- perm_importance_oos.csv\n- shap_importance_oos.csv")

    # ---------------- FINAL REFIT ON ALL DATA + EXPORT ----------------
    # Build full matrix on ALL rows, using the same recipe (numerics + OHE)
    all_idx = np.arange(len(feats))

    # Numeric block
    num_cols_final = [c for c in num_cols if c not in features_to_drop]
    Xn_all = feats.iloc[all_idx][num_cols_final].astype(np.float64).fillna(0.0).to_numpy()

    # Categorical block (fit OHE on ALL data to define serving columns)
    ohe_all = _fit_ohe(feats.iloc[all_idx], cat_cols)
    Xo_all, ohe_names = _apply_ohe(ohe_all, feats.iloc[all_idx], cat_cols)

    # Final feature matrix and ordered names for serving
    X_all = np.hstack([Xn_all, Xo_all])
    cols_all = num_cols_final + ohe_names
    y_all = feats["y"].values.astype(int)
    w_all = feats["sample_weight"].values

    # Class imbalance weight on full data
    pos = max(1, int((y_all == 1).sum()))
    neg = max(1, int((y_all == 0).sum()))
    spw = neg / pos

    final_params = dict(BASE_LGBM_PARAMS)
    final_params["scale_pos_weight"] = spw
    final_params["random_state"] = 1337

    # Train final model with stable column names (f0..fN)
    fcols_all = [f"f{i}" for i in range(X_all.shape[1])]
    X_all_df = pd.DataFrame(X_all, columns=fcols_all)
    final = LGBMClassifier(**final_params)
    final.fit(X_all_df, y_all, sample_weight=w_all)

    # Optional: probability calibration using OOS predictions
    calibrator = None
    try:
        from sklearn.isotonic import IsotonicRegression
        p_oos = preds["y_proba"].values
        y_oos = preds["y_true"].values
        if np.isfinite(p_oos).all() and 0.0 <= p_oos.min() and p_oos.max() <= 1.0:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p_oos, y_oos)
            calibrator = ir
    except Exception as e:
        print(f"[warn] calibration skipped: {e}")

    # Snapshot of feature-building config (helps ensure live parity)
    cfg_snapshot = dict(
        DONCH_BASIS=str(getattr(cfg, "DONCH_BASIS", "days")),
        DON_N_DAYS=int(getattr(cfg, "DON_N_DAYS", 20)),
        VOL_SPIKE_MODE=str(getattr(cfg, "VOL_SPIKE_MODE", "multiple")),
        VOL_MULTIPLE=float(getattr(cfg, "VOL_MULTIPLE", 2.0)),
        VOL_QUANTILE_Q=float(getattr(cfg, "VOL_QUANTILE_Q", 0.95)),
        VOL_LOOKBACK_DAYS=int(getattr(cfg, "VOL_LOOKBACK_DAYS", 30)),
        ATR_TIMEFRAME=str(getattr(cfg, "ATR_TIMEFRAME", "1h")),
        ATR_LEN=int(getattr(cfg, "ATR_LEN", 14)),
        REGIME_TIMEFRAME=str(getattr(cfg, "REGIME_TIMEFRAME", "4h")),
        REGIME_MACD=tuple(map(int, (
            getattr(cfg, "REGIME_MACD_FAST", 12),
            getattr(cfg, "REGIME_MACD_SLOW", 26),
            getattr(cfg, "REGIME_MACD_SIGNAL", 9),
        ))),
        RS_MIN_PERCENTILE=int(getattr(cfg, "RS_MIN_PERCENTILE", 70)),
    )

    # Where to export (use the same outdir by default)
    export_dir = outdir
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save the OHE used for categorical mapping at inference time
    joblib.dump(ohe_all, export_dir / "ohe.joblib")

    # Export everything needed for live inference
    export_artifacts(
        model=final,
        feature_names=cols_all,      # exact order the live bot must produce
        outdir=export_dir,
        cfg_snapshot=cfg_snapshot,
        calibrator=calibrator,
        pstar=pstar,                 # optional deployment threshold from CLI
    )
    print(f"[export] artifacts saved to {export_dir}")

# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Meta-model with alternate targets (pnl_pos / hit_tp / mfe_k)"
    )
    ap.add_argument("--trades-csv", default=None, help="Path to results/trades.csv")
    ap.add_argument("--signals-parquet", default=None, help="Path to signals/signals.parquet")

    ap.add_argument(
        "--target",
        default="pnl_pos",
        choices=["pnl_pos", "hit_tp", "mfe_k"],
        help="Meta target choice",
    )
    ap.add_argument(
        "--r-threshold",
        type=float,
        default=0.0,
        help="Threshold for pnl_pos target (R units)",
    )
    ap.add_argument(
        "--mfe-k",
        type=float,
        default=2.0,
        help="k for mfe_k target: label=1 if mfe_over_atr >= k",
    )

    ap.add_argument("--blocks", type=int, default=12, help="CSCV blocks")
    ap.add_argument("--k-test", type=int, default=3, help="Blocks in test")
    ap.add_argument("--embargo", type=int, default=1, help="Embargo blocks")
    ap.add_argument("--max-splits", type=int, default=30, help="Max combinations")
    ap.add_argument(
        "--pstar",
        type=float,
        default=None,
        help="Optional deployment threshold to store in pstar.txt",
    )
    ap.add_argument("--outdir", default=None, help="Output folder (default: results/meta)")

    # NEW: schema inspection mode
    ap.add_argument(
        "--schema-only",
        action="store_true",
        help="Only build meta dataset and print schema stats; skip training.",
    )
    ap.add_argument(
        "--schema-max-rows",
        type=int,
        default=200_000,
        help="Limit rows used for schema stats (most recent rows kept).",
    )

    args = ap.parse_args()

    if args.schema_only:
        describe_meta_schema(
            trades_csv=args.trades_csv,
            signals_parquet=args.signals_parquet,
            target=args.target,
            r_threshold=args.r_threshold,
            mfe_k=args.mfe_k,
            max_rows=args.schema_max_rows,
            outdir=args.outdir,
        )
        return

    run_meta(
        trades_csv=args.trades_csv,
        signals_parquet=args.signals_parquet,
        target=args.target,
        r_threshold=args.r_threshold,
        mfe_k=args.mfe_k,
        blocks=args.blocks,
        k_test=args.k_test,
        embargo=args.embargo,
        max_splits=args.max_splits,
        outdir=args.outdir,
        pstar=args.pstar,
    )


if __name__ == "__main__":
    main()
