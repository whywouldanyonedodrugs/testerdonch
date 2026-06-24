# live/oi_funding.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from indicators import align_series_point_in_time

# 5-minute windows (training parity)
WIN_1H, WIN_4H, WIN_1D, WIN_3D, WIN_7D = 12, 48, 288, 864, 2016
FUNDING_PUBLISH_LAG = pd.Timedelta("5min")


# -------------------- TZ helpers --------------------
def _now_utc() -> pd.Timestamp:
    """UTC-aware 'now' (safe across pandas versions)."""
    return pd.Timestamp.now(tz="UTC")

def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a UTC-aware Timestamp whether input was naive or tz-aware."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")   # naive → localize
    return ts.tz_convert("UTC")        # aware → convert


# -------------------- parsing helpers --------------------
def _as_df(items: list[dict], ts_key: str, val_key: str) -> pd.DataFrame:
    """
    Accept a list of dicts or list-of-lists and return a DataFrame indexed by UTC ms,
    with a single float column named `val_key`.
    """
    if not items:
        return pd.DataFrame(columns=[val_key])

    # list of dicts
    if isinstance(items[0], dict):
        df = pd.DataFrame(items)
        # normalize timestamp key
        cand_ts = [ts_key, "timestamp", "time", "fundingRateTimestamp"]
        ts_key = next((k for k in cand_ts if k in df.columns), ts_key)
        # normalize value key (OI or funding aliases)
        if val_key not in df.columns:
            for k in ("openInterest", "open_interest", "value", "openInterestValue",
                      "fundingRate", "rate", "funding_rate"):
                if k in df.columns:
                    val_key = k
                    break
        df = df[[ts_key, val_key]].copy()
        df[ts_key] = pd.to_numeric(df[ts_key], errors="coerce").astype("Int64")
        df[val_key] = pd.to_numeric(df[val_key], errors="coerce")
    else:
        # list of [timestamp, value, ...]
        df = pd.DataFrame(items, columns=[ts_key, val_key])
        df[ts_key] = pd.to_numeric(df[ts_key], errors="coerce").astype("Int64")
        df[val_key] = pd.to_numeric(df[val_key], errors="coerce")

    df.dropna(subset=[ts_key], inplace=True)
    # Create a tz-aware UTC index directly (no later tz_localize on tz-aware values!)
    idx = pd.to_datetime(df[ts_key].astype("int64"), unit="ms", utc=True)
    df = df.drop(columns=[ts_key]).set_index(idx).sort_index()
    return df


# -------------------- main fetch/alignment --------------------
async def fetch_series_5m(exchange, symbol: str, lookback_oi_days: int = 7, lookback_fr_days: int = 7) -> Tuple[pd.Series, pd.Series]:
    """
    Returns sparse 5m-grid-aligned OI and funding series.
    Point-in-time propagation is handled later when features are computed.
    """
    # Pull raw histories (helpers are on ExchangeProxy)
    oi_hist = await exchange.fetch_open_interest_history_5m(symbol, lookback_days=lookback_oi_days)
    fr_hist = await exchange.fetch_funding_rate_history(symbol, lookback_days=lookback_fr_days)

    oi_df = _as_df(oi_hist, "timestamp", "openInterest")
    fr_df = _as_df(fr_hist, "timestamp", "fundingRate")

    # Build a 5m grid covering the union of inputs (or a minimal window if empty)
    if not oi_df.empty:
        start = _ensure_utc(oi_df.index.min())
    elif not fr_df.empty:
        start = _ensure_utc(fr_df.index.min())
    else:
        start = _now_utc() - pd.Timedelta(days=max(lookback_oi_days, lookback_fr_days))

    end = _now_utc()

    # Important: start/end are already tz-aware UTC → don't pass tz= again here
    idx5 = pd.date_range(start=start.floor("5min"), end=end.floor("5min"), freq="5min")

    # Reindex to the 5m grid
    oi5 = oi_df.reindex(idx5)["openInterest"] if "openInterest" in oi_df.columns else pd.Series(index=idx5, dtype=float)
    fr5 = fr_df.reindex(idx5)["fundingRate"] if "fundingRate" in fr_df.columns else pd.Series(index=idx5, dtype=float)

    return oi5, fr5


def compute_oi_funding_feature_panel(
    df5: pd.DataFrame,
    oi5: pd.Series,
    fr5: pd.Series,
    *,
    funding_publish_lag: pd.Timedelta = FUNDING_PUBLISH_LAG,
) -> pd.DataFrame:
    """
    Compute the OI/Funding feature panel on `df5.index`.

    Contract:
    - OI is available from its timestamp forward.
    - Funding is treated as a discrete published event and is only available
      from `timestamp + funding_publish_lag` forward.
    - Nothing is prefilled before the first observed print.
    """
    base = df5.copy()
    if not isinstance(base.index, pd.DatetimeIndex):
        raise ValueError("df5 must be indexed by timestamps")
    if base.index.tz is None:
        base.index = base.index.tz_localize("UTC")
    else:
        base.index = base.index.tz_convert("UTC")
    base = base.sort_index()

    oi = align_series_point_in_time(base.index, oi5.astype(float))
    fr = align_series_point_in_time(base.index, fr5.astype(float), availability_lag=funding_publish_lag)

    close = base["close"].astype(float)
    volume = base.get("volume", pd.Series(index=base.index, dtype=float)).astype(float)

    # 1-2) levels
    oi_level        = oi
    oi_notional_est = oi * close

    # 3-5) pct changes (no implicit fill)
    oi_pct_1h = oi.pct_change(WIN_1H, fill_method=None)
    oi_pct_4h = oi.pct_change(WIN_4H, fill_method=None)
    oi_pct_1d = oi.pct_change(WIN_1D, fill_method=None)

    # 6) OI z-score (7d)
    oi_mean_7d = oi.rolling(WIN_7D, min_periods=WIN_1D).mean()
    oi_std_7d  = oi.rolling(WIN_7D, min_periods=WIN_1D).std()
    oi_z_7d    = (oi - oi_mean_7d) / (oi_std_7d + 1e-12)

    # 7) ΔOI normalized by recent turnover (1h)
    vol_1h = volume.rolling(WIN_1H).sum()
    oi_chg_norm_vol_1h = (oi - oi.shift(WIN_1H)) / (vol_1h + 1e-9)

    # 8) OI–price interaction
    ret_1h          = close.pct_change(WIN_1H, fill_method=None)
    oi_price_div_1h = np.sign(ret_1h) * oi_pct_1h

    # 9-12) Funding transforms
    funding_rate       = fr
    funding_abs        = fr.abs()
    fr_mean_7d         = fr.rolling(WIN_7D, min_periods=WIN_1D).mean()
    fr_std_7d          = fr.rolling(WIN_7D, min_periods=WIN_1D).std()
    funding_z_7d       = (fr - fr_mean_7d) / (fr_std_7d + 1e-12)
    funding_rollsum_3d = fr.rolling(WIN_3D, min_periods=WIN_1D).sum()

    # 13) Interaction
    funding_oi_div = funding_z_7d * oi_z_7d

    # Assemble last-bar snapshot
    panel = pd.DataFrame(
        {
            "oi_level": oi_level,
            "oi_notional_est": oi_notional_est,
            "oi_pct_1h": oi_pct_1h,
            "oi_pct_4h": oi_pct_4h,
            "oi_pct_1d": oi_pct_1d,
            "oi_z_7d": oi_z_7d,
            "oi_chg_norm_vol_1h": oi_chg_norm_vol_1h,
            "oi_price_div_1h": oi_price_div_1h,
            "funding_rate": funding_rate,
            "funding_abs": funding_abs,
            "funding_z_7d": funding_z_7d,
            "funding_rollsum_3d": funding_rollsum_3d,
            "funding_oi_div": funding_oi_div,
        },
        index=base.index,
    )
    panel["crowded_long"] = ((panel["oi_z_7d"] >= 1.0) & (panel["funding_z_7d"] >= 1.0)).astype(float)
    panel["crowded_short"] = ((panel["oi_z_7d"] >= 1.0) & (panel["funding_z_7d"] <= -1.0)).astype(float)
    panel["crowd_side"] = 0.0
    panel.loc[panel["crowded_long"] == 1.0, "crowd_side"] = 1.0
    panel.loc[panel["crowded_short"] == 1.0, "crowd_side"] = -1.0
    panel["est_leverage"] = oi_notional_est / (oi_notional_est.rolling(WIN_1D, min_periods=WIN_1H).mean() + 1e-9)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel.index.name = "timestamp"
    return panel


# -------------------- feature computation (13 features) --------------------
def compute_oi_funding_features(df5: pd.DataFrame, oi5: pd.Series, fr5: pd.Series, *, allow_nans: bool = True) -> Dict[str, float]:
    """
    Compute the OI+Funding feature snapshot for the **last** bar of df5.
    If allow_nans=False, replace missing with 0.0 to satisfy strict parity.
    """
    panel = compute_oi_funding_feature_panel(df5, oi5, fr5)
    fields = {
        "oi_level":            float(panel["oi_level"].iloc[-1]) if len(panel) else np.nan,
        "oi_notional_est":     float(panel["oi_notional_est"].iloc[-1]) if len(panel) else np.nan,
        "oi_pct_1h":           float(panel["oi_pct_1h"].iloc[-1]) if len(panel) else np.nan,
        "oi_pct_4h":           float(panel["oi_pct_4h"].iloc[-1]) if len(panel) else np.nan,
        "oi_pct_1d":           float(panel["oi_pct_1d"].iloc[-1]) if len(panel) else np.nan,
        "oi_z_7d":             float(panel["oi_z_7d"].iloc[-1]) if len(panel) else np.nan,
        "oi_chg_norm_vol_1h":  float(panel["oi_chg_norm_vol_1h"].iloc[-1]) if len(panel) else np.nan,
        "oi_price_div_1h":     float(panel["oi_price_div_1h"].iloc[-1]) if len(panel) else np.nan,
        "funding_rate":        float(panel["funding_rate"].iloc[-1]) if len(panel) else np.nan,
        "funding_abs":         float(panel["funding_abs"].iloc[-1]) if len(panel) else np.nan,
        "funding_z_7d":        float(panel["funding_z_7d"].iloc[-1]) if len(panel) else np.nan,
        "funding_rollsum_3d":  float(panel["funding_rollsum_3d"].iloc[-1]) if len(panel) else np.nan,
        "funding_oi_div":      float(panel["funding_oi_div"].iloc[-1]) if len(panel) else np.nan,
        "crowded_long":        float(panel["crowded_long"].iloc[-1]) if len(panel) else np.nan,
        "crowded_short":       float(panel["crowded_short"].iloc[-1]) if len(panel) else np.nan,
        "crowd_side":          float(panel["crowd_side"].iloc[-1]) if len(panel) else np.nan,
        "est_leverage":        float(panel["est_leverage"].iloc[-1]) if len(panel) else np.nan,
    }

    if not allow_nans:
        for k, v in list(fields.items()):
            if v is None or not np.isfinite(v):
                fields[k] = 0.0

    return fields
