# live/parity_utils.py
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from . import indicators as ta
from indicators import (
    align_series_point_in_time as _align_series_point_in_time,
    donchian_upper_days_no_lookahead as _donchian_upper_days_no_lookahead,
    resample_ohlcv as _canonical_resample_ohlcv,
)


def _norm_tf(tf: str) -> str:
    """
    Normalize timeframe strings to pandas-safe aliases.
    Prevents '15m' being interpreted as 15 months.
    """
    tf = str(tf).strip()
    mapping = {
        "1m": "1min", "3m": "3min", "5m": "5min",
        "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h",
        "1d": "1D", "1D": "1D",
    }
    if tf in mapping:
        return mapping[tf]

    if tf.endswith("m") and not tf.endswith("min"):
        return tf[:-1] + "min"

    if tf.lower().endswith("d"):
        return tf.upper()

    return tf


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Canonical close-labeled resampling delegated to top-level indicators.py.
    """
    if df is None or df.empty:
        return df
    return _canonical_resample_ohlcv(df, _norm_tf(timeframe))


def resample_ohlcv_robust(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Robust resampling (Label='right', Closed='right').
    Matches offline 'scout' usage for Regime detection and Donchian 'completed days' logic.
    """
    if df is None or df.empty:
        return df
    return _canonical_resample_ohlcv(df, _norm_tf(timeframe))


def map_to_left_index(target_index: pd.DatetimeIndex, source_series: pd.Series) -> pd.Series:
    """
    Canonical backward as-of alignment delegated to top-level indicators.py.
    """
    return _align_series_point_in_time(target_index, source_series)


def donchian_upper_days_no_lookahead(high_5m: pd.Series, n_days: int) -> pd.Series:
    """
    Daily Donchian upper on *completed* days only.
    Matches offline scout.py logic (Right/Right + dropna + shift).

    Returns a pd.Series indexed by high_5m.index (wrapping the numpy array from scout logic).
    """
    return _donchian_upper_days_no_lookahead(high_5m, n_days)


# =============================================================================
# META scope evaluation (pure helper; unit-testable)
# =============================================================================

def _to_float_or_none(v: Any) -> Optional[float]:
    """
    Best-effort parse to float. Returns None if missing/unparseable/non-finite.
    This is used as the internal primitive for "to_numeric(errors='coerce')".
    """
    if v is None:
        return None

    # pandas/np missing
    try:
        if isinstance(v, (float, np.floating)) and (not np.isfinite(float(v))):
            return None
    except Exception:
        pass

    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            x = float(s)
        except Exception:
            return None
        return float(x) if np.isfinite(float(x)) else None

    try:
        x = float(v)
    except Exception:
        return None

    return float(x) if np.isfinite(float(x)) else None


def eval_meta_scope(pstar_scope: Optional[str], meta_row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate meta scope with offline-parity semantics and fail-closed behavior.

    Offline semantics for decision.scope == "RISK_ON_1":
      - Use risk_on_1 if the key exists in the row dict.
      - Otherwise (only if the key is absent), fallback to risk_on.
      - Coerce with to_numeric(errors='coerce').fillna(0), then compare == 1.
      - If neither key exists at all, offline raises; live returns False and flags missing_cols.

    Supported scopes:
      - None / ""     : scope passes (True)
      - "RISK_ON_1"   : described above
      - any other     : False (fail-closed)

    Returns (scope_ok, info) where info includes raw inputs and the chosen source.
    """
    sc = (pstar_scope or "").strip()
    sc_u = sc.upper() if sc else ""

    has_risk_on_1 = ("risk_on_1" in meta_row)
    has_risk_on = ("risk_on" in meta_row)

    info: Dict[str, Any] = {
        "scope": sc_u or None,
        "risk_on_1_raw": meta_row.get("risk_on_1", None),
        "risk_on_raw": meta_row.get("risk_on", None),
        "risk_on_1_present": bool(has_risk_on_1),
        "risk_on_present": bool(has_risk_on),
        "scope_val": None,     # numeric value used after coercion/fillna(0)
        "scope_src": None,     # "risk_on_1" | "risk_on" | None
        "missing_cols": False,
    }

    if not sc_u:
        return True, info

    if sc_u != "RISK_ON_1":
        return False, info

    # Choose column by presence only (no row-wise fallback when present-but-NaN)
    if has_risk_on_1:
        raw = meta_row.get("risk_on_1", None)
        x = _to_float_or_none(raw)
        x = 0.0 if x is None else float(x)
        info["scope_val"] = x
        info["scope_src"] = "risk_on_1"
        return (x == 1.0), info

    if has_risk_on:
        raw = meta_row.get("risk_on", None)
        x = _to_float_or_none(raw)
        x = 0.0 if x is None else float(x)
        info["scope_val"] = x
        info["scope_src"] = "risk_on"
        return (x == 1.0), info

    info["missing_cols"] = True
    return False, info
