# bt_meta_online.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Iterable

import numpy as np
import pandas as pd

import config as cfg
from winprob_loader import WinProbScorer

# Global, single-load scorer
_SCORER: Optional[WinProbScorer] = None
_DIAG_ONCE: bool = False


def _model_dir() -> Path:
    p = getattr(cfg, "META_MODEL_DIR", "results/meta_export")
    return Path(p).expanduser().resolve()


def _get_scorer() -> Optional[WinProbScorer]:
    """
    Load scorer once. Keep behavior explicit:
      - If load fails or scorer isn't loaded, return None (and print once).
    """
    global _SCORER, _DIAG_ONCE

    if _SCORER is not None:
        if getattr(_SCORER, "is_loaded", False):
            return _SCORER
        return None

    try:
        sc = WinProbScorer(_model_dir())
        _SCORER = sc
        if not getattr(sc, "is_loaded", False):
            if not _DIAG_ONCE:
                print("[bt_meta_online] WinProbScorer not loaded; meta_p missing.", flush=True)
                _DIAG_ONCE = True
            return None
        return sc
    except Exception as e:
        if not _DIAG_ONCE:
            print(f"[bt_meta_online] WinProbScorer load failed: {e!r}", flush=True)
            _DIAG_ONCE = True
        _SCORER = None
        return None


def _as_dict(row: Any) -> Dict[str, Any]:
    """
    Accept pd.Series / dict-like / object with __dict__.
    """
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "to_dict"):
        try:
            return dict(row.to_dict())  # pd.Series
        except Exception:
            pass
    # fallback: try mapping protocol
    try:
        return dict(row)
    except Exception:
        pass
    # last resort
    try:
        return dict(getattr(row, "__dict__", {}))
    except Exception:
        return {}


def _normalize_ts(ts: Any) -> pd.Timestamp:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        # keep as NaT; schema checks may still pass if ts features not in manifest
        return pd.NaT
    return pd.Timestamp(t)


def _apply_aliases(d: Dict[str, Any]) -> None:
    """
    Deterministic aliasing between known schema variants.
    Only adds canonical keys if canonical is missing.
    """
    alias = {
        # BTC/ETH context variants (signals often have these prefixed)
        "btcusdt_funding_rate": "btc_funding_rate",
        "btcusdt_oi_z_7d": "btc_oi_z_7d",
        "btcusdt_vol_regime_level": "btc_vol_regime_level",
        "btcusdt_trend_slope": "btc_trend_slope",
        "ethusdt_funding_rate": "eth_funding_rate",
        "ethusdt_oi_z_7d": "eth_oi_z_7d",
        "ethusdt_vol_regime_level": "eth_vol_regime_level",
        "ethusdt_trend_slope": "eth_trend_slope",
    }
    for src, dst in alias.items():
        if dst not in d and src in d:
            d[dst] = d.get(src)


def _maybe_add_time_features(meta_row: Dict[str, Any], ts: pd.Timestamp, raw_cols: Iterable[str]) -> None:
    """
    Add ONLY those time features that are explicitly present in the training manifest.
    This avoids guessing names.
    """
    if ts is pd.NaT:
        return
    raw = set(raw_cols)

    # common candidates; only materialize if manifest wants them and they're absent
    candidates = {
        "hour": int(ts.hour),
        "hour_utc": int(ts.hour),
        "minute": int(ts.minute),
        "dayofweek": int(ts.dayofweek),
        "dow": int(ts.dayofweek),
        "weekday": int(ts.dayofweek),
        "day": int(ts.day),
        "month": int(ts.month),
        "is_weekend": int(1 if ts.dayofweek >= 5 else 0),
    }
    for k, v in candidates.items():
        if k in raw and k not in meta_row:
            meta_row[k] = v


def _ensure_manifest_complete(meta_row: Dict[str, Any], raw_cols: Iterable[str]) -> None:
    """
    Enforce strict schema if configured.
    Missing means "key absent", not "NaN value".
    """
    missing = [c for c in raw_cols if c not in meta_row]
    if not missing:
        return

    if bool(getattr(cfg, "META_STRICT_SCHEMA", False)):
        # fail fast; caller wants exact parity
        preview = missing[:40]
        print(f"[bt_meta_online] STRICT schema mismatch: missing {len(missing)} cols (first 40): {preview}", flush=True)
        raise RuntimeError(f"STRICT meta schema mismatch: missing {len(missing)} training columns")
    else:
        # permissive mode: fill with NaN so pipeline can run
        for c in missing:
            meta_row[c] = np.nan
        global _DIAG_ONCE
        if not _DIAG_ONCE:
            print(f"[bt_meta_online] non-strict: filled {len(missing)} missing cols with NaN", flush=True)
            _DIAG_ONCE = True


def diagnose_signal_schema(sig_row: Any, sig_ts: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Debug helper: returns {'missing': [...], 'present': N, 'needed': M}.
    Does NOT score.
    """
    sc = _get_scorer()
    if sc is None or not getattr(sc, "is_loaded", False):
        return {"error": "scorer_not_loaded"}

    raw_cols = list(getattr(sc, "raw_cols", []))
    ts = _normalize_ts(sig_ts)

    meta_row = _as_dict(sig_row)
    if extra:
        meta_row.update(dict(extra))
    _apply_aliases(meta_row)
    _maybe_add_time_features(meta_row, ts, raw_cols)

    missing = [c for c in raw_cols if c not in meta_row]
    return {"missing": missing, "present": len(meta_row), "needed": len(raw_cols)}


def score_signal_with_meta(
    sig_row: Any,
    sig_ts: Any,
    *,
    extra: Optional[Dict[str, Any]] = None,
    entry_override: Optional[float] = None,
) -> float:
    """
    Online scorer entrypoint.

    CRITICAL: merges `extra` into the model row BEFORE schema validation.
    This is what allows backtester to supply the exact training features.
    """
    global _DIAG_ONCE

    sc = _get_scorer()
    if sc is None or not getattr(sc, "is_loaded", False):
        if not _DIAG_ONCE:
            print("[bt_meta_online] WinProbScorer not loaded; meta_p missing.", flush=True)
            _DIAG_ONCE = True
        return float("nan")

    raw_cols = list(getattr(sc, "raw_cols", []))
    ts = _normalize_ts(sig_ts)

    meta_row = _as_dict(sig_row)

    # merge extras FIRST (this is the parity fix)
    if extra:
        meta_row.update(dict(extra))

    # override effective entry if provided (sim/live parity)
    if entry_override is not None:
        meta_row["entry"] = float(entry_override)

    _apply_aliases(meta_row)
    _maybe_add_time_features(meta_row, ts, raw_cols)

    # hard schema enforcement
    _ensure_manifest_complete(meta_row, raw_cols)

    try:
        p = float(sc.score(meta_row))
    except Exception as e:
        if not _DIAG_ONCE:
            print(f"[bt_meta_online] scoring failed once: {e!r}; meta_p missing.", flush=True)
            _DIAG_ONCE = True
        return float("nan")

    if not np.isfinite(p):
        return float("nan")
    # clamp
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    return float(p)
