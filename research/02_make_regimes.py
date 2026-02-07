#!/usr/bin/env python3
"""
research/02_make_regimes.py

Step 2: Build a regime library + regime-set codes from trades.clean.csv (entry-known fields only).

Outputs (in --outdir):
  - regimes.parquet
  - regimes_report.json
  - regime_state_counts.csv
  - regime_set_counts.csv
  - regime_sets.yaml

Regime columns produced (where inputs exist):
  Anchors:
    - regime_code_1d                  (existing; int)
    - markov_state_4h                 (existing; binary)
    - regime_up                       (existing; binary)
    - trend_regime_code_1d            (derived from trend_regime_1d if present; else from regime_code_1d)
    - vol_regime_code_1d              (derived from vol_regime_1d/vol_prob_low_1d if present; else from regime_code_1d)

  Derivatives positioning:
    - funding_regime_code             (-1 NEG, 0 NEU, 1 POS) with neutral band based on abs(funding_rate) quantile
    - oi_regime_code                  (-1 DOWN, 0 NORM, 1 UP) from oi_z_7d (or oi_pct_1d fallback) quantiles
    - crowd_side                      (existing; -1/0/1 if present)

  BTC risk regime:
    - btc_trend_up                    (0/1 from btc_trend_slope > 0)
    - btc_vol_high                    (0/1 from btc_vol_regime_level >= hi threshold)
    - btc_risk_regime_code            (0..3 = DOWN_LOW, DOWN_HIGH, UP_LOW, UP_HIGH)

  Setup-quality regimes:
    - freshness_code                  (0/1/2 from days_since_prev_break quantiles; 0=fresh)
    - compression_code                (0/1/2 from consolidation_range_atr quantiles; 0=tight)
    - volimpulse_code                 (0/1/2 from vol_mult quantiles; 2=high)

  Product slice:
    - risk_on                          = 1[(regime_up==1) & (btc_trend_up==1) & (btc_vol_high==0)]
    - risk_on_1                        = Explicit alias of risk_on (standardized column for scope RISK_ON_1)

Regime sets produced (code columns):
  - S1_regime_code_1d                 (0..3)
  - S2_markov_x_vol1d                 (0..3) = markov_state_4h x vol_regime_code_1d
  - S3_funding_x_oi                   (0..8) = funding_regime_code x oi_regime_code (3x3)
  - S4_crowd_x_trend1d                (0..5) = crowd_side x trend_regime_code_1d (3x2)
  - S5_btcRisk_x_regimeUp             (0..7) = btc_risk_regime_code x regime_up (4x2)
  - S6_fresh_x_compress               (0..8) = freshness_code x compression_code (3x3)

Notes:
  - Quantile thresholds are estimated from a bounded sample per column (Pass 1).
  - Everything is computed from entry-time known columns; no leakage from outcomes.

"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError(
        "pyarrow is required for chunked parquet writing. "
        "It was used successfully in 01_make_targets.py, so it should be installed. "
        f"Import error: {e}"
    )


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_cols(all_cols: List[str], candidates: List[str]) -> List[str]:
    s = set(all_cols)
    return [c for c in candidates if c in s]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _int8_nullable(x: pd.Series) -> pd.Series:
    # Convert boolean/0-1-like to pandas nullable Int8 where possible
    if pd.api.types.is_bool_dtype(x):
        return x.astype("Int8")
    return _to_num(x).round().astype("Int8")


def _int16_nullable(x: pd.Series) -> pd.Series:
    return _to_num(x).round().astype("Int16")


def _sample_values_from_series(
    s: pd.Series,
    max_keep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Keep up to max_keep values from series using chunk-level random subsampling.
    This is not exact reservoir sampling, but good enough for quantile cutpoints.
    """
    v = _to_num(s)
    v = v[np.isfinite(v)]
    if v.empty:
        return np.array([], dtype=np.float64)

    arr = v.to_numpy(dtype=np.float64, copy=False)
    if arr.size <= max_keep:
        return arr

    idx = rng.choice(arr.size, size=max_keep, replace=False)
    return arr[idx]


def _quantiles(arr: np.ndarray, qs: List[float]) -> Dict[str, Optional[float]]:
    if arr is None or arr.size == 0:
        return {f"q{int(q*100)}": None for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q*100)}"] = float(np.quantile(arr, q))
    return out


def _value_counts_series(s: pd.Series) -> Dict[str, int]:
    vc = s.value_counts(dropna=False)
    out: Dict[str, int] = {}
    for k, v in vc.items():
        if pd.isna(k):
            out["<NA>"] = int(v)
        else:
            out[str(k)] = int(v)
    return out


def _write_yaml(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------
# Column candidates
# ----------------------------

SAMPLE_COLS = [
    "funding_rate",
    "oi_z_7d",
    "oi_pct_1d",
    "btc_vol_regime_level",
    "days_since_prev_break",
    "consolidation_range_atr",
    "vol_mult",
]

PASS2_REQUIRED_COLS = [
    "trade_id",
    "symbol",
    "entry_ts",
    # Anchors
    "regime_code_1d",
    "trend_regime_1d",
    "vol_regime_1d",
    "vol_prob_low_1d",
    "markov_state_4h",
    "regime_up",
    # Positioning
    "funding_rate",
    "oi_z_7d",
    "oi_pct_1d",
    "crowd_side",
    # BTC context
    "btc_trend_slope",
    "btc_vol_regime_level",
    # Setup-quality
    "days_since_prev_break",
    "consolidation_range_atr",
    "vol_mult",
]


# ----------------------------
# Discretization / regime builders
# ----------------------------

def _build_trend_vol_codes(chunk: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    trend_regime_code_1d: 1 bull, 0 bear, NA unknown
    vol_regime_code_1d  : 0 low vol, 1 high vol, NA unknown
    Prefer explicit columns, fallback to regime_code_1d mapping:
      regime_code_1d: 0 bear high, 1 bear low, 2 bull high, 3 bull low
    """
    idx = chunk.index
    trend_code = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)
    vol_code = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)

    # Trend
    if "trend_regime_1d" in chunk.columns:
        tr = chunk["trend_regime_1d"].astype(str).str.upper()
        trend_code = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)
        trend_code = trend_code.mask(tr.str.contains("BULL", regex=False), 1)
        trend_code = trend_code.mask(tr.str.contains("BEAR", regex=False), 0)

    # Vol
    if "vol_regime_1d" in chunk.columns:
        vr = chunk["vol_regime_1d"].astype(str).str.upper()
        vol_code = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)
        vol_code = vol_code.mask(vr.str.contains("LOW", regex=False), 0)
        vol_code = vol_code.mask(vr.str.contains("HIGH", regex=False), 1)

    # Fallback to vol_prob_low_1d
    if (vol_code.isna().all()) and ("vol_prob_low_1d" in chunk.columns):
        vpl = _to_num(chunk["vol_prob_low_1d"])
        # Low vol if prob_low >= 0.5
        vol_code = (vpl < 0.5).where(vpl.notna(), other=pd.NA).astype("Int8")  # 1=high,0=low

    # Fallback to regime_code_1d (both)
    if "regime_code_1d" in chunk.columns:
        rc = _to_num(chunk["regime_code_1d"]).astype("Int16")
        # trend: bear for 0/1, bull for 2/3
        if trend_code.isna().all():
            trend_code = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)
            trend_code = trend_code.mask(rc.isin([2, 3]), 1)
            trend_code = trend_code.mask(rc.isin([0, 1]), 0)
        # vol: high for 0/2, low for 1/3
        if vol_code.isna().all():
            vol_code = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)
            vol_code = vol_code.mask(rc.isin([0, 2]), 1)  # high
            vol_code = vol_code.mask(rc.isin([1, 3]), 0)  # low

    return trend_code, vol_code


def _funding_regime(funding_rate: pd.Series, eps: float) -> pd.Series:
    """
    -1 NEG, 0 NEU, 1 POS using neutral band [-eps, +eps].
    """
    fr = _to_num(funding_rate)
    out = pd.Series(pd.array([pd.NA] * len(fr), dtype="Int8"), index=fr.index)
    out = out.mask(fr <= -eps, -1)
    out = out.mask(fr >= eps, 1)
    out = out.mask((fr > -eps) & (fr < eps), 0)
    return out


def _oi_regime_from_continuous(x: pd.Series, q33: float, q66: float) -> pd.Series:
    """
    -1 DOWN (<=q33), 0 NORM, 1 UP (>=q66)
    """
    v = _to_num(x)
    out = pd.Series(pd.array([pd.NA] * len(v), dtype="Int8"), index=v.index)
    out = out.mask(v <= q33, -1)
    out = out.mask(v >= q66, 1)
    out = out.mask((v > q33) & (v < q66), 0)
    return out


def _bucket_terciles(x: pd.Series, q33: float, q66: float, low_is_0: bool = True) -> pd.Series:
    """
    0/1/2 buckets by terciles:
      0: <=q33
      1: (q33,q66)
      2: >=q66
    If low_is_0 is True, lower values -> lower code.
    """
    v = _to_num(x)
    out = pd.Series(pd.array([pd.NA] * len(v), dtype="Int8"), index=v.index)
    out = out.mask(v <= q33, 0)
    out = out.mask(v >= q66, 2)
    out = out.mask((v > q33) & (v < q66), 1)
    return out


def _btc_risk_regime(
    btc_trend_slope: pd.Series,
    btc_vol_regime_level: pd.Series,
    vol_hi: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    btc_trend_up: 1 if slope>0 else 0 (NA if missing)
    btc_vol_high: 1 if vol_level>=vol_hi else 0 (NA if missing)
    btc_risk_regime_code: 0..3:
      0 DOWN_LOW, 1 DOWN_HIGH, 2 UP_LOW, 3 UP_HIGH
    """
    slope = _to_num(btc_trend_slope)
    vol = _to_num(btc_vol_regime_level)

    trend_up = (slope > 0).where(slope.notna(), other=pd.NA).astype("Int8")
    vol_high = (vol >= vol_hi).where(vol.notna(), other=pd.NA).astype("Int8")

    # combine: trend_up*2 + vol_high
    rc = pd.Series(pd.array([pd.NA] * len(slope), dtype="Int8"), index=slope.index)
    mask = trend_up.notna() & vol_high.notna()
    rc = rc.mask(mask, (trend_up[mask].astype("int8") * 2 + vol_high[mask].astype("int8")).astype("int8"))
    return trend_up, vol_high, rc


def _risk_on(regime_up: pd.Series, btc_trend_up: pd.Series, btc_vol_high: pd.Series) -> pd.Series:
    ru = _int8_nullable(regime_up)
    out = ((ru == 1) & (btc_trend_up == 1) & (btc_vol_high == 0)).astype("int8")
    # If any component is NA, set NA
    na_mask = ru.isna() | btc_trend_up.isna() | btc_vol_high.isna()
    out = pd.Series(out, index=ru.index).astype("Int8")
    out = out.where(~na_mask, other=pd.NA)
    return out


def _set_code_2x2(a01: pd.Series, b01: pd.Series) -> pd.Series:
    """
    a01 in {0,1}, b01 in {0,1} -> code = a*2 + b in {0..3}
    NA if any NA.
    """
    out = pd.Series(pd.array([pd.NA] * len(a01), dtype="Int16"), index=a01.index)
    mask = a01.notna() & b01.notna()
    out = out.mask(mask, (a01[mask].astype("int16") * 2 + b01[mask].astype("int16")).astype("int16"))
    return out


def _set_code_3x3(a_m101: pd.Series, b_m101: pd.Series) -> pd.Series:
    """
    a,b in {-1,0,1} -> map to {0,1,2} via +1
    code = (a+1)*3 + (b+1) in {0..8}
    """
    out = pd.Series(pd.array([pd.NA] * len(a_m101), dtype="Int16"), index=a_m101.index)
    mask = a_m101.notna() & b_m101.notna()
    a3 = (a_m101[mask].astype("int16") + 1)
    b3 = (b_m101[mask].astype("int16") + 1)
    out = out.mask(mask, (a3 * 3 + b3).astype("int16"))
    return out


def _set_code_3x2(a_m101: pd.Series, b01: pd.Series) -> pd.Series:
    """
    a in {-1,0,1} -> a3=a+1 in {0,1,2}
    b in {0,1}
    code = a3*2 + b in {0..5}
    """
    out = pd.Series(pd.array([pd.NA] * len(a_m101), dtype="Int16"), index=a_m101.index)
    mask = a_m101.notna() & b01.notna()
    a3 = (a_m101[mask].astype("int16") + 1)
    b2 = b01[mask].astype("int16")
    out = out.mask(mask, (a3 * 2 + b2).astype("int16"))
    return out


def _set_code_4x2(a03: pd.Series, b01: pd.Series) -> pd.Series:
    """
    a in {0..3}, b in {0,1} -> code = a*2 + b in {0..7}
    """
    out = pd.Series(pd.array([pd.NA] * len(a03), dtype="Int16"), index=a03.index)
    mask = a03.notna() & b01.notna()
    out = out.mask(mask, (a03[mask].astype("int16") * 2 + b01[mask].astype("int16")).astype("int16"))
    return out


def _set_code_3x3_from_terciles(a02: pd.Series, b02: pd.Series) -> pd.Series:
    """
    a,b in {0,1,2} -> code = a*3 + b in {0..8}
    """
    out = pd.Series(pd.array([pd.NA] * len(a02), dtype="Int16"), index=a02.index)
    mask = a02.notna() & b02.notna()
    out = out.mask(mask, (a02[mask].astype("int16") * 3 + b02[mask].astype("int16")).astype("int16"))
    return out


# ----------------------------
# Main processing
# ----------------------------

def pass1_estimate_cutpoints(
    infile: str,
    cols_present: List[str],
    chunksize: int,
    sample_per_col: int,
    seed: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Pass 1: collect bounded random samples for selected columns, compute quantile cutpoints.
    """
    rng = np.random.default_rng(seed)

    sample_cols = _safe_cols(cols_present, SAMPLE_COLS)
    if not sample_cols:
        return {}

    # Accumulate samples per col (bounded by sample_per_col)
    samples: Dict[str, np.ndarray] = {c: np.array([], dtype=np.float64) for c in sample_cols}

    reader = pd.read_csv(infile, chunksize=chunksize, low_memory=False, usecols=sample_cols)

    for i, chunk in enumerate(reader):
        for c in sample_cols:
            cur = samples[c]
            remaining = max(0, sample_per_col - cur.size)
            if remaining <= 0:
                continue
            take = min(remaining, 5000)  # per-chunk cap
            arr = _sample_values_from_series(chunk[c], max_keep=take, rng=rng)
            if arr.size > 0:
                samples[c] = np.concatenate([cur, arr], axis=0)

        if (i + 1) % 20 == 0:
            done_cols = sum(int(samples[c].size >= sample_per_col) for c in sample_cols)
            print(f"[02_make_regimes pass1] chunks={i+1} samples_done_cols={done_cols}/{len(sample_cols)}", flush=True)

        if all(samples[c].size >= sample_per_col for c in sample_cols):
            break

    # Compute cutpoints
    cutpoints: Dict[str, Dict[str, Optional[float]]] = {}

    # funding: eps from abs funding q20
    if "funding_rate" in samples and samples["funding_rate"].size > 0:
        abs_fr = np.abs(samples["funding_rate"])
        cutpoints["funding_rate_abs"] = _quantiles(abs_fr, [0.20, 0.50, 0.80])
        # also store raw funding quantiles (optional)
        cutpoints["funding_rate"] = _quantiles(samples["funding_rate"], [0.10, 0.50, 0.90])

    # oi: prefer oi_z_7d quantiles; fallback to oi_pct_1d
    if "oi_z_7d" in samples and samples["oi_z_7d"].size > 0:
        cutpoints["oi_z_7d"] = _quantiles(samples["oi_z_7d"], [0.33, 0.66])
    if "oi_pct_1d" in samples and samples["oi_pct_1d"].size > 0:
        cutpoints["oi_pct_1d"] = _quantiles(samples["oi_pct_1d"], [0.33, 0.66])

    if "btc_vol_regime_level" in samples and samples["btc_vol_regime_level"].size > 0:
        cutpoints["btc_vol_regime_level"] = _quantiles(samples["btc_vol_regime_level"], [0.50, 0.66, 0.80])

    if "days_since_prev_break" in samples and samples["days_since_prev_break"].size > 0:
        cutpoints["days_since_prev_break"] = _quantiles(samples["days_since_prev_break"], [0.33, 0.66])

    if "consolidation_range_atr" in samples and samples["consolidation_range_atr"].size > 0:
        cutpoints["consolidation_range_atr"] = _quantiles(samples["consolidation_range_atr"], [0.33, 0.66])

    if "vol_mult" in samples and samples["vol_mult"].size > 0:
        cutpoints["vol_mult"] = _quantiles(samples["vol_mult"], [0.33, 0.66])

    return cutpoints


def pass2_build_regimes(
    infile: str,
    outdir: str,
    outfile: str,
    cols_present: List[str],
    chunksize: int,
    cutpoints: Dict[str, Dict[str, Optional[float]]],
) -> None:
    """
    Pass 2: chunked generation of regime features and regime-set codes into Parquet.
    Also accumulates state counts.
    """
    out_path = os.path.join(outdir, outfile)
    report_path = os.path.join(outdir, "regimes_report.json")
    state_counts_path = os.path.join(outdir, "regime_state_counts.csv")
    set_counts_path = os.path.join(outdir, "regime_set_counts.csv")
    sets_yaml_path = os.path.join(outdir, "regime_sets.yaml")

    # Determine thresholds with safe fallbacks
    eps = 0.0
    if "funding_rate_abs" in cutpoints and cutpoints["funding_rate_abs"].get("q20") is not None:
        eps = float(cutpoints["funding_rate_abs"]["q20"])
    # If eps is too tiny (or 0), enforce a small minimum to create a neutral band
    eps = max(eps, 1e-8)

    # oi quantiles
    oi_q33 = oi_q66 = None
    oi_source = None
    if "oi_z_7d" in cutpoints and cutpoints["oi_z_7d"].get("q33") is not None and cutpoints["oi_z_7d"].get("q66") is not None:
        oi_q33 = float(cutpoints["oi_z_7d"]["q33"])
        oi_q66 = float(cutpoints["oi_z_7d"]["q66"])
        oi_source = "oi_z_7d"
    elif "oi_pct_1d" in cutpoints and cutpoints["oi_pct_1d"].get("q33") is not None and cutpoints["oi_pct_1d"].get("q66") is not None:
        oi_q33 = float(cutpoints["oi_pct_1d"]["q33"])
        oi_q66 = float(cutpoints["oi_pct_1d"]["q66"])
        oi_source = "oi_pct_1d"

    # btc vol high threshold (use q66 if available; else median; else 1.0)
    btc_vol_hi = 1.0
    if "btc_vol_regime_level" in cutpoints:
        if cutpoints["btc_vol_regime_level"].get("q66") is not None:
            btc_vol_hi = float(cutpoints["btc_vol_regime_level"]["q66"])
        elif cutpoints["btc_vol_regime_level"].get("q50") is not None:
            btc_vol_hi = float(cutpoints["btc_vol_regime_level"]["q50"])

    # tercile cutpoints for setup-quality
    fresh_q33 = fresh_q66 = None
    if "days_since_prev_break" in cutpoints:
        fresh_q33 = cutpoints["days_since_prev_break"].get("q33")
        fresh_q66 = cutpoints["days_since_prev_break"].get("q66")

    comp_q33 = comp_q66 = None
    if "consolidation_range_atr" in cutpoints:
        comp_q33 = cutpoints["consolidation_range_atr"].get("q33")
        comp_q66 = cutpoints["consolidation_range_atr"].get("q66")

    volm_q33 = volm_q66 = None
    if "vol_mult" in cutpoints:
        volm_q33 = cutpoints["vol_mult"].get("q33")
        volm_q66 = cutpoints["vol_mult"].get("q66")

    # Build usecols for pass2 (only those present)
    usecols = _safe_cols(cols_present, PASS2_REQUIRED_COLS)
    if "trade_id" not in usecols:
        raise RuntimeError("trade_id is required.")
    if "symbol" not in usecols:
        raise RuntimeError("symbol is required.")

    # State counts accumulators
    regime_cols_for_counts = [
        "regime_code_1d",
        "trend_regime_code_1d",
        "vol_regime_code_1d",
        "markov_state_4h",
        "regime_up",
        "funding_regime_code",
        "oi_regime_code",
        "crowd_side",
        "btc_trend_up",
        "btc_vol_high",
        "btc_risk_regime_code",
        "freshness_code",
        "compression_code",
        "volimpulse_code",
        "risk_on",
        "risk_on_1",
    ]
    set_cols_for_counts = [
        "S1_regime_code_1d",
        "S2_markov_x_vol1d",
        "S3_funding_x_oi",
        "S4_crowd_x_trend1d",
        "S5_btcRisk_x_regimeUp",
        "S6_fresh_x_compress",
    ]

    state_counts: Dict[str, Dict[str, int]] = {c: {} for c in regime_cols_for_counts}
    set_counts: Dict[str, Dict[str, int]] = {c: {} for c in set_cols_for_counts}

    # Chunked parquet writer
    writer: Optional[pq.ParquetWriter] = None

    reader = pd.read_csv(infile, chunksize=chunksize, low_memory=False, usecols=usecols)

    for i, chunk in enumerate(reader):
        if len(chunk) == 0:
            continue

        # Parse trade_id
        tid = pd.to_numeric(chunk["trade_id"], errors="coerce")
        chunk = chunk.loc[tid.notna()].copy()
        if len(chunk) == 0:
            continue
        chunk["trade_id"] = pd.to_numeric(chunk["trade_id"], errors="coerce").astype("int64")

        idx = chunk.index

        # Anchors
        regime_code_1d = _int16_nullable(chunk["regime_code_1d"]) if "regime_code_1d" in chunk.columns else pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int16"), index=idx)
        markov_state_4h = _int8_nullable(chunk["markov_state_4h"]) if "markov_state_4h" in chunk.columns else pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        regime_up = _int8_nullable(chunk["regime_up"]) if "regime_up" in chunk.columns else pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)

        trend_code, vol_code = _build_trend_vol_codes(chunk)

        # Positioning
        funding_regime_code = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        if "funding_rate" in chunk.columns:
            funding_regime_code = _funding_regime(chunk["funding_rate"], eps=eps)

        oi_regime_code = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        if oi_q33 is not None and oi_q66 is not None:
            if oi_source == "oi_z_7d" and ("oi_z_7d" in chunk.columns):
                oi_regime_code = _oi_regime_from_continuous(chunk["oi_z_7d"], q33=float(oi_q33), q66=float(oi_q66))
            elif oi_source == "oi_pct_1d" and ("oi_pct_1d" in chunk.columns):
                oi_regime_code = _oi_regime_from_continuous(chunk["oi_pct_1d"], q33=float(oi_q33), q66=float(oi_q66))

        crowd_side = _int8_nullable(chunk["crowd_side"]) if "crowd_side" in chunk.columns else pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)

        # BTC risk regime
        btc_trend_up = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        btc_vol_high = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        btc_risk_regime_code = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        if ("btc_trend_slope" in chunk.columns) and ("btc_vol_regime_level" in chunk.columns):
            btc_trend_up, btc_vol_high, btc_risk_regime_code = _btc_risk_regime(
                chunk["btc_trend_slope"],
                chunk["btc_vol_regime_level"],
                vol_hi=float(btc_vol_hi),
            )

        # Setup-quality: terciles
        freshness_code = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        if ("days_since_prev_break" in chunk.columns) and (fresh_q33 is not None) and (fresh_q66 is not None):
            freshness_code = _bucket_terciles(chunk["days_since_prev_break"], q33=float(fresh_q33), q66=float(fresh_q66))

        compression_code = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        if ("consolidation_range_atr" in chunk.columns) and (comp_q33 is not None) and (comp_q66 is not None):
            compression_code = _bucket_terciles(chunk["consolidation_range_atr"], q33=float(comp_q33), q66=float(comp_q66))

        volimpulse_code = pd.Series(pd.array([pd.NA]*len(chunk), dtype="Int8"), index=idx)
        if ("vol_mult" in chunk.columns) and (volm_q33 is not None) and (volm_q66 is not None):
            volimpulse_code = _bucket_terciles(chunk["vol_mult"], q33=float(volm_q33), q66=float(volm_q66))

        # Product slice
        risk_on = _risk_on(regime_up=regime_up, btc_trend_up=btc_trend_up, btc_vol_high=btc_vol_high)

        # Regime sets
        S1 = regime_code_1d.astype("Int16")

        # S2: markov x vol (2x2)
        S2 = _set_code_2x2(markov_state_4h, vol_code)

        # S3: funding x oi (3x3)
        S3 = _set_code_3x3(funding_regime_code, oi_regime_code)

        # S4: crowd x trend (3x2)
        S4 = _set_code_3x2(crowd_side, trend_code)

        # S5: btc risk x regime_up (4x2)
        S5 = _set_code_4x2(btc_risk_regime_code, regime_up)

        # S6: freshness x compression (3x3)
        S6 = _set_code_3x3_from_terciles(freshness_code, compression_code)

        # Output frame
        out = pd.DataFrame(
            {
                "trade_id": chunk["trade_id"].astype("int64"),
                "symbol": chunk["symbol"].astype(str),
                "entry_ts": chunk["entry_ts"] if "entry_ts" in chunk.columns else np.nan,

                "regime_code_1d": regime_code_1d,
                "trend_regime_code_1d": trend_code,
                "vol_regime_code_1d": vol_code,
                "markov_state_4h": markov_state_4h,
                "regime_up": regime_up,

                "funding_regime_code": funding_regime_code,
                "oi_regime_code": oi_regime_code,
                "crowd_side": crowd_side,

                "btc_trend_up": btc_trend_up,
                "btc_vol_high": btc_vol_high,
                "btc_risk_regime_code": btc_risk_regime_code,

                "freshness_code": freshness_code,
                "compression_code": compression_code,
                "volimpulse_code": volimpulse_code,

                "risk_on": risk_on,

                "risk_on_1": risk_on,  # alias for standardized scope name


                "S1_regime_code_1d": S1,
                "S2_markov_x_vol1d": S2,
                "S3_funding_x_oi": S3,
                "S4_crowd_x_trend1d": S4,
                "S5_btcRisk_x_regimeUp": S5,
                "S6_fresh_x_compress": S6,
            }
        )

        # Update counts (regimes)
        for c in regime_cols_for_counts:
            if c not in out.columns:
                continue
            vc = _value_counts_series(out[c])
            acc = state_counts[c]
            for k, v in vc.items():
                acc[k] = acc.get(k, 0) + v

        # Update counts (sets)
        for c in set_cols_for_counts:
            if c not in out.columns:
                continue
            vc = _value_counts_series(out[c])
            acc = set_counts[c]
            for k, v in vc.items():
                acc[k] = acc.get(k, 0) + v

        # Write to parquet (chunked)
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)

        if (i + 1) % 10 == 0:
            print(f"[02_make_regimes pass2] processed chunks={i+1}", flush=True)

    if writer is not None:
        writer.close()

    # Write counts CSVs
    # regime_state_counts.csv: long format
    state_rows = []
    for col, d in state_counts.items():
        for state, cnt in d.items():
            state_rows.append({"regime_col": col, "state": state, "count": int(cnt)})
    state_df = pd.DataFrame(state_rows).sort_values(["regime_col", "count"], ascending=[True, False])
    state_df.to_csv(state_counts_path, index=False)

    # regime_set_counts.csv: long format
    set_rows = []
    for col, d in set_counts.items():
        for state, cnt in d.items():
            set_rows.append({"set_col": col, "code": state, "count": int(cnt)})
    set_df = pd.DataFrame(set_rows).sort_values(["set_col", "count"], ascending=[True, False])
    set_df.to_csv(set_counts_path, index=False)

    # Write regime_sets.yaml
    # (Plain YAML text; no dependency on PyYAML)
    yaml_text = f"""# regime_sets.yaml
# Generated by research/02_make_regimes.py
#
# Conventions:
# - Codes are integers; missing is <NA> in reports.
# - For set codes, see mapping notes under each set.

regime_columns:
  regime_code_1d:
    type: int
    mapping:
      0: BEAR_HIGH
      1: BEAR_LOW
      2: BULL_HIGH
      3: BULL_LOW
  trend_regime_code_1d:
    type: int
    mapping:
      0: BEAR
      1: BULL
  vol_regime_code_1d:
    type: int
    mapping:
      0: LOW_VOL
      1: HIGH_VOL
  funding_regime_code:
    type: int
    mapping:
      -1: NEG
       0: NEU
       1: POS
    params:
      neutral_eps_abs_funding: {eps}
  oi_regime_code:
    type: int
    mapping:
      -1: DOWN
       0: NORM
       1: UP
    params:
      source: {oi_source}
      q33: {oi_q33}
      q66: {oi_q66}
  btc_risk_regime_code:
    type: int
    mapping:
      0: DOWN_LOW
      1: DOWN_HIGH
      2: UP_LOW
      3: UP_HIGH
    params:
      btc_vol_hi: {btc_vol_hi}
  freshness_code:
    type: int
    mapping:
      0: FRESH
      1: MID
      2: STALE
    params:
      q33: {fresh_q33}
      q66: {fresh_q66}
  compression_code:
    type: int
    mapping:
      0: TIGHT
      1: MID
      2: WIDE
    params:
      q33: {comp_q33}
      q66: {comp_q66}

regime_sets:
  S1_regime_code_1d:
    columns: [regime_code_1d]
    code: "same as regime_code_1d"
  S2_markov_x_vol1d:
    columns: [markov_state_4h, vol_regime_code_1d]
    code: "markov_state_4h*2 + vol_regime_code_1d  => 0..3"
  S3_funding_x_oi:
    columns: [funding_regime_code, oi_regime_code]
    code: "(funding_regime_code+1)*3 + (oi_regime_code+1) => 0..8"
  S4_crowd_x_trend1d:
    columns: [crowd_side, trend_regime_code_1d]
    code: "(crowd_side+1)*2 + trend_regime_code_1d => 0..5"
  S5_btcRisk_x_regimeUp:
    columns: [btc_risk_regime_code, regime_up]
    code: "btc_risk_regime_code*2 + regime_up => 0..7"
  S6_fresh_x_compress:
    columns: [freshness_code, compression_code]
    code: "freshness_code*3 + compression_code => 0..8"

product_slice:
  risk_on:
    definition: "1[(regime_up==1) & (btc_trend_up==1) & (btc_vol_high==0)]"
"""
    _write_yaml(sets_yaml_path, yaml_text)

    # Write regimes_report.json
    report = {
        "infile": infile,
        "outfile": outfile,
        "thresholds": {
            "funding_neutral_eps": eps,
            "oi_source": oi_source,
            "oi_q33": oi_q33,
            "oi_q66": oi_q66,
            "btc_vol_hi": btc_vol_hi,
            "fresh_q33": fresh_q33,
            "fresh_q66": fresh_q66,
            "compression_q33": comp_q33,
            "compression_q66": comp_q66,
            "vol_mult_q33": volm_q33,
            "vol_mult_q66": volm_q66,
        },
        "cutpoints_raw": cutpoints,
        "columns_used_pass2": usecols,
        "columns_missing_from_pass2_candidates": sorted(set(PASS2_REQUIRED_COLS) - set(usecols)),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"[02_make_regimes] DONE. Wrote: {out_path}")
    print(f"[02_make_regimes] - regimes_report.json")
    print(f"[02_make_regimes] - regime_state_counts.csv")
    print(f"[02_make_regimes] - regime_set_counts.csv")
    print(f"[02_make_regimes] - regime_sets.yaml")


def process(
    infile: str,
    outdir: str,
    outfile: str,
    chunksize: int,
    sample_per_col: int,
    seed: int,
) -> None:
    _ensure_dir(outdir)

    header = pd.read_csv(infile, nrows=0)
    cols_present = list(header.columns)
    if "trade_id" not in cols_present:
        raise RuntimeError("trade_id missing from input.")
    if "symbol" not in cols_present:
        raise RuntimeError("symbol missing from input.")

    cutpoints = pass1_estimate_cutpoints(
        infile=infile,
        cols_present=cols_present,
        chunksize=chunksize,
        sample_per_col=sample_per_col,
        seed=seed,
    )

    pass2_build_regimes(
        infile=infile,
        outdir=outdir,
        outfile=outfile,
        cols_present=cols_present,
        chunksize=chunksize,
        cutpoints=cutpoints,
    )


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2: create regime library + regime sets from trades CSV.")
    p.add_argument("--infile", type=str, default="results/trades.clean.csv", help="Input trades CSV.")
    p.add_argument("--outdir", type=str, default="research_outputs/02_regimes", help="Output directory.")
    p.add_argument("--outfile", type=str, default="regimes.parquet", help="Output parquet filename (inside outdir).")
    p.add_argument("--chunksize", type=int, default=250_000, help="CSV chunk size.")
    p.add_argument("--sample-per-col", type=int, default=200_000, help="Max sample kept per continuous column for cutpoints.")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed for sampling.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if int(args.chunksize) < 10_000:
        raise ValueError("--chunksize too small; use at least 10,000.")

    process(
        infile=args.infile,
        outdir=args.outdir,
        outfile=args.outfile,
        chunksize=int(args.chunksize),
        sample_per_col=int(args.sample_per_col),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()