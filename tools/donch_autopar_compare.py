#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BTC_VOL_HI_FALLBACK = 0.753777980804443
CANONICAL_BUCKETS = ("no_signals", "schema_fail", "scope_fail", "below_pstar", "gate_fail", "other")
PIPELINE_BUCKETS = {"no_signals", "schema_fail"}
NO_EDGE_BUCKETS = {"scope_fail", "below_pstar", "gate_fail"}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cl = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in cl:
            return cl[c.lower()]
    return None


def _to_bool(x: object) -> Optional[bool]:
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "taken", "enter", "open"):
        return True
    if s in ("false", "0", "no", "n", "skipped", "skip", "reject"):
        return False
    return None


def _decision_to_int(v: object) -> Optional[int]:
    b = _to_bool(v)
    if b is None:
        return None
    return int(bool(b))


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _fmt(x: object, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        f = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(f):
        return ""
    return f"{f:.{nd}f}"


def _parse_num(v: object) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    return float(x)


def _parse_bool(v: object) -> Optional[bool]:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return _to_bool(v)


def _contains_any(txt: str, tokens: List[str]) -> bool:
    s = str(txt or "").strip().lower()
    if not s:
        return False
    return any(t in s for t in tokens)


def _normalize_reason(reason: object) -> str:
    s = str(reason or "").strip().lower()
    if s in ("meta_prob", "below_threshold", "below_prob_threshold"):
        return "below_pstar"
    if s in ("no_prob_gate", "strategy_fail", "regime_down", "regime_slope_down"):
        return "gate_fail"
    return s


def _btc_vol_hi() -> float:
    try:
        import config as cfg  # type: ignore

        return float(getattr(cfg, "BTC_VOL_HI", BTC_VOL_HI_FALLBACK))
    except Exception:
        return float(BTC_VOL_HI_FALLBACK)


def _norm_decisions(
    df: pd.DataFrame,
    *,
    side: str,
    bucket: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cols = list(df.columns)
    sym_col = _pick_col(cols, ["symbol", "sym", "market"])
    ts_col = _pick_col(cols, ["decision_ts", "ts_effective", "signal_ts", "timestamp", "ts"])
    dec_col = _pick_col(cols, ["decision", "meta_ok"])
    reason_col = _pick_col(cols, ["reason_canonical", "reason", "decision_reason"])
    p_col = _pick_col(cols, ["p_cal", "meta_p", "prob_val", "p", "p_raw"])
    sz_col = _pick_col(cols, ["size_mult", "risk_scale"])
    strat_col = _pick_col(cols, ["strat_ok"])
    meta_col = _pick_col(cols, ["meta_ok"])
    schema_col = _pick_col(cols, ["schema_ok"])
    scope_col = _pick_col(cols, ["scope_ok"])

    if sym_col is None or ts_col is None:
        raise ValueError(f"{side}: missing required symbol/timestamp columns")

    d = df.copy()
    d[sym_col] = d[sym_col].astype(str).str.upper().str.strip()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d[d[sym_col] != ""].copy()
    d = d[d[ts_col].notna()].copy()
    if d.empty:
        raise ValueError(f"{side}: no valid rows after timestamp/symbol normalization")

    if dec_col is not None:
        d["__enter"] = d[dec_col].map(_decision_to_int)
    else:
        d["__enter"] = np.nan

    # live log parser case: decision may require meta_ok && strat_ok
    if side == "live" and d["__enter"].isna().all():
        if (meta_col is not None) and (strat_col is not None):
            m = d[meta_col].map(_to_bool)
            s = d[strat_col].map(_to_bool)
            d["__enter"] = (m.fillna(False) & s.fillna(False)).astype(int)
        elif meta_col is not None:
            d["__enter"] = d[meta_col].map(_to_bool).fillna(False).astype(int)

    # backtester should always have decision text. If not, fallback.
    if side == "backtest" and d["__enter"].isna().all() and meta_col is not None:
        d["__enter"] = d[meta_col].map(_to_bool).fillna(False).astype(int)

    d["__reason_raw"] = d[reason_col].astype(str) if reason_col is not None else ""
    d["__reason_norm"] = d["__reason_raw"].map(_normalize_reason)
    d["__p"] = pd.to_numeric(d[p_col], errors="coerce") if p_col is not None else np.nan
    d["__size"] = pd.to_numeric(d[sz_col], errors="coerce") if sz_col is not None else np.nan
    d["__schema_ok"] = d[schema_col].map(_to_bool) if schema_col is not None else None
    d["__scope_ok"] = d[scope_col].map(_to_bool) if scope_col is not None else None
    d["__meta_ok"] = d[meta_col].map(_to_bool) if meta_col is not None else None
    d["__strat_ok"] = d[strat_col].map(_to_bool) if strat_col is not None else None
    d["__ts_bucket"] = d[ts_col].dt.floor(bucket)

    out = pd.DataFrame(
        {
            "symbol": d[sym_col],
            "ts": d[ts_col],
            "ts_bucket": d["__ts_bucket"],
            "enter": pd.to_numeric(d["__enter"], errors="coerce"),
            "reason_raw": d["__reason_raw"],
            "reason": d["__reason_norm"],
            "prob": d["__p"],
            "size_mult": d["__size"],
            "schema_ok": d["__schema_ok"],
            "scope_ok": d["__scope_ok"],
            "meta_ok": d["__meta_ok"],
            "strat_ok": d["__strat_ok"],
        }
    )
    out = out.sort_values(["symbol", "ts_bucket", "ts"], kind="mergesort")
    out = out.drop_duplicates(subset=["symbol", "ts_bucket"], keep="last").reset_index(drop=True)
    stats: Dict[str, object] = {
        "rows_raw": int(len(df)),
        "rows_norm": int(len(out)),
        "columns_used": {
            "symbol": sym_col,
            "timestamp": ts_col,
            "decision": dec_col,
            "reason": reason_col,
            "prob": p_col,
            "size_mult": sz_col,
            "schema_ok": schema_col,
            "scope_ok": scope_col,
            "meta_ok": meta_col,
            "strat_ok": strat_col,
        },
    }
    return out, stats


def _trade_summary(path: Optional[Path], side: str) -> Dict[str, object]:
    if path is None:
        return {"side": side, "present": False}
    if not path.exists():
        return {"side": side, "present": False, "error": f"not_found:{path}"}
    try:
        df = _read_table(path)
    except Exception as e:
        return {"side": side, "present": False, "error": f"read_error:{type(e).__name__}:{e}"}

    cols = list(df.columns)
    ts_col = _pick_col(cols, ["closed_at", "exit_ts", "timestamp", "ts"])
    pnl_col = _pick_col(cols, ["pnl", "pnl_cash", "realized_pnl", "realized p&l", "pnl_net"])
    out: Dict[str, object] = {"side": side, "present": True, "rows": int(len(df))}
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        ts = ts.dropna()
        if not ts.empty:
            out["min_ts"] = str(ts.min())
            out["max_ts"] = str(ts.max())
    if pnl_col is not None:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
        if not pnl.empty:
            out["total_pnl"] = float(pnl.sum())
            out["avg_pnl"] = float(pnl.mean())
            out["win_rate"] = float((pnl > 0).mean())
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            gp = float(wins.sum()) if not wins.empty else 0.0
            gl = float(losses.sum()) if not losses.empty else 0.0
            out["profit_factor"] = float(gp / max(abs(gl), 1e-12))
    return out


def _max_drawdown_from_pnl(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return float("nan")
    eq = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    if dd.size == 0:
        return float("nan")
    return float(np.nanmin(dd))


def _group_perf_rows(df: pd.DataFrame, bucket_col: str, dim_name: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for bucket, g in df.groupby(bucket_col, dropna=False, sort=True):
        pnl = pd.to_numeric(g["pnl_R"], errors="coerce").dropna().to_numpy(dtype=float)
        n = int(len(pnl))
        if n <= 0:
            continue
        out.append(
            {
                "dimension": dim_name,
                "bucket": str(bucket),
                "n_trades": n,
                "hit_rate": float((pnl > 0).mean()),
                "pnl_R_mean": float(np.nanmean(pnl)),
                "pnl_R_sum": float(np.nansum(pnl)),
                "max_drawdown_R": _max_drawdown_from_pnl(pnl),
            }
        )
    return out


def _derive_risk_on_series(tr: pd.DataFrame, btc_vol_hi: float) -> pd.Series:
    cols = list(tr.columns)
    ro_col = _pick_col(cols, ["risk_on_1", "risk_on"])
    if ro_col is not None:
        x = pd.to_numeric(tr[ro_col], errors="coerce")
        return x.where(x.isin([0, 1]), np.nan)

    regime_up_col = _pick_col(cols, ["regime_up"])
    btc_trend_col = _pick_col(cols, ["btc_trend_slope", "btcusdt_trend_slope"])
    btc_vol_col = _pick_col(cols, ["btc_vol_regime_level", "btcusdt_vol_regime_level"])
    if regime_up_col is None or btc_trend_col is None or btc_vol_col is None:
        return pd.Series(np.nan, index=tr.index, dtype=float)

    ru = pd.to_numeric(tr[regime_up_col], errors="coerce")
    bt = pd.to_numeric(tr[btc_trend_col], errors="coerce")
    bv = pd.to_numeric(tr[btc_vol_col], errors="coerce")
    risk_on = (ru == 1) & (bt > 0) & (bv < float(btc_vol_hi))
    return risk_on.astype(float)


def _build_regime_attribution(bt_trades: Optional[Path]) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    if bt_trades is None or (not bt_trades.exists()):
        return {"status": "unavailable", "reason": "bt_trades_missing"}, empty, empty
    try:
        tr = _read_table(bt_trades)
    except Exception as e:
        return {"status": "unavailable", "reason": f"read_error:{type(e).__name__}:{e}"}, empty, empty
    if tr.empty:
        return {"status": "unavailable", "reason": "bt_trades_empty"}, empty, empty

    cols = list(tr.columns)
    ts_col = _pick_col(cols, ["exit_ts", "closed_at", "timestamp", "ts", "entry_ts"])
    pnl_r_col = _pick_col(cols, ["pnl_R", "pnl_r"])
    if ts_col is None or pnl_r_col is None:
        return {
            "status": "unavailable",
            "reason": "required_trade_columns_missing",
            "ts_col": ts_col,
            "pnl_r_col": pnl_r_col,
        }, empty, empty

    w = tr.copy()
    w["ts"] = pd.to_datetime(w[ts_col], utc=True, errors="coerce")
    w["pnl_R"] = pd.to_numeric(w[pnl_r_col], errors="coerce")
    w = w[w["ts"].notna() & w["pnl_R"].notna()].copy()
    if w.empty:
        return {"status": "unavailable", "reason": "no_valid_trade_rows_after_cleaning"}, empty, empty

    btc_hi = _btc_vol_hi()
    w["risk_on_val"] = _derive_risk_on_series(w, btc_hi)
    w["risk_on_bucket"] = np.where(
        w["risk_on_val"] == 1.0,
        "risk_on",
        np.where(w["risk_on_val"] == 0.0, "risk_off", "risk_unknown"),
    )

    eth_col = _pick_col(cols, ["eth_macd_hist_4h"])
    if eth_col is not None:
        eth = pd.to_numeric(w[eth_col], errors="coerce")
        w["eth_hist_bucket"] = np.where(
            eth > 0,
            "eth_hist_pos",
            np.where(eth < 0, "eth_hist_neg", "eth_hist_unknown"),
        )
    else:
        w["eth_hist_bucket"] = "eth_hist_unknown"

    btc_vol_col = _pick_col(cols, ["btc_vol_regime_level", "btcusdt_vol_regime_level"])
    if btc_vol_col is not None:
        bv = pd.to_numeric(w[btc_vol_col], errors="coerce")
        w["btc_vol_bucket"] = np.where(
            bv >= btc_hi,
            "btc_vol_high",
            np.where(bv < btc_hi, "btc_vol_low", "btc_vol_unknown"),
        )
    else:
        w["btc_vol_bucket"] = "btc_vol_unknown"

    w = w.sort_values("ts", kind="mergesort").reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    rows.extend(_group_perf_rows(w, "risk_on_bucket", "risk_on"))
    rows.extend(_group_perf_rows(w, "eth_hist_bucket", "eth_hist_sign"))
    rows.extend(_group_perf_rows(w, "btc_vol_bucket", "btc_vol_regime"))
    regime_df = pd.DataFrame(rows)

    mrows: List[Dict[str, object]] = []
    w["month"] = w["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    for month, g in w.groupby("month", sort=True):
        pnl = pd.to_numeric(g["pnl_R"], errors="coerce").dropna().to_numpy(dtype=float)
        if pnl.size == 0:
            continue
        mrows.append(
            {
                "month": str(month),
                "n_trades": int(pnl.size),
                "hit_rate": float((pnl > 0).mean()),
                "pnl_R_sum": float(np.nansum(pnl)),
                "pnl_R_mean": float(np.nanmean(pnl)),
                "max_drawdown_R": _max_drawdown_from_pnl(pnl),
                "risk_on_share": float((g["risk_on_bucket"] == "risk_on").mean()),
                "eth_hist_neg_share": float((g["eth_hist_bucket"] == "eth_hist_neg").mean()),
                "btc_vol_high_share": float((g["btc_vol_bucket"] == "btc_vol_high").mean()),
            }
        )
    monthly_df = pd.DataFrame(mrows).sort_values("month", kind="mergesort") if mrows else pd.DataFrame()

    by_dim: Dict[str, int] = {}
    if not regime_df.empty:
        for dim, g in regime_df.groupby("dimension", sort=True):
            by_dim[str(dim)] = int(len(g))
    diag: Dict[str, object] = {
        "status": "ok",
        "btc_vol_hi": float(btc_hi),
        "trade_rows_used": int(len(w)),
        "dimension_row_counts": by_dim,
        "monthly_rows": int(len(monthly_df)),
    }
    return diag, regime_df, monthly_df


def _classify_failure_bucket(row: pd.Series) -> str:
    enter_live = _parse_num(row.get("enter_live"))
    if np.isfinite(enter_live) and int(enter_live) == 1:
        return "entered"

    merge_tag = str(row.get("_merge", "")).strip().lower()
    reason = _normalize_reason(row.get("reason_live"))
    schema_ok = _parse_bool(row.get("schema_ok_live"))
    scope_ok = _parse_bool(row.get("scope_ok_live"))
    meta_ok = _parse_bool(row.get("meta_ok_live"))
    strat_ok = _parse_bool(row.get("strat_ok_live"))

    if merge_tag == "left_only":
        return "no_signals"
    if (schema_ok is False) or _contains_any(reason, ["schema_fail", "feature_missing", "schema_error"]):
        return "schema_fail"
    if (scope_ok is False) or _contains_any(reason, ["scope_fail", "scope:fail", "scope_mismatch"]):
        return "scope_fail"
    if _contains_any(reason, ["below_pstar", "meta_prob", "below_threshold", "below_prob"]):
        return "below_pstar"
    if (meta_ok is False) or (strat_ok is False):
        return "gate_fail"
    if reason:
        return "gate_fail"
    return "other"


def _build_failure_bucket_audit(
    merged: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    if merged.empty:
        diag = {
            "status": "unavailable",
            "reason": "no_rows",
            "skip_rows_live": 0,
            "canonical_bucket_counts": {k: 0 for k in CANONICAL_BUCKETS},
            "assigned_rate_live": float("nan"),
        }
        return diag, pd.DataFrame(), pd.DataFrame()

    x = merged.copy()
    x["enter_live_i"] = pd.to_numeric(x.get("enter_live"), errors="coerce")
    live_rows = x[x["_merge"] != "right_only"].copy()
    if live_rows.empty:
        diag = {
            "status": "unavailable",
            "reason": "no_live_rows_after_merge",
            "skip_rows_live": 0,
            "canonical_bucket_counts": {k: 0 for k in CANONICAL_BUCKETS},
            "assigned_rate_live": float("nan"),
        }
        return diag, pd.DataFrame(), pd.DataFrame()

    live_skips = live_rows[(live_rows["enter_live_i"].isna()) | (live_rows["enter_live_i"] != 1)].copy()
    if live_skips.empty:
        diag = {
            "status": "ok",
            "reason": "no_live_skips",
            "skip_rows_live": 0,
            "canonical_bucket_counts": {k: 0 for k in CANONICAL_BUCKETS},
            "assigned_rate_live": 1.0,
            "pipeline_failure_rows": 0,
            "no_edge_rows": 0,
            "other_rows": 0,
        }
        return diag, pd.DataFrame(), pd.DataFrame()

    live_skips["failure_bucket"] = live_skips.apply(_classify_failure_bucket, axis=1)
    live_month_ts = pd.to_datetime(live_skips["ts_bucket"], utc=True, errors="coerce")
    live_skips["month"] = live_month_ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    counts = live_skips["failure_bucket"].value_counts(dropna=False).to_dict()
    bucket_counts = {k: int(counts.get(k, 0)) for k in CANONICAL_BUCKETS}

    skip_n = int(len(live_skips))
    assigned_n = int(skip_n - bucket_counts.get("other", 0))
    assigned_rate = _safe_rate(assigned_n, skip_n)
    pipeline_rows = int(sum(bucket_counts.get(k, 0) for k in PIPELINE_BUCKETS))
    no_edge_rows = int(sum(bucket_counts.get(k, 0) for k in NO_EDGE_BUCKETS))

    monthly = (
        live_skips.groupby(["month", "failure_bucket"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("month", kind="mergesort")
    )
    for k in CANONICAL_BUCKETS:
        if k not in monthly.columns:
            monthly[k] = 0
    monthly["skip_total"] = monthly[list(CANONICAL_BUCKETS)].sum(axis=1)
    monthly["pipeline_failure_skips"] = monthly[[k for k in CANONICAL_BUCKETS if k in PIPELINE_BUCKETS]].sum(axis=1)
    monthly["no_edge_skips"] = monthly[[k for k in CANONICAL_BUCKETS if k in NO_EDGE_BUCKETS]].sum(axis=1)
    monthly["assigned_rate_live"] = np.where(
        monthly["skip_total"] > 0,
        (monthly["skip_total"] - monthly["other"]) / monthly["skip_total"],
        np.nan,
    )
    cols = [
        "month",
        "skip_total",
        "no_signals",
        "schema_fail",
        "scope_fail",
        "below_pstar",
        "gate_fail",
        "other",
        "pipeline_failure_skips",
        "no_edge_skips",
        "assigned_rate_live",
    ]
    monthly = monthly[cols]

    bucketed_rows = live_skips[
        ["symbol", "ts_bucket", "reason_live", "reason_raw_live", "failure_bucket", "_merge"]
    ].copy()
    bucketed_rows = bucketed_rows.sort_values(["ts_bucket", "symbol"], kind="mergesort").reset_index(drop=True)

    top_reasons: List[Dict[str, object]] = []
    reason_counts = (
        live_skips["reason_live"].astype(str).fillna("").value_counts(dropna=False).head(10)
    )
    for reason, cnt in reason_counts.items():
        top_reasons.append({"reason": str(reason), "count": int(cnt)})

    diag = {
        "status": "ok",
        "skip_rows_live": skip_n,
        "canonical_bucket_counts": bucket_counts,
        "assigned_rate_live": float(assigned_rate) if np.isfinite(assigned_rate) else float("nan"),
        "pipeline_failure_rows": pipeline_rows,
        "no_edge_rows": no_edge_rows,
        "other_rows": int(bucket_counts.get("other", 0)),
        "top_skip_reasons_live": top_reasons,
    }
    return diag, monthly, bucketed_rows


def _build_monthly_regime_health(
    regime_monthly: pd.DataFrame,
    failure_monthly: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    if regime_monthly.empty and failure_monthly.empty:
        return {"status": "unavailable", "reason": "no_monthly_inputs"}, pd.DataFrame()

    rm = regime_monthly.copy() if not regime_monthly.empty else pd.DataFrame(columns=["month"])
    fm = failure_monthly.copy() if not failure_monthly.empty else pd.DataFrame(columns=["month"])
    for c in ["month"]:
        if c not in rm.columns:
            rm[c] = []
        if c not in fm.columns:
            fm[c] = []

    out = rm.merge(fm, on="month", how="outer").sort_values("month", kind="mergesort")

    if "skip_total" not in out.columns:
        out["skip_total"] = 0
    if "pipeline_failure_skips" not in out.columns:
        out["pipeline_failure_skips"] = 0
    if "no_edge_skips" not in out.columns:
        out["no_edge_skips"] = 0

    out["pipeline_failure_share"] = np.where(
        out["skip_total"] > 0,
        pd.to_numeric(out["pipeline_failure_skips"], errors="coerce") / pd.to_numeric(out["skip_total"], errors="coerce"),
        np.nan,
    )
    out["no_edge_share"] = np.where(
        out["skip_total"] > 0,
        pd.to_numeric(out["no_edge_skips"], errors="coerce") / pd.to_numeric(out["skip_total"], errors="coerce"),
        np.nan,
    )

    for c in ("pnl_R_sum", "pipeline_failure_share", "no_edge_share"):
        if c not in out.columns:
            out[c] = np.nan

    def _cause(row: pd.Series) -> str:
        p = _parse_num(row.get("pipeline_failure_share"))
        pnl = _parse_num(row.get("pnl_R_sum"))
        if np.isfinite(p) and p >= 0.20:
            return "pipeline_failure_bias"
        if np.isfinite(pnl) and pnl < 0:
            return "unfavorable_regime_or_edge_decay"
        return "stable_or_neutral"

    out["dominant_cause"] = out.apply(_cause, axis=1)

    diag = {
        "status": "ok",
        "months": int(len(out)),
        "negative_months": int((pd.to_numeric(out.get("pnl_R_sum"), errors="coerce") < 0).sum()) if "pnl_R_sum" in out.columns else 0,
        "pipeline_issue_months": int((out["dominant_cause"] == "pipeline_failure_bias").sum()),
        "unfavorable_regime_months": int((out["dominant_cause"] == "unfavorable_regime_or_edge_decay").sum()),
    }
    return diag, out


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 24) -> str:
    if df.empty:
        return "_no rows_"
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in d.iterrows():
        vals: List[str] = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (float, np.floating)):
                vals.append(_fmt(v, 6))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_monthly_health_report(
    out_path: Path,
    *,
    title: str,
    summary: Dict[str, object],
    failure_diag: Dict[str, object],
    regime_diag: Dict[str, object],
    health_diag: Dict[str, object],
    regime_rows: pd.DataFrame,
    health_rows: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- status: `{summary.get('status')}`")
    lines.append(f"- overlap_rate_live: `{_fmt((summary.get('row_stats') or {}).get('overlap_rate_live'), 4)}`")
    lines.append(f"- enter_agreement: `{_fmt((summary.get('agreement') or {}).get('enter_agreement'), 4)}`")
    lines.append("")
    lines.append("## Failure Buckets (Live Skips)")
    lines.append(f"- skip_rows_live: `{failure_diag.get('skip_rows_live')}`")
    lines.append(f"- assigned_rate_live: `{_fmt(failure_diag.get('assigned_rate_live'), 4)}`")
    lines.append(f"- pipeline_failure_rows: `{failure_diag.get('pipeline_failure_rows')}`")
    lines.append(f"- no_edge_rows: `{failure_diag.get('no_edge_rows')}`")
    lines.append("")
    lines.append("## Regime Attribution")
    lines.append(f"- status: `{regime_diag.get('status')}`")
    lines.append(f"- trade_rows_used: `{regime_diag.get('trade_rows_used')}`")
    lines.append(f"- btc_vol_hi: `{_fmt(regime_diag.get('btc_vol_hi'), 6)}`")
    lines.append("")
    lines.append(_df_to_markdown(regime_rows, max_rows=24))
    lines.append("")
    lines.append("## Monthly Health")
    lines.append(f"- status: `{health_diag.get('status')}`")
    lines.append(f"- months: `{health_diag.get('months')}`")
    lines.append(f"- pipeline_issue_months: `{health_diag.get('pipeline_issue_months')}`")
    lines.append(f"- unfavorable_regime_months: `{health_diag.get('unfavorable_regime_months')}`")
    lines.append("")
    lines.append(_df_to_markdown(health_rows, max_rows=36))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare live decisions vs backtest decisions.")
    p.add_argument("--live-decisions", required=True)
    p.add_argument("--bt-decisions", required=True)
    p.add_argument("--live-trades", default="")
    p.add_argument("--bt-trades", default="")
    p.add_argument("--bucket", default="5min", help="Timestamp bucketing interval for key matching.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--min-overlap-rate", type=float, default=0.50)
    p.add_argument("--min-enter-agreement", type=float, default=0.85)
    p.add_argument("--max-mismatches", type=int, default=200)
    p.add_argument("--html-title", default="Donch Autopar Report")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    outdir = Path(a.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    live_path = Path(a.live_decisions).expanduser().resolve()
    bt_path = Path(a.bt_decisions).expanduser().resolve()
    live_tr_path = Path(a.live_trades).expanduser().resolve() if a.live_trades.strip() else None
    bt_tr_path = Path(a.bt_trades).expanduser().resolve() if a.bt_trades.strip() else None

    live_raw = _read_table(live_path)
    bt_raw = _read_table(bt_path)

    live, live_stats = _norm_decisions(live_raw, side="live", bucket=a.bucket)
    bt, bt_stats = _norm_decisions(bt_raw, side="backtest", bucket=a.bucket)

    merged = live.merge(
        bt,
        on=["symbol", "ts_bucket"],
        how="outer",
        suffixes=("_live", "_bt"),
        indicator=True,
    )
    merged = merged.sort_values(["ts_bucket", "symbol"], kind="mergesort").reset_index(drop=True)

    both = merged[merged["_merge"] == "both"].copy()
    live_only = merged[merged["_merge"] == "left_only"].copy()
    bt_only = merged[merged["_merge"] == "right_only"].copy()

    n_live = int(len(live))
    n_bt = int(len(bt))
    n_both = int(len(both))
    n_live_only = int(len(live_only))
    n_bt_only = int(len(bt_only))

    overlap_rate_live = _safe_rate(n_both, n_live)
    overlap_rate_bt = _safe_rate(n_both, n_bt)

    enter_agreement = float("nan")
    reason_agreement = float("nan")
    conf: Dict[str, int] = {}
    p_abs_err_mean = float("nan")
    p_abs_err_p90 = float("nan")
    size_abs_err_mean = float("nan")
    size_abs_err_p90 = float("nan")

    if n_both > 0:
        x = both.copy()
        x["enter_live_i"] = pd.to_numeric(x["enter_live"], errors="coerce")
        x["enter_bt_i"] = pd.to_numeric(x["enter_bt"], errors="coerce")
        valid_enter = x["enter_live_i"].notna() & x["enter_bt_i"].notna()
        if bool(valid_enter.any()):
            enter_agreement = float((x.loc[valid_enter, "enter_live_i"] == x.loc[valid_enter, "enter_bt_i"]).mean())
            cm = x.loc[valid_enter]
            conf = {
                "live0_bt0": int(((cm["enter_live_i"] == 0) & (cm["enter_bt_i"] == 0)).sum()),
                "live0_bt1": int(((cm["enter_live_i"] == 0) & (cm["enter_bt_i"] == 1)).sum()),
                "live1_bt0": int(((cm["enter_live_i"] == 1) & (cm["enter_bt_i"] == 0)).sum()),
                "live1_bt1": int(((cm["enter_live_i"] == 1) & (cm["enter_bt_i"] == 1)).sum()),
            }

        r_ok = (x["reason_live"].astype(str).str.len() > 0) & (x["reason_bt"].astype(str).str.len() > 0)
        if bool(r_ok.any()):
            reason_agreement = float((x.loc[r_ok, "reason_live"].astype(str) == x.loc[r_ok, "reason_bt"].astype(str)).mean())

        p_live = pd.to_numeric(x["prob_live"], errors="coerce")
        p_bt = pd.to_numeric(x["prob_bt"], errors="coerce")
        m = p_live.notna() & p_bt.notna()
        if bool(m.any()):
            ae = (p_live[m] - p_bt[m]).abs().to_numpy(dtype=float)
            p_abs_err_mean = float(np.mean(ae))
            p_abs_err_p90 = float(np.quantile(ae, 0.9))

        s_live = pd.to_numeric(x["size_mult_live"], errors="coerce")
        s_bt = pd.to_numeric(x["size_mult_bt"], errors="coerce")
        m2 = s_live.notna() & s_bt.notna()
        if bool(m2.any()):
            ae2 = (s_live[m2] - s_bt[m2]).abs().to_numpy(dtype=float)
            size_abs_err_mean = float(np.mean(ae2))
            size_abs_err_p90 = float(np.quantile(ae2, 0.9))

    live_trade_stats = _trade_summary(live_tr_path, "live")
    bt_trade_stats = _trade_summary(bt_tr_path, "backtest")

    failure_diag, failure_monthly_df, failure_rows_df = _build_failure_bucket_audit(merged)
    regime_diag, regime_rows_df, regime_monthly_df = _build_regime_attribution(bt_tr_path)
    health_diag, health_df = _build_monthly_regime_health(regime_monthly_df, failure_monthly_df)

    status = "ok"
    checks: Dict[str, object] = {}
    checks["overlap_rate_live"] = overlap_rate_live
    checks["enter_agreement"] = enter_agreement
    checks["min_overlap_rate"] = float(a.min_overlap_rate)
    checks["min_enter_agreement"] = float(a.min_enter_agreement)
    checks["failure_bucket_assigned_rate_live"] = failure_diag.get("assigned_rate_live")
    checks["failure_bucket_assigned_rate_live_min"] = 0.95

    if not np.isfinite(overlap_rate_live) or overlap_rate_live < float(a.min_overlap_rate):
        status = "warn"
    if np.isfinite(enter_agreement) and enter_agreement < float(a.min_enter_agreement):
        status = "warn"
    rate_live = _parse_num(failure_diag.get("assigned_rate_live"))
    if np.isfinite(rate_live) and rate_live < 0.95:
        status = "warn"
    if n_both == 0:
        status = "error"

    summary: Dict[str, object] = {
        "status": status,
        "bucket": a.bucket,
        "inputs": {
            "live_decisions": str(live_path),
            "bt_decisions": str(bt_path),
            "live_trades": str(live_tr_path) if live_tr_path else "",
            "bt_trades": str(bt_tr_path) if bt_tr_path else "",
        },
        "row_stats": {
            "live_rows": n_live,
            "bt_rows": n_bt,
            "overlap_rows": n_both,
            "live_only_rows": n_live_only,
            "bt_only_rows": n_bt_only,
            "overlap_rate_live": overlap_rate_live,
            "overlap_rate_bt": overlap_rate_bt,
        },
        "agreement": {
            "enter_agreement": enter_agreement,
            "reason_agreement": reason_agreement,
            "confusion": conf,
            "p_abs_err_mean": p_abs_err_mean,
            "p_abs_err_p90": p_abs_err_p90,
            "size_abs_err_mean": size_abs_err_mean,
            "size_abs_err_p90": size_abs_err_p90,
        },
        "normalization": {
            "live": live_stats,
            "backtest": bt_stats,
        },
        "trade_stats": {
            "live": live_trade_stats,
            "backtest": bt_trade_stats,
        },
        "failure_bucket_audit": failure_diag,
        "regime_attribution": regime_diag,
        "monthly_regime_health": health_diag,
        "checks": checks,
    }

    mismatches = both.copy()
    mismatches["enter_live_i"] = pd.to_numeric(mismatches["enter_live"], errors="coerce")
    mismatches["enter_bt_i"] = pd.to_numeric(mismatches["enter_bt"], errors="coerce")
    mismatches = mismatches[
        (mismatches["enter_live_i"] != mismatches["enter_bt_i"])
        | (mismatches["reason_live"].astype(str) != mismatches["reason_bt"].astype(str))
    ].copy()
    mismatches = mismatches.head(max(1, int(a.max_mismatches)))

    summary_path = outdir / "summary.json"
    rows_path = outdir / "comparison_rows.csv"
    mismatches_path = outdir / "mismatches.csv"
    html_path = outdir / "report.html"
    failure_monthly_path = outdir / "failure_bucket_audit.csv"
    failure_rows_path = outdir / "failure_bucket_rows.csv"
    regime_rows_path = outdir / "regime_attribution.csv"
    monthly_health_path = outdir / "monthly_regime_health.csv"
    monthly_health_md_path = outdir / "monthly_regime_health_report.md"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    merged.to_csv(rows_path, index=False)
    mismatches.to_csv(mismatches_path, index=False)
    failure_monthly_df.to_csv(failure_monthly_path, index=False)
    failure_rows_df.to_csv(failure_rows_path, index=False)
    regime_rows_df.to_csv(regime_rows_path, index=False)
    health_df.to_csv(monthly_health_path, index=False)

    _write_monthly_health_report(
        monthly_health_md_path,
        title=f"{a.html_title} - Monthly Regime Health",
        summary=summary,
        failure_diag=failure_diag,
        regime_diag=regime_diag,
        health_diag=health_diag,
        regime_rows=regime_rows_df,
        health_rows=health_df,
    )

    fb = failure_diag.get("canonical_bucket_counts", {})
    failure_table_html = pd.DataFrame(
        [
            {"bucket": k, "count": int((fb or {}).get(k, 0))}
            for k in CANONICAL_BUCKETS
        ]
    ).to_html(index=False)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{a.html_title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
h1,h2 {{ margin: 0.4em 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.ok {{ color: #0b8a2f; font-weight: bold; }}
.warn {{ color: #b26a00; font-weight: bold; }}
.error {{ color: #b00020; font-weight: bold; }}
</style></head><body>
<h1>{a.html_title}</h1>
<p>Status: <span class="{status}">{status.upper()}</span></p>

<h2>Row Stats</h2>
<table>
<tr><th>live_rows</th><th>bt_rows</th><th>overlap</th><th>live_only</th><th>bt_only</th><th>overlap_rate_live</th><th>enter_agreement</th><th>reason_agreement</th></tr>
<tr>
<td>{n_live}</td>
<td>{n_bt}</td>
<td>{n_both}</td>
<td>{n_live_only}</td>
<td>{n_bt_only}</td>
<td>{_fmt(overlap_rate_live, 4)}</td>
<td>{_fmt(enter_agreement, 4)}</td>
<td>{_fmt(reason_agreement, 4)}</td>
</tr>
</table>

<h2>Agreement Errors</h2>
<table>
<tr><th>p_abs_err_mean</th><th>p_abs_err_p90</th><th>size_abs_err_mean</th><th>size_abs_err_p90</th></tr>
<tr>
<td>{_fmt(p_abs_err_mean, 6)}</td>
<td>{_fmt(p_abs_err_p90, 6)}</td>
<td>{_fmt(size_abs_err_mean, 6)}</td>
<td>{_fmt(size_abs_err_p90, 6)}</td>
</tr>
</table>

<h2>Failure Buckets (Live Skips)</h2>
<p>assigned_rate_live={_fmt(failure_diag.get("assigned_rate_live"), 4)}; pipeline_failure_rows={failure_diag.get("pipeline_failure_rows")}; no_edge_rows={failure_diag.get("no_edge_rows")}</p>
{failure_table_html}

<h2>Regime Attribution (Backtest Trades)</h2>
<p>status={regime_diag.get("status")} trade_rows_used={regime_diag.get("trade_rows_used")} btc_vol_hi={_fmt(regime_diag.get("btc_vol_hi"), 6)}</p>
{regime_rows_df.to_html(index=False)}

<h2>Monthly Regime Health</h2>
<p>status={health_diag.get("status")} months={health_diag.get("months")} pipeline_issue_months={health_diag.get("pipeline_issue_months")} unfavorable_regime_months={health_diag.get("unfavorable_regime_months")}</p>
{health_df.to_html(index=False)}

<h2>Trade Stats</h2>
<pre>{json.dumps(summary["trade_stats"], indent=2)}</pre>

<h2>Top Mismatches</h2>
{mismatches.head(100).to_html(index=False)}
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")

    print(f"[autopar] status={status}")
    print(f"[autopar] wrote {summary_path}")
    print(f"[autopar] wrote {rows_path}")
    print(f"[autopar] wrote {mismatches_path}")
    print(f"[autopar] wrote {failure_monthly_path}")
    print(f"[autopar] wrote {failure_rows_path}")
    print(f"[autopar] wrote {regime_rows_path}")
    print(f"[autopar] wrote {monthly_health_path}")
    print(f"[autopar] wrote {monthly_health_md_path}")
    print(f"[autopar] wrote {html_path}")

    return 2 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
