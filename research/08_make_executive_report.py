#!/usr/bin/env python3
"""
research/08_make_executive_report.py

Executive HTML report for the 00–07 research pipeline outputs.

Key behaviors:
- Uses Step 05 outputs for calibration/EV/sizing (no need to recompute EV from calibrated predictions).
- Robustly merges regimes to get risk_on (supports risk_on, risk_on_1, etc).
- Computes SCORE COVERAGE using 04 oof_predictions.parquet and writes unscored_trades.csv.

Outputs:
- <outdir>/report.html
- <outdir>/unscored_trades.csv   (if any unscored trades are detected)
- <outdir>/report_debug.json     (merge/coverage diagnostics)

Usage:
  python research/08_make_executive_report.py \
    --trades results/trades.clean.csv \
    --root research_outputs \
    --outdir research_outputs/EXEC_REPORT \
    --title "Donch Meta-Model: Executive Summary"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# -----------------------------
# Utility helpers
# -----------------------------
def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S (UTC)")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_dt_utc(s: pd.Series) -> pd.Series:
    # tz-aware UTC; consistent across merges
    return pd.to_datetime(s, utc=True, errors="coerce")


def _make_entry_ts_ns(df: pd.DataFrame, col: str = "entry_ts") -> pd.DataFrame:
    """
    Create a merge-safe int64 nanosecond timestamp key.
    This avoids tz-aware vs tz-naive join failures.
    """
    if col not in df.columns:
        return df
    dt = _to_dt_utc(df[col])
    # datetime64[ns, UTC] -> int64 ns since epoch; NaT -> <NA> after mask
    ns = dt.view("int64")
    # pandas uses min int for NaT; convert to NA
    ns = pd.Series(ns, index=df.index)
    ns = ns.mask(dt.isna(), other=pd.NA)
    df[f"{col}_ns"] = ns
    return df


def _standardize_symbol(df: pd.DataFrame, col: str = "symbol") -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype(str)
    return df


def _df_to_html_table(df: Optional[pd.DataFrame], title: Optional[str] = None, max_rows: int = 30, float_digits: int = 4) -> str:
    if df is None or len(df) == 0:
        return "<p class='note'><em>No data available.</em></p>"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows).copy()
    # format floats
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}f}")
    html = d.to_html(index=False, escape=True)
    if title:
        return f"<h4>{title}</h4>\n{html}"
    return html


def _fig_to_data_uri(fig) -> Optional[str]:
    if not HAS_PLT:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _card(title: str, body: str) -> str:
    return f"""
    <section class="card">
      <h2>{title}</h2>
      {body}
    </section>
    """


def _info_box(title: str, bullets: List[str]) -> str:
    li = "\n".join([f"<li>{b}</li>" for b in bullets])
    return f"""
    <div class="info">
      <div class="info-title">{title}</div>
      <ul>{li}</ul>
    </div>
    """


def _warn(msg: str) -> str:
    return f"<div class='warn'><strong>Note:</strong> {msg}</div>"


def _fmt_int(x) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{int(x):,}"
    except Exception:
        return "—"


def _fmt_pct(x, digits: int = 1) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{100.0*float(x):.{digits}f}%"
    except Exception:
        return "—"


def _fmt_float(x, digits: int = 4) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def _fmt_date(ts) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


# -----------------------------
# Loading pipeline outputs
# -----------------------------
def load_trades(trades_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_csv, low_memory=False)
    if "entry_ts" not in df.columns:
        raise ValueError("Trades CSV missing required column: entry_ts")
    df = _standardize_symbol(df, "symbol")
    df["entry_ts"] = _to_dt_utc(df["entry_ts"])
    df = _make_entry_ts_ns(df, "entry_ts")

    if "trade_id" in df.columns:
        df["trade_id"] = _safe_num(df["trade_id"])
        df = df[df["trade_id"].notna()].copy()
        df["trade_id"] = df["trade_id"].astype("int64")

    for c in ["pnl_R", "WIN", "EXIT_FINAL", "rs_pct"]:
        if c in df.columns:
            df[c] = _safe_num(df[c])

    return df


def load_regimes(regimes_pq: Path) -> Optional[pd.DataFrame]:
    if not regimes_pq.exists():
        return None
    df = pd.read_parquet(regimes_pq)
    if "entry_ts" in df.columns:
        df["entry_ts"] = _to_dt_utc(df["entry_ts"])
        df = _make_entry_ts_ns(df, "entry_ts")
    df = _standardize_symbol(df, "symbol")
    if "trade_id" in df.columns:
        df["trade_id"] = _safe_num(df["trade_id"])
        df = df[df["trade_id"].notna()].copy()
        df["trade_id"] = df["trade_id"].astype("int64")
    return df


def load_cv_oof(cvdir: Path) -> Optional[pd.DataFrame]:
    p = cvdir / "oof_predictions.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)

    # normalize keys (best effort)
    if "entry_ts" in df.columns:
        df["entry_ts"] = _to_dt_utc(df["entry_ts"])
        df = _make_entry_ts_ns(df, "entry_ts")
    if "symbol" in df.columns:
        df = _standardize_symbol(df, "symbol")
    if "trade_id" in df.columns:
        df["trade_id"] = _safe_num(df["trade_id"])
        df = df[df["trade_id"].notna()].copy()
        df["trade_id"] = df["trade_id"].astype("int64")
    return df


def load_step05_tables(caldir: Path, model_col: str) -> Dict[str, Optional[pd.DataFrame]]:
    out: Dict[str, Optional[pd.DataFrame]] = {
        "calibration_bins": None,
        "ev_thresholds": None,
        "sizing_curve": None,
    }

    p_bins = caldir / f"calibration_bins_{model_col}.csv"
    p_ev = caldir / f"ev_thresholds_{model_col}.csv"
    p_sc = caldir / f"sizing_curve_{model_col}.csv"

    if p_bins.exists():
        out["calibration_bins"] = pd.read_csv(p_bins)
    if p_ev.exists():
        out["ev_thresholds"] = pd.read_csv(p_ev)
    if p_sc.exists():
        out["sizing_curve"] = pd.read_csv(p_sc)

    return out


def load_deployment_config(deploydir: Path) -> Tuple[Optional[dict], Optional[str]]:
    """
    Reads deployment_config.json if present (preferred), else any json.
    Returns (config, path_str).
    """
    p = deploydir / "deployment_config.json"
    if p.exists():
        return _read_json(p), str(p)
    # fallback: first json
    js = sorted(deploydir.glob("*.json"))
    if js:
        return _read_json(js[0]), str(js[0])
    return None, None


def extract_policy_from_config(cfg: Optional[dict]) -> Dict[str, object]:
    """
    Best-effort parsing of exported config into:
      model_col, model_name, calib_method, scope, threshold
    """
    out: Dict[str, object] = {
        "model_col": None,
        "model_name": None,
        "calib_method": None,
        "scope": "ALL",
        "threshold": None,
    }
    if not isinstance(cfg, dict):
        return out

    # common keys (your export format may vary)
    out["model_col"] = cfg.get("probability_column") or cfg.get("model_col") or cfg.get("proba_col") or cfg.get("model_probability_column")
    model = cfg.get("model", {})
    if isinstance(model, dict):
        out["model_name"] = model.get("name") or model.get("model_name")
    else:
        out["model_name"] = cfg.get("model_name")

    cal = cfg.get("calibration", {})
    if isinstance(cal, dict):
        out["calib_method"] = cal.get("method") or cal.get("name")
    else:
        out["calib_method"] = cfg.get("calibration")

    decision = cfg.get("decision", {})
    if isinstance(decision, dict):
        out["scope"] = decision.get("scope") or out["scope"]
        out["threshold"] = decision.get("threshold") or out["threshold"]
    else:
        out["scope"] = cfg.get("threshold_scope") or out["scope"]
        out["threshold"] = cfg.get("threshold") or out["threshold"]

    # normalize
    if out["scope"] is None:
        out["scope"] = "ALL"
    out["scope"] = str(out["scope"]).strip()

    try:
        if out["threshold"] is not None:
            out["threshold"] = float(out["threshold"])
    except Exception:
        out["threshold"] = None

    return out


# -----------------------------
# Risk-on + scoring coverage logic
# -----------------------------
def merge_regimes_into_trades(trades: pd.DataFrame, regimes: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    diag: List[str] = []
    if regimes is None or len(regimes) == 0:
        diag.append("regimes: not loaded")
        out = trades.copy()
        return out, diag

    out = trades.copy()

    # Prefer trade_id if both have it (most robust)
    if "trade_id" in out.columns and "trade_id" in regimes.columns:
        r = regimes.drop_duplicates(subset=["trade_id"]).copy()
        r = r.set_index("trade_id", drop=True)
        overlap = sorted(set(out.columns).intersection(set(r.columns)))
        # keep overlap only if it's the key itself
        overlap = [c for c in overlap if c != "trade_id"]
        if overlap:
            # remove overlapping from regimes to avoid confusion
            r = r.drop(columns=overlap, errors="ignore")
            diag.append(f"regimes overlap dropped: {overlap}")
        out = out.set_index("trade_id", drop=False).join(r, how="left").reset_index(drop=True)
        hit = float(out[r.columns].notna().any(axis=1).mean()) if len(r.columns) > 0 else 0.0
        diag.append(f"merged regimes on trade_id; hit_rate={hit:.3f}")
        return out, diag

    # Else use (entry_ts_ns, symbol) if available
    keys = []
    if "entry_ts_ns" in out.columns and "entry_ts_ns" in regimes.columns:
        keys.append("entry_ts_ns")
    if "symbol" in out.columns and "symbol" in regimes.columns:
        keys.append("symbol")

    if not keys:
        diag.append("regimes merge failed: no compatible keys (need trade_id or entry_ts/symbol)")
        return out, diag

    r = regimes.copy()
    r = r.drop_duplicates(subset=keys)
    overlap = sorted(set(out.columns).intersection(set(r.columns)) - set(keys))
    if overlap:
        r = r.drop(columns=overlap, errors="ignore")
        diag.append(f"regimes overlap dropped: {overlap}")

    before = len(out)
    out = out.merge(r, on=keys, how="left")
    # hit-rate based on any non-key regime col
    regime_cols = [c for c in r.columns if c not in keys]
    hit = float(out[regime_cols].notna().any(axis=1).mean()) if regime_cols else 0.0
    diag.append(f"merged regimes on {keys}; rows {before}->{len(out)}; hit_rate={hit:.3f}")
    return out, diag


def pick_risk_flag_column(df: pd.DataFrame, scope: str) -> Tuple[Optional[str], List[str]]:
    notes: List[str] = []
    scope = (scope or "ALL").strip()

    # For RISK_ON_1 we expect risk_on_1 in regimes outputs
    if scope.upper().startswith("RISK_ON"):
        want = scope.lower()  # e.g. risk_on_1
        if want in df.columns:
            notes.append(f"risk flag selected from scope: {scope} -> {want}")
            return want, notes
        # Fallback to risk_on if risk_on_1 missing
        if "risk_on" in df.columns:
            notes.append(f"scope implies '{want}', but column not found; falling back to 'risk_on'")
            return "risk_on", notes

    if "risk_on_1" in df.columns:
        notes.append("risk flag fallback: using 'risk_on_1'")
        return "risk_on_1", notes

    if "risk_on" in df.columns:
        notes.append("risk flag fallback: using 'risk_on'")
        return "risk_on", notes

    risk_like = sorted([c for c in df.columns if c.lower().startswith("risk_on")])
    if risk_like:
        notes.append(f"risk flag fallback: using '{risk_like[0]}' from {risk_like}")
        return risk_like[0], notes

    notes.append("no risk_on* column found")
    return None, notes


def compute_risk_split(df: pd.DataFrame, risk_col: str) -> pd.DataFrame:
    d = df.copy()
    r = d[risk_col]
    if r.dtype == bool:
        flag = r.astype("Int8")
    else:
        flag = _safe_num(r).fillna(0).astype("Int8")
    d["_risk"] = flag

    # win
    if "WIN" in d.columns:
        d["_win"] = (_safe_num(d["WIN"]) == 1).astype(int)
    elif "pnl_R" in d.columns:
        d["_win"] = (_safe_num(d["pnl_R"]) > 0).astype(int)
    else:
        d["_win"] = np.nan

    # time exits
    if "EXIT_FINAL" in d.columns:
        d["_time"] = (_safe_num(d["EXIT_FINAL"]) == 2).astype(int)
    elif "exit_reason" in d.columns:
        d["_time"] = d["exit_reason"].astype(str).str.lower().str.contains("time", na=False).astype(int)
    else:
        d["_time"] = np.nan

    d["_pnl_R"] = _safe_num(d["pnl_R"]) if "pnl_R" in d.columns else np.nan

    g = d.groupby("_risk")
    out = pd.DataFrame({
        "segment": ["NOT_RISK_ON" if k == 0 else "RISK_ON" for k in g.size().index.astype(int)],
        "n": g.size().values.astype(int),
        "mean_pnl_R": g["_pnl_R"].mean().values,
        "win_rate": g["_win"].mean().values,
        "time_exit_rate": g["_time"].mean().values,
    })
    return out.sort_values("segment")


def compute_rs_buckets(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "rs_pct" not in df.columns:
        return None
    rs = _safe_num(df["rs_pct"])
    if rs.isna().all():
        return None

    d = df.copy()
    d["_rs_bucket"] = pd.cut(rs, bins=[-np.inf, 30, 50, 70, np.inf], labels=["<=30", "30-50", "50-70", ">=70"], include_lowest=True)

    if "WIN" in d.columns:
        d["_win"] = (_safe_num(d["WIN"]) == 1).astype(int)
    else:
        d["_win"] = (_safe_num(d["pnl_R"]) > 0).astype(int) if "pnl_R" in d.columns else np.nan

    if "EXIT_FINAL" in d.columns:
        d["_time"] = (_safe_num(d["EXIT_FINAL"]) == 2).astype(int)
    else:
        d["_time"] = np.nan

    d["_pnl_R"] = _safe_num(d["pnl_R"]) if "pnl_R" in d.columns else np.nan

    g = d.groupby("_rs_bucket", dropna=False)
    out = pd.DataFrame({
        "rs_bucket": g.size().index.astype(str),
        "n": g.size().values.astype(int),
        "mean_pnl_R": g["_pnl_R"].mean().values,
        "win_rate": g["_win"].mean().values,
        "time_exit_rate": g["_time"].mean().values,
    })
    return out


def compute_score_coverage(
    trades: pd.DataFrame,
    oof: Optional[pd.DataFrame],
    model_col: str,
    outdir: Path,
) -> Tuple[Dict[str, object], Optional[pd.DataFrame]]:
    """
    Determine if "missing probabilities" are real or a join mismatch.

    Returns:
      summary dict
      unscored dataframe (written to csv if non-empty)
    """
    summary: Dict[str, object] = {
        "trades_n": int(len(trades)),
        "oof_loaded": bool(oof is not None and len(oof) > 0),
        "oof_n": None,
        "model_col": model_col,
        "oof_score_nonnull_rate": None,
        "matched_trades_rate": None,
        "scored_trades_rate": None,
        "unscored_n": None,
    }

    if oof is None or len(oof) == 0:
        summary["note"] = "oof_predictions.parquet not found or empty"
        return summary, None

    summary["oof_n"] = int(len(oof))

    if model_col not in oof.columns:
        summary["note"] = f"model_col '{model_col}' not present in oof_predictions"
        return summary, None

    score = _safe_num(oof[model_col])
    summary["oof_score_nonnull_rate"] = float(score.notna().mean())

    # Build join keys for matching trades -> oof
    # Prefer trade_id if present in both
    use_trade_id = ("trade_id" in trades.columns) and ("trade_id" in oof.columns)
    if use_trade_id:
        o = oof[["trade_id", model_col]].copy()
        o = o.drop_duplicates(subset=["trade_id"])
        merged = trades[["trade_id", "symbol", "entry_ts", "entry_ts_ns"]].copy()
        merged = merged.merge(o, on="trade_id", how="left")
        matched = merged[model_col].notna().mean()  # match implies score
        summary["matched_trades_rate"] = float((merged["trade_id"].isin(o["trade_id"])).mean())
        summary["scored_trades_rate"] = float(matched)
    else:
        # Use (entry_ts_ns, symbol) if possible
        if ("entry_ts_ns" in trades.columns) and ("entry_ts_ns" in oof.columns) and ("symbol" in trades.columns) and ("symbol" in oof.columns):
            o = oof[["entry_ts_ns", "symbol", model_col]].copy()
            o = o.drop_duplicates(subset=["entry_ts_ns", "symbol"])
            merged = trades[["symbol", "entry_ts", "entry_ts_ns"]].copy()
            merged = merged.merge(o, on=["entry_ts_ns", "symbol"], how="left")
            summary["matched_trades_rate"] = float(merged[["entry_ts_ns", "symbol"]].merge(o[["entry_ts_ns", "symbol"]], on=["entry_ts_ns", "symbol"], how="left", indicator=True)["_merge"].eq("both").mean())
            summary["scored_trades_rate"] = float(merged[model_col].notna().mean())
        else:
            summary["note"] = "No compatible join keys between trades and oof (need trade_id OR entry_ts_ns+symbol)."
            return summary, None

    # Build unscored list
    merged_df = merged.copy()
    merged_df["_unscored"] = merged_df[model_col].isna()
    unscored = merged_df[merged_df["_unscored"]].copy()
    summary["unscored_n"] = int(len(unscored))
    if len(unscored) > 0 and "reason" in unscored.columns:
        summary["unscored_reason_counts"] = {k: int(v) for k, v in unscored["reason"].value_counts().to_dict().items()}
        # After Step 04 fix, missing scores should never occur; treat as critical.
        summary["note"] = (
            f"CRITICAL: {int(len(unscored))} trades are unscored in OOF ({summary.get('unscored_reason_counts')}). "
            "This typically means CV did not fill some OOF slots (OOF_SCORE_NA) or the join keys failed (NO_OOF_ROW). "
            "Re-run research/04_models_cv.py; it should fail loudly if this persists."
        )

    # Reason inference (best effort)
    # If join match is missing vs score missing in oof, distinguish:
    if use_trade_id:
        known_ids = set(oof["trade_id"].dropna().astype("int64").unique().tolist())
        unscored["reason"] = np.where(unscored["trade_id"].isin(known_ids), "OOF_SCORE_NA", "NO_OOF_ROW")
        cols = ["trade_id", "symbol", "entry_ts", "reason"]
    else:
        # we don’t have a clean membership test without big memory; approximate via merge indicator
        tmp = trades[["symbol", "entry_ts_ns", "entry_ts"]].copy()
        tmp = tmp.merge(o[["entry_ts_ns", "symbol"]].assign(_has_oof=1), on=["entry_ts_ns", "symbol"], how="left")
        unscored = tmp[tmp["_has_oof"].isna()].copy()
        unscored["reason"] = "NO_OOF_ROW"
        cols = ["symbol", "entry_ts", "entry_ts_ns", "reason"]

    # Write CSV if any
    if len(unscored) > 0:
        out_csv = outdir / "unscored_trades.csv"
        unscored[cols].to_csv(out_csv, index=False)
    return summary, unscored


# -----------------------------
# Charts from tables
# -----------------------------
def plot_pnl_hist(df: pd.DataFrame) -> Optional[str]:
    if not HAS_PLT or "pnl_R" not in df.columns:
        return None
    x = _safe_num(df["pnl_R"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 50:
        return None
    fig = plt.figure(figsize=(9, 3.8))
    plt.hist(x.values, bins=60)
    plt.title("Distribution of trade outcomes (pnl_R)")
    plt.xlabel("pnl_R (R-multiple)")
    plt.ylabel("Count")
    return _fig_to_data_uri(fig)


def plot_ev_curve(ev: pd.DataFrame, scope: str) -> Optional[str]:
    if not HAS_PLT or ev is None or len(ev) == 0:
        return None
    if "scope" not in ev.columns:
        return None
    sub = ev[ev["scope"].astype(str) == str(scope)].copy()
    if len(sub) < 3:
        return None
    for c in ["threshold", "mean_pnl_R"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["threshold", "mean_pnl_R"]).sort_values("threshold")
    if len(sub) < 3:
        return None
    fig = plt.figure(figsize=(9, 3.8))
    plt.plot(sub["threshold"], sub["mean_pnl_R"], marker="o")
    plt.title(f"Trade quality vs confidence threshold (scope={scope})")
    plt.xlabel("Threshold")
    plt.ylabel("Mean pnl_R of selected trades")
    return _fig_to_data_uri(fig)


def plot_reliability(cal_bins: pd.DataFrame, title: str) -> Optional[str]:
    if not HAS_PLT or cal_bins is None or len(cal_bins) == 0:
        return None
    needed = {"variant", "mean_p", "obs_rate"}
    if not needed.issubset(set(cal_bins.columns)):
        return None
    fig = plt.figure(figsize=(9, 3.8))
    for v, sub in cal_bins.groupby("variant"):
        x = pd.to_numeric(sub["mean_p"], errors="coerce")
        y = pd.to_numeric(sub["obs_rate"], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 3:
            continue
        plt.plot(x[m], y[m], marker="o", linewidth=1.0, label=str(v))
    plt.plot([0, 1], [0, 1], linewidth=1.0)
    plt.title(title)
    plt.xlabel("Predicted probability (binned)")
    plt.ylabel("Observed success rate (binned)")
    plt.legend()
    return _fig_to_data_uri(fig)


def plot_sizing_curve(sc: pd.DataFrame) -> Optional[str]:
    if not HAS_PLT or sc is None or len(sc) == 0:
        return None
    needed = {"mean_score", "risk_multiplier"}
    if not needed.issubset(set(sc.columns)):
        return None
    x = pd.to_numeric(sc["mean_score"], errors="coerce")
    y = pd.to_numeric(sc["risk_multiplier"], errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return None
    fig = plt.figure(figsize=(9, 3.8))
    plt.plot(x[m], y[m], marker="o", linewidth=1.0)
    plt.title("Position sizing curve (risk multiplier vs model confidence)")
    plt.xlabel("Calibrated probability (mean per bin)")
    plt.ylabel("Risk multiplier")
    return _fig_to_data_uri(fig)


# -----------------------------
# HTML assembly
# -----------------------------
def build_html(
    title: str,
    trades: pd.DataFrame,
    policy: Dict[str, object],
    deploy_cfg_path: Optional[str],
    regimes_diag: List[str],
    risk_col: Optional[str],
    risk_notes: List[str],
    risk_split: Optional[pd.DataFrame],
    rs_table: Optional[pd.DataFrame],
    step05: Dict[str, Optional[pd.DataFrame]],
    charts: Dict[str, Optional[str]],
    score_cov: Dict[str, object],
) -> str:
    # Snapshot
    n_trades = int(len(trades))
    n_syms = int(trades["symbol"].nunique()) if "symbol" in trades.columns else None
    dt_min = trades["entry_ts"].min() if "entry_ts" in trades.columns else pd.NaT
    dt_max = trades["entry_ts"].max() if "entry_ts" in trades.columns else pd.NaT

    win_rate = None
    if "WIN" in trades.columns:
        win_rate = float((_safe_num(trades["WIN"]) == 1).mean())
    elif "pnl_R" in trades.columns:
        win_rate = float((_safe_num(trades["pnl_R"]) > 0).mean())

    time_rate = None
    if "EXIT_FINAL" in trades.columns:
        time_rate = float((_safe_num(trades["EXIT_FINAL"]) == 2).mean())

    mean_pnl_R = float(_safe_num(trades["pnl_R"]).mean()) if "pnl_R" in trades.columns else None

    bullets = [
        f"Trades analyzed: <strong>{_fmt_int(n_trades)}</strong> across <strong>{_fmt_int(n_syms)}</strong> symbols.",
        f"Date range: <strong>{_fmt_date(dt_min)}</strong> to <strong>{_fmt_date(dt_max)}</strong>.",
        f"Win rate (simple): <strong>{_fmt_pct(win_rate)}</strong>.",
        f"Average outcome per trade (pnl_R): <strong>{_fmt_float(mean_pnl_R, 4)}</strong>.",
        f"Time-exit share (stale trades): <strong>{_fmt_pct(time_rate)}</strong>.",
    ]
    if policy.get("scope"):
        bullets.append(f"Deployment scope: <strong>{policy.get('scope')}</strong>.")
    if policy.get("threshold") is not None:
        bullets.append(f"Exported threshold: <strong>{_fmt_float(policy.get('threshold'), 2)}</strong>.")

    exec_summary = _info_box("Executive summary", bullets)

    plain = _info_box(
        "In plain English: what the system does",
        [
            "A model assigns each trade setup a score from 0 to 1 (higher = better).",
            "Calibration makes the score behave like a probability (e.g., ~0.70 means ‘about 70% of similar trades were good’).",
            "A threshold decides which trades to take (below threshold: skip or trade very small).",
            "A sizing curve scales position size based on confidence.",
        ],
    )

    # Policy block
    policy_html = ""
    if isinstance(policy, dict):
        policy_html = f"""
        <p><strong>Model column:</strong> {policy.get("model_col") or "—"}</p>
        <p><strong>Model name:</strong> {policy.get("model_name") or "—"}</p>
        <p><strong>Calibration:</strong> {policy.get("calib_method") or "—"}</p>
        <p><strong>Scope:</strong> <code>{policy.get("scope") or "ALL"}</code></p>
        <p><strong>Threshold:</strong> <strong>{_fmt_float(policy.get("threshold"), 2)}</strong></p>
        """
        if deploy_cfg_path:
            policy_html += f"<p class='note'>Source: {deploy_cfg_path}</p>"
    else:
        policy_html = _warn("Deployment config not found; policy summary unavailable.")

    # Risk-on block
    risk_html = ""
    if risk_col and (risk_split is not None) and len(risk_split) > 0:
        risk_html += f"<p><strong>Risk-on flag used:</strong> <code>{risk_col}</code></p>"
        risk_html += _df_to_html_table(risk_split, title="Performance split: RISK_ON vs NOT_RISK_ON")
    else:
        risk_html += _warn("Risk-on split not available in this report.")
        risk_html += "<ul>"
        risk_html += f"<li><strong>Regime merge diagnostics:</strong> {' | '.join(regimes_diag) if regimes_diag else 'n/a'}</li>"
        risk_html += f"<li><strong>Risk flag selection:</strong> {' | '.join(risk_notes) if risk_notes else 'n/a'}</li>"
        risk_html += "</ul>"

    # RS block
    rs_html = _df_to_html_table(rs_table, title="Performance split by RS bucket") if rs_table is not None else "<p class='note'>rs_pct not present.</p>"

    # Step 05 plots/tables
    cal_bins = step05.get("calibration_bins")
    ev = step05.get("ev_thresholds")
    sc = step05.get("sizing_curve")

    cal_plot = charts.get("reliability")
    ev_all_plot = charts.get("ev_all")
    ev_scope_plot = charts.get("ev_scope")
    sizing_plot = charts.get("sizing")
    pnl_plot = charts.get("pnl_hist")

    cal_html = ""
    if cal_plot:
        cal_html += f"<img class='chart' src='{cal_plot}'/>"
    cal_html += _df_to_html_table(cal_bins, title="Calibration bins (excerpt)", max_rows=12) if isinstance(cal_bins, pd.DataFrame) else _warn("Calibration bins CSV not found in Step 05 outputs.")

    ev_html = ""
    if ev_all_plot:
        ev_html += f"<img class='chart' src='{ev_all_plot}'/>"
    if ev_scope_plot:
        ev_html += f"<img class='chart' src='{ev_scope_plot}'/>"
    ev_html += _df_to_html_table(ev, title="EV thresholds table (excerpt)", max_rows=15) if isinstance(ev, pd.DataFrame) else _warn("EV thresholds CSV not found in Step 05 outputs.")

    sizing_html = ""
    if sizing_plot:
        sizing_html += f"<img class='chart' src='{sizing_plot}'/>"
    sizing_html += _df_to_html_table(sc, title="Sizing curve (excerpt)", max_rows=15) if isinstance(sc, pd.DataFrame) else _warn("Sizing curve CSV not found in Step 05 outputs.")

    # Scoring coverage block
    cov_html = _info_box(
        "Scoring coverage (critical operational check)",
        [
            f"Trades: <strong>{_fmt_int(score_cov.get('trades_n'))}</strong>",
            f"OOF loaded: <strong>{score_cov.get('oof_loaded')}</strong> (rows: <strong>{_fmt_int(score_cov.get('oof_n'))}</strong>)",
            f"OOF non-null score rate: <strong>{_fmt_pct(score_cov.get('oof_score_nonnull_rate'))}</strong>",
            f"Matched trades rate: <strong>{_fmt_pct(score_cov.get('matched_trades_rate'))}</strong>",
            f"Scored trades rate: <strong>{_fmt_pct(score_cov.get('scored_trades_rate'))}</strong>",
            f"Unscored trades: <strong>{_fmt_int(score_cov.get('unscored_n'))}</strong> (see <code>unscored_trades.csv</code> if generated)",
        ],
    )
    if score_cov.get("note"):
        cov_html += _warn(str(score_cov.get("note")))

    # Dataset plot
    pnl_html = ""
    if pnl_plot:
        pnl_html += f"<img class='chart' src='{pnl_plot}'/>"
    else:
        pnl_html += "<p class='note'>PnL histogram not available (matplotlib missing or too few trades).</p>"

    # Assemble
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --card: #121925;
      --text: #e7eef7;
      --muted: #a9b7c7;
      --accent: #67b0ff;
      --border: rgba(255,255,255,0.10);
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 24px;
      line-height: 1.45;
    }}
    .header {{
      margin-bottom: 16px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }}
    .header h1 {{
      margin: 0 0 6px 0;
      font-size: 26px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 13px;
    }}
    .info {{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      padding: 14px 16px;
      border-radius: 14px;
      margin: 14px 0;
    }}
    .info-title {{
      font-weight: 700;
      margin-bottom: 8px;
      color: var(--accent);
    }}
    .warn {{
      border: 1px solid rgba(255,204,102,0.35);
      background: rgba(255,204,102,0.08);
      padding: 12px 14px;
      border-radius: 12px;
      margin: 10px 0;
      color: var(--text);
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    .card {{
      border: 1px solid var(--border);
      background: var(--card);
      padding: 16px 16px;
      border-radius: 16px;
      margin: 16px 0;
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .chart {{
      display: block;
      max-width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--border);
      margin: 10px 0;
      background: #0a0e13;
    }}
    code {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <div class="sub">Generated: {_utc_now_str()}</div>
  </div>

  {exec_summary}
  {plain}

  {_card("What policy is being deployed (the simple rule)", policy_html)}
  {_card("Scoring coverage (do we actually score all trades?)", cov_html)}
  {_card("Data snapshot (pnl_R distribution)", pnl_html)}
  {_card("Market context: risk_on and RS effects", risk_html + rs_html)}
  {_card("Calibration (Step 05)", cal_html)}
  {_card("Threshold EV (Step 05)", ev_html)}
  {_card("Sizing curve (Step 05)", sizing_html)}

</body>
</html>
"""
    return html


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=str, default="results/trades.clean.csv")
    p.add_argument("--root", type=str, default="research_outputs")
    p.add_argument("--outdir", type=str, default="research_outputs/EXEC_REPORT")
    p.add_argument("--title", type=str, default="Trading Meta-Model Executive Report")
    p.add_argument("--model-col", type=str, default="", help="Override model probability column (e.g., p_lgbm)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    trades_csv = Path(args.trades)
    regimes_pq = root / "02_regimes" / "regimes.parquet"
    cvdir = root / "04_models_cv"
    caldir = root / "05_calibration_ev"
    deploydir = root / "07_deployment_artifacts"

    trades = load_trades(trades_csv)

    regimes = load_regimes(regimes_pq) if regimes_pq.exists() else None
    trades2, regimes_diag = merge_regimes_into_trades(trades, regimes)

    deploy_cfg, deploy_path = load_deployment_config(deploydir)
    policy = extract_policy_from_config(deploy_cfg)

    # choose model_col
    model_col = (args.model_col.strip() or (policy.get("model_col") or "")).strip()
    if not model_col:
        # fallback: try likely default
        model_col = "p_lgbm"
    policy["model_col"] = model_col

    # risk flag selection
    risk_col, risk_notes = pick_risk_flag_column(trades2, str(policy.get("scope") or "ALL"))
    risk_split = None
    if risk_col and risk_col in trades2.columns and trades2[risk_col].notna().any():
        risk_split = compute_risk_split(trades2, risk_col)

    rs_table = compute_rs_buckets(trades2)

    # Step 05 tables (direct)
    step05 = load_step05_tables(caldir, model_col)

    # Charts
    charts: Dict[str, Optional[str]] = {}
    charts["pnl_hist"] = plot_pnl_hist(trades2)

    if isinstance(step05.get("calibration_bins"), pd.DataFrame):
        charts["reliability"] = plot_reliability(step05["calibration_bins"], f"Calibration check for {model_col}")

    ev = step05.get("ev_thresholds")
    scope = str(policy.get("scope") or "ALL")
    if isinstance(ev, pd.DataFrame):
        charts["ev_all"] = plot_ev_curve(ev, "ALL")
        # if exported scope exists in table, plot it too
        if "scope" in ev.columns and (ev["scope"].astype(str) == scope).any():
            charts["ev_scope"] = plot_ev_curve(ev, scope)
        else:
            charts["ev_scope"] = None

    if isinstance(step05.get("sizing_curve"), pd.DataFrame):
        charts["sizing"] = plot_sizing_curve(step05["sizing_curve"])

    # Score coverage: compare trades vs oof predictions
    oof = load_cv_oof(cvdir)
    score_cov, _ = compute_score_coverage(trades2, oof, model_col, outdir)

    # Debug JSON
    debug = {
        "policy": policy,
        "deploy_config_path": deploy_path,
        "regimes_merge_diag": regimes_diag,
        "risk_flag": risk_col,
        "risk_notes": risk_notes,
        "score_coverage": score_cov,
    }
    (outdir / "report_debug.json").write_text(json.dumps(debug, indent=2, default=str), encoding="utf-8")

    # HTML
    html = build_html(
        title=args.title,
        trades=trades2,
        policy=policy,
        deploy_cfg_path=deploy_path,
        regimes_diag=regimes_diag,
        risk_col=risk_col,
        risk_notes=risk_notes,
        risk_split=risk_split,
        rs_table=rs_table,
        step05=step05,
        charts=charts,
        score_cov=score_cov,
    )
    out_html = outdir / "report.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"[08_exec_report] DONE. Wrote: {out_html}")
    if (outdir / "unscored_trades.csv").exists():
        print(f"[08_exec_report] Unscored trades written: {outdir / 'unscored_trades.csv'}")
    print(f"[08_exec_report] Debug: {outdir / 'report_debug.json'}")


if __name__ == "__main__":
    main()