#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a human-readable cross-run sweep dashboard with comparison charts."
    )
    p.add_argument("--sweeps-root", default="results/policy_sweeps")
    p.add_argument("--outdir", default="results/policy_sweeps/_reports")
    p.add_argument("--topn", type=int, default=20)
    p.add_argument(
        "--lam",
        type=float,
        default=2.0,
        help="Utility lambda if utility is missing in source summaries.",
    )
    p.add_argument(
        "--mu",
        type=float,
        default=1.0,
        help="Utility mu if utility is missing in source summaries.",
    )
    p.add_argument(
        "--include-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include run dirs even when summary.csv is missing.",
    )
    p.add_argument(
        "--run-include-regex",
        default="",
        help="Optional regex: include only run_id values matching this pattern.",
    )
    p.add_argument(
        "--run-exclude-regex",
        default="",
        help="Optional regex: exclude run_id values matching this pattern.",
    )
    p.add_argument(
        "--min-ok-variants",
        type=int,
        default=0,
        help="Optional filter: include only runs with at least this many status=ok variants.",
    )
    return p.parse_args()


def _run_id_ts(run_id: str, fallback: float) -> pd.Timestamp:
    # Expected format from sweep tools: YYYYMMDD_HHMMSS
    try:
        return pd.Timestamp(datetime.strptime(run_id, "%Y%m%d_%H%M%S"), tz="UTC")
    except Exception:
        return pd.Timestamp(datetime.fromtimestamp(fallback, tz=timezone.utc))


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_setting_tokens(setting: str) -> Dict[str, str]:
    parts = str(setting).split("__")
    out: Dict[str, str] = {
        "meta_mode": "",
        "blockdown": "",
        "probe": "",
        "slope": "",
    }
    for p in parts:
        pl = p.lower()
        if pl.startswith("no_meta_gate") or pl.startswith("gate_with_pstar"):
            out["meta_mode"] = p
        elif pl.startswith("blockdown_"):
            out["blockdown"] = p
        elif pl.startswith("probe_") or pl.startswith("probe"):
            out["probe"] = p
        elif pl.startswith("slope_") or pl.startswith("slope"):
            out["slope"] = p
    return out


def _without_probe(setting: str) -> str:
    parts = str(setting).split("__")
    keep = []
    for p in parts:
        pl = p.lower()
        if pl.startswith("probe_") or pl.startswith("probe"):
            continue
        keep.append(p)
    return "__".join(keep)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_rows_from_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in sorted([p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        met = d / "metrics.json"
        if not met.exists():
            rows.append({"setting": d.name, "status": "missing_metrics"})
            continue
        try:
            obj = _read_json(met)
        except Exception:
            obj = {"setting": d.name, "status": "bad_metrics_json"}
        obj.setdefault("setting", d.name)
        rows.append(obj)
    return rows


def _load_run_variants(run_dir: Path, lam: float, mu: float) -> pd.DataFrame:
    summary_recomputed = run_dir / "summary.recomputed.csv"
    summary = summary_recomputed if summary_recomputed.exists() else (run_dir / "summary.csv")
    summary_source = summary.name if summary.exists() else ""
    if summary.exists():
        try:
            df = pd.read_csv(summary)
        except Exception:
            df = pd.DataFrame(_variant_rows_from_metrics(run_dir))
    else:
        df = pd.DataFrame(_variant_rows_from_metrics(run_dir))

    if "setting" not in df.columns:
        # fallback: infer setting from directories
        settings = [d.name for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        df["setting"] = settings[: len(df)] if len(settings) >= len(df) else df.index.astype(str)

    # normalize common fields
    for c in ["total_pnl", "max_drawdown", "number_of_trades", "win_rate", "tail_loss_p05", "utility"]:
        if c in df.columns:
            df[c] = _to_num(df[c])
    if "status" not in df.columns:
        df["status"] = "ok"
    df["status"] = df["status"].astype(str)
    df["summary_source"] = summary_source

    if "pnl_col" not in df.columns:
        df["pnl_col"] = ""
    df["pnl_col"] = df["pnl_col"].fillna("").astype(str)
    df["total_pnl_unit"] = np.where(df["pnl_col"].str.lower().eq("pnl_r"), "R", "cash")

    # recompute utility when missing
    if "utility" not in df.columns:
        df["utility"] = np.nan
    need_u = df["utility"].isna()
    req = {"pnl_risk_on", "pnl_risk_off", "max_drawdown"}
    if need_u.any() and req.issubset(set(df.columns)):
        pon = _to_num(df["pnl_risk_on"])
        poff = _to_num(df["pnl_risk_off"])
        mdd = _to_num(df["max_drawdown"])
        util = pon - lam * poff.clip(upper=0).abs() - mu * mdd.clip(upper=0).abs()
        df.loc[need_u, "utility"] = util.loc[need_u]

    toks = df["setting"].astype(str).apply(_parse_setting_tokens)
    df["meta_mode"] = toks.apply(lambda d: d["meta_mode"])
    df["blockdown"] = toks.apply(lambda d: d["blockdown"])
    df["probe"] = toks.apply(lambda d: d["probe"])
    df["slope"] = toks.apply(lambda d: d["slope"])
    df["setting_no_probe"] = df["setting"].astype(str).apply(_without_probe)
    return df


def _probe_invariance_ratio(df: pd.DataFrame) -> float:
    # fraction of no-probe groups where all probe variants produced same pnl+tradecount
    req = {"setting_no_probe", "setting", "total_pnl", "number_of_trades"}
    if not req.issubset(df.columns):
        return float("nan")

    g = df.dropna(subset=["total_pnl"]).groupby("setting_no_probe", as_index=False)
    total = 0
    invariant = 0
    for _, sub in g:
        # only meaningful if multiple probe variants
        probes = sub["setting"].astype(str).str.contains("probe", case=False, na=False).sum()
        if len(sub) < 2 or probes < 2:
            continue
        total += 1
        key = (
            sub["total_pnl"].round(8).astype(str)
            + "|"
            + _to_num(sub["number_of_trades"]).round(8).astype(str)
        )
        if key.nunique(dropna=True) <= 1:
            invariant += 1
    if total == 0:
        return float("nan")
    return float(invariant) / float(total)


def _run_enrich_counts(run_dir: Path) -> Tuple[int, int, int]:
    p = run_dir / "enrichment_summary.csv"
    if not p.exists():
        return 0, 0, 0
    try:
        df = pd.read_csv(p)
    except Exception:
        return 0, 0, 0
    st = df.get("status", pd.Series(dtype=str)).astype(str)
    ok = int((st == "ok").sum())
    err = int((st == "error").sum())
    skipped = int((st.str.startswith("skipped")).sum())
    return ok, err, skipped


def _scan_runs(
    root: Path,
    lam: float,
    mu: float,
    include_partial: bool,
    run_include_regex: str = "",
    run_exclude_regex: str = "",
    min_ok_variants: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_variants: List[pd.DataFrame] = []
    run_rows: List[Dict[str, Any]] = []

    include_pat = re.compile(run_include_regex) if str(run_include_regex).strip() else None
    exclude_pat = re.compile(run_exclude_regex) if str(run_exclude_regex).strip() else None

    if not root.exists():
        raise FileNotFoundError(f"sweeps root not found: {root}")

    for run_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        rid = run_dir.name
        if include_pat is not None and not include_pat.search(rid):
            continue
        if exclude_pat is not None and exclude_pat.search(rid):
            continue

        summary_exists = (run_dir / "summary.csv").exists()
        if (not include_partial) and (not summary_exists):
            continue

        stage = ""
        stage_p = run_dir / "_STAGE.txt"
        if stage_p.exists():
            try:
                stage = stage_p.read_text(encoding="utf-8").strip()
            except Exception:
                stage = "<unreadable>"

        df = _load_run_variants(run_dir, lam=lam, mu=mu)
        if df.empty:
            continue
        df["run_id"] = run_dir.name
        df["run_dir"] = str(run_dir)

        run_ts = _run_id_ts(run_dir.name, run_dir.stat().st_mtime)
        df["run_ts"] = run_ts
        all_variants.append(df)

        ok_mask = df["status"].astype(str).str.lower().eq("ok")
        ok_df = df.loc[ok_mask].copy()
        if int(min_ok_variants) > 0 and int(ok_mask.sum()) < int(min_ok_variants):
            continue

        best_util_setting = ""
        best_util = float("nan")
        best_pnl_setting = ""
        best_pnl = float("nan")
        if not ok_df.empty:
            if "utility" in ok_df.columns and ok_df["utility"].notna().any():
                s = ok_df.sort_values("utility", ascending=False).iloc[0]
                best_util_setting = str(s.get("setting", ""))
                best_util = float(s.get("utility", np.nan))
            if "total_pnl" in ok_df.columns and ok_df["total_pnl"].notna().any():
                s = ok_df.sort_values("total_pnl", ascending=False).iloc[0]
                best_pnl_setting = str(s.get("setting", ""))
                best_pnl = float(s.get("total_pnl", np.nan))

        enrich_ok, enrich_err, enrich_skip = _run_enrich_counts(run_dir)
        run_rows.append(
            {
                "run_id": run_dir.name,
                "run_ts": run_ts,
                "summary_exists": bool(summary_exists),
                "summary_source": str(df.get("summary_source", pd.Series([""])).iloc[0]) if len(df) else "",
                "stage": stage,
                "n_variants": int(len(df)),
                "n_ok": int(ok_mask.sum()),
                "n_error": int((df["status"].astype(str).str.lower() == "error").sum()),
                "best_utility": best_util,
                "best_utility_setting": best_util_setting,
                "best_total_pnl": best_pnl,
                "best_total_pnl_setting": best_pnl_setting,
                "median_total_pnl": float(_to_num(ok_df.get("total_pnl", pd.Series(dtype=float))).median())
                if not ok_df.empty
                else float("nan"),
                "median_max_drawdown": float(_to_num(ok_df.get("max_drawdown", pd.Series(dtype=float))).median())
                if not ok_df.empty
                else float("nan"),
                "pnl_unit_mode": (
                    ok_df["total_pnl_unit"].mode(dropna=True).iloc[0]
                    if (not ok_df.empty and "total_pnl_unit" in ok_df.columns and not ok_df["total_pnl_unit"].mode(dropna=True).empty)
                    else ""
                ),
                "probe_invariance_ratio": _probe_invariance_ratio(ok_df),
                "enrich_ok": enrich_ok,
                "enrich_error": enrich_err,
                "enrich_skipped": enrich_skip,
            }
        )

    if not all_variants:
        return pd.DataFrame(), pd.DataFrame()
    all_df = pd.concat(all_variants, ignore_index=True)
    runs_df = pd.DataFrame(run_rows).sort_values("run_ts")
    return all_df, runs_df


def _plot_best_by_run(runs_df: pd.DataFrame, out_path: Path) -> None:
    df = runs_df.copy().sort_values("run_ts")
    if df.empty:
        return
    x = np.arange(len(df))
    labels = df["run_id"].astype(str).tolist()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].bar(x, _to_num(df["best_total_pnl"]).to_numpy(), color="#2b8cbe")
    axes[0].set_ylabel("Best Total PnL")
    axes[0].set_title("Best Variant Metrics By Run")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x, _to_num(df["best_utility"]).to_numpy(), color="#31a354")
    axes[1].set_ylabel("Best Utility")
    axes[1].grid(alpha=0.25, axis="y")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_variants(all_df: pd.DataFrame, out_path: Path) -> None:
    df = all_df.copy()
    ok = df["status"].astype(str).str.lower().eq("ok")
    df = df[ok]
    df["total_pnl"] = _to_num(df.get("total_pnl", pd.Series(dtype=float)))
    df["max_drawdown"] = _to_num(df.get("max_drawdown", pd.Series(dtype=float)))
    df = df.dropna(subset=["total_pnl", "max_drawdown"])
    if df.empty:
        return

    run_codes = {rid: i for i, rid in enumerate(sorted(df["run_id"].astype(str).unique()))}
    colors = df["run_id"].astype(str).map(run_codes).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(df["max_drawdown"], df["total_pnl"], c=colors, cmap="tab20", alpha=0.8, s=26)
    ax.set_title("Variant Comparison: Total PnL vs Max Drawdown")
    ax.set_xlabel("Max Drawdown (fraction, more negative is worse)")
    ax.set_ylabel("Total PnL")
    ax.grid(alpha=0.25)

    # annotate top variants by pnl
    top = df.sort_values("total_pnl", ascending=False).head(8)
    for _, r in top.iterrows():
        label = f"{r['run_id']}:{str(r['setting'])[:24]}"
        ax.annotate(
            label,
            (r["max_drawdown"], r["total_pnl"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    # lightweight legend from run codes
    inv = {v: k for k, v in run_codes.items()}
    handles = []
    labels = []
    for code in sorted(inv.keys())[:20]:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab20(code % 20), markersize=6))
        labels.append(inv[code])
    ax.legend(handles, labels, title="Run", fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_meta_mode_box(all_df: pd.DataFrame, out_path: Path) -> None:
    df = all_df.copy()
    ok = df["status"].astype(str).str.lower().eq("ok")
    df = df[ok]
    df["total_pnl"] = _to_num(df.get("total_pnl", pd.Series(dtype=float)))
    df = df.dropna(subset=["total_pnl"])
    if df.empty:
        return
    groups = []
    labels = []
    for mode in sorted([m for m in df["meta_mode"].dropna().unique() if str(m).strip()]):
        vals = _to_num(df.loc[df["meta_mode"] == mode, "total_pnl"]).dropna().to_numpy()
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(str(mode))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(groups, tick_labels=labels, showfliers=True)
    ax.set_title("Total PnL Distribution By Meta Mode")
    ax.set_ylabel("Total PnL")
    ax.grid(alpha=0.25, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_probe_invariance(runs_df: pd.DataFrame, out_path: Path) -> None:
    df = runs_df.copy().sort_values("run_ts")
    if df.empty:
        return
    y = _to_num(df["probe_invariance_ratio"]).to_numpy()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, y, color="#756bb1")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Probe-Dimension Invariance Ratio By Run (higher = suspicious)")
    ax.set_ylabel("Invariant Group Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(df["run_id"].astype(str), rotation=40, ha="right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_enrichment_status(runs_df: pd.DataFrame, out_path: Path) -> None:
    df = runs_df.copy().sort_values("run_ts")
    if df.empty:
        return
    x = np.arange(len(df))
    ok = _to_num(df["enrich_ok"]).fillna(0).to_numpy()
    err = _to_num(df["enrich_error"]).fillna(0).to_numpy()
    sk = _to_num(df["enrich_skipped"]).fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, ok, label="enrich_ok", color="#31a354")
    ax.bar(x, err, bottom=ok, label="enrich_error", color="#de2d26")
    ax.bar(x, sk, bottom=ok + err, label="enrich_skipped", color="#9ecae1")
    ax.set_title("Enrichment Status By Run")
    ax.set_ylabel("Variant Count")
    ax.set_xticks(x)
    ax.set_xticklabels(df["run_id"].astype(str), rotation=40, ha="right")
    ax.legend(loc="best")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return ""
    return f"{f:.{nd}f}"


def _render_html(
    out_html: Path,
    assets_rel: str,
    all_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    topn: int,
    sweeps_root: Path,
) -> None:
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    n_runs = len(runs_df)
    n_variants = len(all_df)
    n_complete = int(runs_df["summary_exists"].fillna(False).sum()) if not runs_df.empty else 0
    n_warn_probe = int((_to_num(runs_df["probe_invariance_ratio"]) >= 0.9).sum()) if not runs_df.empty else 0

    ok_df = all_df[all_df["status"].astype(str).str.lower().eq("ok")].copy()
    ok_df["utility"] = _to_num(ok_df.get("utility", pd.Series(dtype=float)))
    ok_df["total_pnl"] = _to_num(ok_df.get("total_pnl", pd.Series(dtype=float)))
    ok_df["max_drawdown"] = _to_num(ok_df.get("max_drawdown", pd.Series(dtype=float)))
    top_util = ok_df.sort_values("utility", ascending=False).head(topn)
    top_pnl = ok_df.sort_values("total_pnl", ascending=False).head(topn)

    run_cols = [
        "run_id",
        "run_ts",
        "pnl_unit_mode",
        "summary_source",
        "n_variants",
        "n_ok",
        "n_error",
        "best_total_pnl",
        "best_utility",
        "median_total_pnl",
        "median_max_drawdown",
        "probe_invariance_ratio",
        "enrich_ok",
        "enrich_error",
        "stage",
    ]
    run_tbl = runs_df[run_cols].copy() if not runs_df.empty else pd.DataFrame(columns=run_cols)

    def _tbl(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p><em>No data.</em></p>"
        return df.to_html(index=False, classes="tbl", border=0)

    # compact/pretty formatting
    if "run_ts" in run_tbl.columns:
        run_tbl["run_ts"] = pd.to_datetime(run_tbl["run_ts"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    for c in ["best_total_pnl", "best_utility", "median_total_pnl", "median_max_drawdown", "probe_invariance_ratio"]:
        if c in run_tbl.columns:
            run_tbl[c] = run_tbl[c].map(lambda x: _fmt(x, 4))
    for c in ["total_pnl", "utility", "max_drawdown", "win_rate", "tail_loss_p05"]:
        if c in top_util.columns:
            top_util[c] = top_util[c].map(lambda x: _fmt(x, 4))
        if c in top_pnl.columns:
            top_pnl[c] = top_pnl[c].map(lambda x: _fmt(x, 4))

    if "run_ts" in top_util.columns:
        top_util["run_ts"] = pd.to_datetime(top_util["run_ts"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    if "run_ts" in top_pnl.columns:
        top_pnl["run_ts"] = pd.to_datetime(top_pnl["run_ts"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    probe_warn = runs_df[_to_num(runs_df.get("probe_invariance_ratio", pd.Series(dtype=float))) >= 0.9] if not runs_df.empty else pd.DataFrame()
    partial_runs = runs_df[~runs_df["summary_exists"].astype(bool)] if not runs_df.empty else pd.DataFrame()

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Policy Sweep Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin: 0.4em 0; }}
    .meta {{ color: #444; margin-bottom: 14px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; margin: 14px 0 18px 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; background: #fafafa; }}
    .card .k {{ color: #555; font-size: 12px; }}
    .card .v {{ font-size: 22px; font-weight: 700; }}
    .section {{ margin-top: 24px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .plot {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; background: #fff; }}
    img {{ width: 100%; height: auto; }}
    .tbl {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .tbl th, .tbl td {{ border: 1px solid #e3e3e3; padding: 6px 8px; text-align: left; }}
    .tbl th {{ background: #f3f6f9; position: sticky; top: 0; }}
    .warn {{ background: #fff4e5; border: 1px solid #ffd59e; border-radius: 8px; padding: 10px 12px; }}
    .ok {{ background: #eefaf0; border: 1px solid #b7e4be; border-radius: 8px; padding: 10px 12px; }}
    .small {{ font-size: 12px; color: #666; }}
  </style>
</head>
<body>
  <h1>Policy Sweep Cross-Run Dashboard</h1>
  <div class="meta">Generated: {now} | Root: {sweeps_root}</div>
  <div class="small">Note: <code>run_id</code> is the folder name and may encode when the run started; use <code>run_ts</code> as the actual timestamp shown by this report.</div>
  <div class="small"><strong>Important:</strong> <code>total_pnl</code> is interpreted in the run's <code>pnl_unit_mode</code> (cash or R). Do not compare cash and R directly.</div>
  <div class="cards">
    <div class="card"><div class="k">Runs scanned</div><div class="v">{n_runs}</div></div>
    <div class="card"><div class="k">Complete runs</div><div class="v">{n_complete}</div></div>
    <div class="card"><div class="k">Variants scanned</div><div class="v">{n_variants}</div></div>
    <div class="card"><div class="k">Probe warnings</div><div class="v">{n_warn_probe}</div></div>
  </div>

  <div class="section">
    <h2>Run Health</h2>
    {_tbl(run_tbl)}
  </div>

  <div class="section grid2">
    <div class="plot"><img src="{assets_rel}/best_by_run.png" alt="Best by run"></div>
    <div class="plot"><img src="{assets_rel}/probe_invariance.png" alt="Probe invariance"></div>
  </div>
  <div class="section grid2">
    <div class="plot"><img src="{assets_rel}/variant_scatter.png" alt="Variant scatter"></div>
    <div class="plot"><img src="{assets_rel}/meta_mode_box.png" alt="Meta mode box"></div>
  </div>
  <div class="section">
    <div class="plot"><img src="{assets_rel}/enrichment_status.png" alt="Enrichment status"></div>
  </div>

  <div class="section">
    <h2>Top Variants By Utility (All Runs)</h2>
    {_tbl(top_util[[c for c in ["run_id","run_ts","setting","total_pnl_unit","utility","total_pnl","max_drawdown","number_of_trades","win_rate","tail_loss_p05","status"] if c in top_util.columns]])}
  </div>

  <div class="section">
    <h2>Top Variants By Total PnL (All Runs)</h2>
    {_tbl(top_pnl[[c for c in ["run_id","run_ts","setting","total_pnl_unit","total_pnl","utility","max_drawdown","number_of_trades","win_rate","tail_loss_p05","status"] if c in top_pnl.columns]])}
  </div>

  <div class="section">
    <h2>Warnings</h2>
    <div class="warn">
      <strong>Probe invariance suspicious runs (ratio ≥ 0.90):</strong>
      {", ".join(probe_warn["run_id"].astype(str).tolist()) if not probe_warn.empty else "none"}
    </div>
    <br/>
    <div class="warn">
      <strong>Partial/in-progress runs:</strong>
      {", ".join(partial_runs["run_id"].astype(str).tolist()) if not partial_runs.empty else "none"}
    </div>
  </div>

  <div class="section small">
    <p>Files exported alongside this report:</p>
    <ul>
      <li>runs_overview.csv</li>
      <li>variants_all.csv</li>
      <li>variants_top_utility.csv</li>
      <li>variants_top_total_pnl.csv</li>
    </ul>
  </div>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> int:
    a = parse_args()
    sweeps_root = Path(a.sweeps_root).resolve()
    outdir = Path(a.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    assets = outdir / "all_runs_assets"
    assets.mkdir(parents=True, exist_ok=True)

    all_df, runs_df = _scan_runs(
        sweeps_root,
        lam=float(a.lam),
        mu=float(a.mu),
        include_partial=bool(a.include_partial),
        run_include_regex=str(a.run_include_regex),
        run_exclude_regex=str(a.run_exclude_regex),
        min_ok_variants=int(a.min_ok_variants),
    )
    if all_df.empty or runs_df.empty:
        raise RuntimeError(f"No sweep data found under {sweeps_root}")

    # Write machine-readable outputs
    all_df.to_csv(outdir / "variants_all.csv", index=False)
    runs_df.to_csv(outdir / "runs_overview.csv", index=False)
    ok_df = all_df[all_df["status"].astype(str).str.lower().eq("ok")].copy()
    ok_df["utility"] = _to_num(ok_df.get("utility", pd.Series(dtype=float)))
    ok_df["total_pnl"] = _to_num(ok_df.get("total_pnl", pd.Series(dtype=float)))
    ok_df.sort_values("utility", ascending=False).head(int(a.topn)).to_csv(outdir / "variants_top_utility.csv", index=False)
    ok_df.sort_values("total_pnl", ascending=False).head(int(a.topn)).to_csv(outdir / "variants_top_total_pnl.csv", index=False)

    # Plots
    _plot_best_by_run(runs_df, assets / "best_by_run.png")
    _plot_probe_invariance(runs_df, assets / "probe_invariance.png")
    _plot_scatter_variants(all_df, assets / "variant_scatter.png")
    _plot_meta_mode_box(all_df, assets / "meta_mode_box.png")
    _plot_enrichment_status(runs_df, assets / "enrichment_status.png")

    # HTML
    out_html = outdir / "all_runs_dashboard.html"
    _render_html(
        out_html=out_html,
        assets_rel="all_runs_assets",
        all_df=all_df,
        runs_df=runs_df,
        topn=int(a.topn),
        sweeps_root=sweeps_root,
    )

    print(str(out_html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
