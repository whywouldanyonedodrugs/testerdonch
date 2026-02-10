#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRADES_PATTERNS = [
    "trades.clean.csv",
    "trades.enriched.filled.csv",
    "trades.enriched.csv",
    "trades.csv",
]


@dataclass
class RunSeries:
    daily_pnl: pd.Series
    equity: pd.Series
    dd_pct: pd.Series
    monthly_pnl: pd.Series


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render an advanced risk/return HTML report for one sweep run."
    )
    p.add_argument("--run-id", required=True, help="Sweep run folder name under results/policy_sweeps.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--outdir", default="", help="Output dir (default: <run>/risk_report).")
    p.add_argument("--initial-capital", type=float, default=2000.0)
    p.add_argument("--topn-table", type=int, default=30)
    p.add_argument("--topn-plot", type=int, default=12)
    return p.parse_args()


def _safe_div(a: float, b: float) -> float:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return float("nan")
    if not np.isfinite(af) or not np.isfinite(bf) or abs(bf) <= 1e-12:
        return float("nan")
    return af / bf


def _fmt(v: object, nd: int = 4) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(f):
        return ""
    return f"{f:.{nd}f}"


def _find_trades_file(setting_dir: Path) -> Optional[Path]:
    for name in TRADES_PATTERNS:
        p = setting_dir / name
        if p.exists():
            return p
    return None


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _compute_metrics(
    setting: str,
    trades_path: Path,
    initial_capital: float,
) -> Tuple[Dict[str, object], Optional[RunSeries]]:
    try:
        tr = pd.read_csv(trades_path, low_memory=False)
    except Exception as e:
        return {
            "setting": setting,
            "status": f"read_error:{type(e).__name__}",
            "trades_path": str(trades_path),
        }, None

    cols = list(tr.columns)
    ts_col = _pick_col(cols, ["exit_ts", "timestamp", "entry_ts"])
    pnl_col = _pick_col(cols, ["pnl", "pnl_cash", "pnl_usd", "pnl_net", "pnl_after_fees"])
    if ts_col is None or pnl_col is None:
        return {
            "setting": setting,
            "status": "missing_ts_or_pnl_col",
            "trades_path": str(trades_path),
            "ts_col": ts_col,
            "pnl_col": pnl_col,
        }, None

    ts = pd.to_datetime(tr[ts_col], utc=True, errors="coerce")
    pnl = pd.to_numeric(tr[pnl_col], errors="coerce")
    m = pd.DataFrame({"ts": ts, "pnl": pnl}).dropna()
    if m.empty:
        return {
            "setting": setting,
            "status": "no_valid_rows",
            "trades_path": str(trades_path),
            "ts_col": ts_col,
            "pnl_col": pnl_col,
        }, None

    m = m.sort_values("ts", kind="mergesort")
    daily_pnl = m.groupby(m["ts"].dt.floor("D"), as_index=True)["pnl"].sum().sort_index()
    if daily_pnl.index.tz is None:
        daily_pnl.index = daily_pnl.index.tz_localize("UTC")

    # Force continuous daily timeline for robust DD/vol calculations
    idx = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D", tz="UTC")
    daily_pnl = daily_pnl.reindex(idx, fill_value=0.0)

    equity = float(initial_capital) + daily_pnl.cumsum()
    peak = equity.cummax()
    dd_abs = equity - peak
    dd_pct = (equity / peak) - 1.0
    max_dd_abs = float(dd_abs.min()) if len(dd_abs) else float("nan")
    max_dd_pct = float(dd_pct.min()) if len(dd_pct) else float("nan")

    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ann_factor = math.sqrt(252.0)
    mu = float(rets.mean()) if len(rets) else float("nan")
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    neg = rets[rets < 0]
    sd_down = float(neg.std(ddof=1)) if len(neg) > 1 else float("nan")
    sharpe = _safe_div(mu, sd) * ann_factor if np.isfinite(_safe_div(mu, sd)) else float("nan")
    sortino = _safe_div(mu, sd_down) * ann_factor if np.isfinite(_safe_div(mu, sd_down)) else float("nan")
    ann_vol = sd * ann_factor if np.isfinite(sd) else float("nan")

    # CAGR on equity (only valid with positive start/end equity)
    n_days = max((daily_pnl.index.max() - daily_pnl.index.min()).days, 1)
    years = n_days / 365.25
    eq0 = float(equity.iloc[0]) if len(equity) else float("nan")
    eq1 = float(equity.iloc[-1]) if len(equity) else float("nan")
    if eq0 > 0 and eq1 > 0 and years > 0:
        cagr = (eq1 / eq0) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")
    calmar = _safe_div(cagr, abs(max_dd_pct)) if np.isfinite(max_dd_pct) and max_dd_pct < 0 else float("nan")

    pnl_vals = m["pnl"]
    wins = pnl_vals[pnl_vals > 0]
    losses = pnl_vals[pnl_vals < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    profit_factor = _safe_div(gross_win, abs(gross_loss)) if abs(gross_loss) > 0 else float("nan")
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")
    payoff = _safe_div(avg_win, abs(avg_loss)) if np.isfinite(avg_loss) and avg_loss < 0 else float("nan")
    win_rate = float((pnl_vals > 0).mean())
    expectancy = float(pnl_vals.mean())

    var95 = float(np.nanpercentile(rets.values, 5)) if len(rets) else float("nan")
    cvar95 = float(rets[rets <= var95].mean()) if len(rets) else float("nan")
    ulcer = float(np.sqrt(np.mean(np.square(dd_pct.values)))) if len(dd_pct) else float("nan")
    recovery = _safe_div(float(pnl_vals.sum()), abs(max_dd_abs)) if np.isfinite(max_dd_abs) and max_dd_abs < 0 else float("nan")

    size_med = float("nan")
    size_p90 = float("nan")
    size_max = float("nan")
    if "size_mult" in tr.columns:
        s = pd.to_numeric(tr["size_mult"], errors="coerce").dropna()
        if len(s):
            size_med = float(s.median())
            size_p90 = float(s.quantile(0.9))
            size_max = float(s.max())

    risk_real_med = float("nan")
    risk_real_p90 = float("nan")
    risk_real_max = float("nan")
    if "risk_cash_realized" in tr.columns:
        r = pd.to_numeric(tr["risk_cash_realized"], errors="coerce").dropna()
        if len(r):
            risk_real_med = float(r.median())
            risk_real_p90 = float(r.quantile(0.9))
            risk_real_max = float(r.max())

    risk_on_frac = float("nan")
    if "risk_on" in tr.columns:
        ro = pd.to_numeric(tr["risk_on"], errors="coerce").dropna()
        if len(ro):
            risk_on_frac = float((ro == 1).mean())

    monthly = daily_pnl.resample("MS").sum()

    row: Dict[str, object] = {
        "setting": setting,
        "status": "ok",
        "trades_path": str(trades_path),
        "ts_col": ts_col,
        "pnl_col": pnl_col,
        "n_trades": int(len(pnl_vals)),
        "total_pnl": float(pnl_vals.sum()),
        "avg_trade_pnl": expectancy,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "profit_factor": profit_factor,
        "max_dd_abs": max_dd_abs,
        "max_dd_pct": max_dd_pct,
        "cagr": cagr,
        "calmar": calmar,
        "sharpe_daily": sharpe,
        "sortino_daily": sortino,
        "ann_vol_daily": ann_vol,
        "daily_var_95": var95,
        "daily_cvar_95": cvar95,
        "ulcer_index": ulcer,
        "recovery_factor": recovery,
        "size_med": size_med,
        "size_p90": size_p90,
        "size_max": size_max,
        "risk_real_med": risk_real_med,
        "risk_real_p90": risk_real_p90,
        "risk_real_max": risk_real_max,
        "risk_on_frac": risk_on_frac,
        "start_ts": str(m["ts"].iloc[0]),
        "end_ts": str(m["ts"].iloc[-1]),
    }
    series = RunSeries(
        daily_pnl=daily_pnl,
        equity=equity,
        dd_pct=dd_pct,
        monthly_pnl=monthly,
    )
    return row, series


def _plot_monthly_heatmap(monthly_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if monthly_df.empty:
        return
    fig_w = max(12, 0.38 * max(1, monthly_df.shape[1]))
    fig_h = max(6, 0.35 * max(1, monthly_df.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vals = monthly_df.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    vmax = max(vmax, 1e-9)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_yticks(np.arange(monthly_df.shape[0]))
    ax.set_yticklabels(monthly_df.index.tolist(), fontsize=8)
    ax.set_xticks(np.arange(monthly_df.shape[1]))
    ax.set_xticklabels(monthly_df.columns.tolist(), rotation=45, ha="right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Monthly PnL")
    # annotate only if not too dense
    if monthly_df.shape[0] <= 16 and monthly_df.shape[1] <= 24:
        for i in range(monthly_df.shape[0]):
            for j in range(monthly_df.shape[1]):
                v = vals[i, j]
                txt = f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_drawdowns(dd_map: Dict[str, pd.Series], settings: List[str], out_path: Path) -> None:
    if not settings:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in settings:
        ser = dd_map.get(s)
        if ser is None or ser.empty:
            continue
        ax.plot(ser.index, ser.values, linewidth=1.4, label=s)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Drawdown Curves (Top Settings)")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_equity(eq_map: Dict[str, pd.Series], settings: List[str], out_path: Path) -> None:
    if not settings:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in settings:
        ser = eq_map.get(s)
        if ser is None or ser.empty:
            continue
        base = float(ser.iloc[0])
        if not np.isfinite(base) or abs(base) <= 1e-12:
            continue
        norm = ser / base
        ax.plot(norm.index, norm.values, linewidth=1.4, label=s)
    ax.set_title("Normalized Equity Curves (Top Settings)")
    ax.set_ylabel("Equity / Start")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(metrics_df: pd.DataFrame, out_path: Path) -> None:
    df = metrics_df.copy()
    for c in ["sharpe_daily", "calmar", "total_pnl", "max_dd_pct"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df = df.dropna(subset=["sharpe_daily", "calmar", "total_pnl"])
    if df.empty:
        return
    sizes = 30 + 170 * (df["total_pnl"] - df["total_pnl"].min()) / max(df["total_pnl"].max() - df["total_pnl"].min(), 1e-9)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    sc = ax.scatter(df["max_dd_pct"], df["sharpe_daily"], c=df["calmar"], cmap="viridis", s=sizes, alpha=0.8)
    ax.set_title("Risk-Adjusted Map: Sharpe vs Max Drawdown (color=Calmar, size=Total PnL)")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Sharpe (daily, annualized)")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Calmar")
    top = df.sort_values("sharpe_daily", ascending=False).head(8)
    for _, r in top.iterrows():
        ax.annotate(str(r["setting"])[:26], (r["max_dd_pct"], r["sharpe_daily"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>No data.</em></p>"
    return df.to_html(index=False, classes="tbl", border=0)


def _render_html(
    out_html: Path,
    run_id: str,
    run_root: Path,
    metrics_df: pd.DataFrame,
    top_df: pd.DataFrame,
    cards: Dict[str, str],
    assets_rel: str,
    monthly_top_tbl: pd.DataFrame,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    # prettify numbers
    view_top = top_df.copy()
    num_cols = [
        "utility",
        "total_pnl",
        "sharpe_daily",
        "sortino_daily",
        "calmar",
        "max_dd_pct",
        "ann_vol_daily",
        "win_rate",
        "profit_factor",
        "n_trades",
        "risk_real_med",
        "risk_real_p90",
    ]
    for c in num_cols:
        if c in view_top.columns:
            view_top[c] = view_top[c].map(lambda x: _fmt(x, 4))
    view_top = view_top[
        [c for c in [
            "setting",
            "utility",
            "total_pnl",
            "sharpe_daily",
            "sortino_daily",
            "calmar",
            "max_dd_pct",
            "ann_vol_daily",
            "n_trades",
            "win_rate",
            "profit_factor",
            "risk_real_med",
            "risk_real_p90",
        ] if c in view_top.columns]
    ]

    m_tbl = monthly_top_tbl.copy()
    if not m_tbl.empty:
        for c in m_tbl.columns:
            m_tbl[c] = pd.to_numeric(m_tbl[c], errors="coerce").map(lambda x: _fmt(x, 0))
        m_tbl = m_tbl.reset_index().rename(columns={"index": "setting"})

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Sweep Risk Report - {run_id}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111; }}
    .meta {{ color: #444; margin-bottom: 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 10px; margin: 14px 0 18px 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; background: #fafafa; }}
    .card .k {{ color: #555; font-size: 12px; }}
    .card .v {{ font-size: 18px; font-weight: 700; }}
    .section {{ margin-top: 20px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .plot {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; background: #fff; }}
    img {{ width: 100%; height: auto; }}
    .tbl {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    .tbl th, .tbl td {{ border: 1px solid #e3e3e3; padding: 6px 8px; text-align: left; }}
    .tbl th {{ background: #f3f6f9; position: sticky; top: 0; }}
    .small {{ font-size: 12px; color: #666; }}
  </style>
</head>
<body>
  <h1>Sweep Risk Report</h1>
  <div class="meta">Run: <code>{run_id}</code> | Generated: {now} | Root: <code>{run_root}</code></div>
  <div class="small">Metrics are calculated from per-trade realized PnL with daily equity aggregation (based on trade exit timestamps).</div>
  <div class="cards">
    <div class="card"><div class="k">Variants analyzed</div><div class="v">{cards.get("n_variants","")}</div></div>
    <div class="card"><div class="k">Best Sharpe</div><div class="v">{cards.get("best_sharpe","")}</div></div>
    <div class="card"><div class="k">Best Calmar</div><div class="v">{cards.get("best_calmar","")}</div></div>
    <div class="card"><div class="k">Best Total PnL</div><div class="v">{cards.get("best_total_pnl","")}</div></div>
    <div class="card"><div class="k">Worst Max DD</div><div class="v">{cards.get("worst_dd","")}</div></div>
  </div>

  <div class="section grid2">
    <div class="plot"><img src="{assets_rel}/risk_scatter.png" alt="Risk scatter"></div>
    <div class="plot"><img src="{assets_rel}/monthly_heatmap_top.png" alt="Monthly heatmap top"></div>
  </div>
  <div class="section grid2">
    <div class="plot"><img src="{assets_rel}/equity_top.png" alt="Equity top"></div>
    <div class="plot"><img src="{assets_rel}/drawdown_top.png" alt="Drawdown top"></div>
  </div>
  <div class="section">
    <div class="plot"><img src="{assets_rel}/monthly_heatmap_all.png" alt="Monthly heatmap all"></div>
  </div>

  <div class="section">
    <h2>Top Settings (Risk + Return)</h2>
    {_table_html(view_top)}
  </div>

  <div class="section">
    <h2>Monthly PnL Table (Top Settings)</h2>
    {_table_html(m_tbl)}
  </div>

  <div class="section small">
    <p>Exported files:</p>
    <ul>
      <li>metrics_all.csv</li>
      <li>metrics_top.csv</li>
      <li>monthly_pnl_matrix.csv</li>
      <li>monthly_pnl_top.csv</li>
    </ul>
  </div>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> int:
    a = parse_args()
    run_root = Path(a.results_dir).resolve() / "policy_sweeps" / str(a.run_id)
    if not run_root.exists():
        raise RuntimeError(f"run folder not found: {run_root}")

    outdir = Path(a.outdir).resolve() if a.outdir else (run_root / "risk_report")
    outdir.mkdir(parents=True, exist_ok=True)
    assets = outdir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Optional: merge utility from summary
    util_map: Dict[str, float] = {}
    summary_path = run_root / "summary.csv"
    if summary_path.exists():
        try:
            s = pd.read_csv(summary_path)
            if "setting" in s.columns and "utility" in s.columns:
                for _, r in s.iterrows():
                    util_map[str(r["setting"])] = float(pd.to_numeric(r["utility"], errors="coerce"))
        except Exception:
            util_map = {}

    rows: List[Dict[str, object]] = []
    eq_map: Dict[str, pd.Series] = {}
    dd_map: Dict[str, pd.Series] = {}
    monthly_map: Dict[str, pd.Series] = {}

    setting_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and not p.name.startswith("_")])
    for d in setting_dirs:
        setting = d.name
        tpath = _find_trades_file(d)
        if tpath is None:
            rows.append({"setting": setting, "status": "no_trades_file"})
            continue
        row, series = _compute_metrics(setting=setting, trades_path=tpath, initial_capital=float(a.initial_capital))
        if setting in util_map:
            row["utility"] = util_map.get(setting, float("nan"))
        else:
            row["utility"] = float("nan")
        rows.append(row)
        if series is not None:
            eq_map[setting] = series.equity
            dd_map[setting] = series.dd_pct
            monthly_map[setting] = series.monthly_pnl

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(f"no metrics rows built for {run_root}")

    # Keep only analyzable rows for ranking/plots
    ok = metrics["status"].astype(str).str.lower().eq("ok")
    m_ok = metrics[ok].copy()
    if m_ok.empty:
        raise RuntimeError(f"no valid setting metrics (all rows failed) for {run_root}")

    for c in [
        "utility",
        "total_pnl",
        "sharpe_daily",
        "sortino_daily",
        "calmar",
        "max_dd_pct",
        "ann_vol_daily",
        "n_trades",
        "win_rate",
        "profit_factor",
        "risk_real_med",
        "risk_real_p90",
    ]:
        if c in m_ok.columns:
            m_ok[c] = pd.to_numeric(m_ok[c], errors="coerce")

    # Rank by utility when available, else Sharpe
    has_util = m_ok["utility"].notna().any() if "utility" in m_ok.columns else False
    rank_col = "utility" if has_util else "sharpe_daily"
    m_ok = m_ok.sort_values(rank_col, ascending=False, kind="mergesort")
    top = m_ok.head(int(a.topn_table)).copy()
    top_plot_settings = m_ok.head(int(a.topn_plot))["setting"].astype(str).tolist()

    # Monthly matrix
    month_keys: List[str] = sorted(
        {
            idx.strftime("%Y-%m")
            for ser in monthly_map.values()
            for idx in ser.index
        }
    )
    monthly_mat = pd.DataFrame(index=m_ok["setting"].astype(str).tolist(), columns=month_keys, dtype=float)
    for s_name, ser in monthly_map.items():
        for ts, v in ser.items():
            monthly_mat.loc[s_name, ts.strftime("%Y-%m")] = float(v)
    monthly_mat = monthly_mat.fillna(0.0)
    monthly_top = monthly_mat.loc[[s for s in top_plot_settings if s in monthly_mat.index]].copy()

    # Plots
    _plot_scatter(m_ok, assets / "risk_scatter.png")
    _plot_monthly_heatmap(monthly_top, assets / "monthly_heatmap_top.png", "Monthly PnL Heatmap (Top Settings)")
    _plot_monthly_heatmap(monthly_mat, assets / "monthly_heatmap_all.png", "Monthly PnL Heatmap (All Settings)")
    _plot_equity(eq_map, top_plot_settings, assets / "equity_top.png")
    _plot_drawdowns(dd_map, top_plot_settings, assets / "drawdown_top.png")

    # CSV outputs
    metrics.to_csv(outdir / "metrics_all.csv", index=False)
    top.to_csv(outdir / "metrics_top.csv", index=False)
    monthly_mat.to_csv(outdir / "monthly_pnl_matrix.csv", index=True)
    monthly_top.to_csv(outdir / "monthly_pnl_top.csv", index=True)

    # Summary cards
    def _best_label(df: pd.DataFrame, col: str, asc: bool = False) -> str:
        if col not in df.columns:
            return "n/a"
        x = df.dropna(subset=[col])
        if x.empty:
            return "n/a"
        s = x.sort_values(col, ascending=asc).iloc[0]
        return f"{s['setting']} ({_fmt(s[col], 3)})"

    cards = {
        "n_variants": str(len(m_ok)),
        "best_sharpe": _best_label(m_ok, "sharpe_daily", asc=False),
        "best_calmar": _best_label(m_ok, "calmar", asc=False),
        "best_total_pnl": _best_label(m_ok, "total_pnl", asc=False),
        "worst_dd": _best_label(m_ok, "max_dd_pct", asc=True),
    }

    out_html = outdir / "report.html"
    _render_html(
        out_html=out_html,
        run_id=str(a.run_id),
        run_root=run_root,
        metrics_df=m_ok,
        top_df=top,
        cards=cards,
        assets_rel="assets",
        monthly_top_tbl=monthly_top,
    )

    print(str(out_html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
