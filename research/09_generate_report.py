# 09_generate_report.py
# Generates results/Strategy_Performance_Report.html from results/trades.csv
# Robust version: Converts all data to standard Python lists to prevent Plotly index-plotting artifacts.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "../results"
TRADES_PATH = RESULTS_DIR / "trades.csv"
OUTPUT_FILE = RESULTS_DIR / "Strategy_Performance_Report.html"

INITIAL_CAPITAL = 1000.0
ANNUALIZATION = 365.0  # crypto-style

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------
# Helpers
# ----------------------------
def _placeholder_figure(title: str, note: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=420,
        margin=dict(l=50, r=30, t=60, b=40),
        annotations=[dict(
            text=note,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="#9aa4b2")
        )]
    )
    return fig


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    mapping = {
        True: True, False: False,
        1: True, 0: False,
        "1": True, "0": False,
        "true": True, "false": False,
        "t": True, "f": False,
        "yes": True, "no": False
    }
    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            return mapping.get(v.strip().lower(), np.nan)
        return mapping.get(v, np.nan)
    return s.map(conv)


def _fmt_money(x: float, decimals: int = 0) -> str:
    if not np.isfinite(x):
        return "—"
    return f"${x:,.{decimals}f}"


def _fmt_pf(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "∞" if np.isinf(x) else "—"
    return f"{x:.2f}"


# ----------------------------
# Load + clean
# ----------------------------
def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        return df

    required = {"entry_ts", "exit_ts", "pnl"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"trades.csv missing required columns: {missing}")

    # Robust UTC parse
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")

    # Drop unusable rows
    df = df.dropna(subset=["exit_ts"]).copy()

    # Filter out bad years (e.g. Year 1)
    df = df[df["exit_ts"].dt.year > 2000].copy()

    # Numeric PnL
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    if "fees" in df.columns:
        df["fees"] = pd.to_numeric(df["fees"], errors="coerce")
    if "pnl_R" in df.columns:
        df["pnl_R"] = pd.to_numeric(df["pnl_R"], errors="coerce")

    if "regime_up" in df.columns:
        df["regime_up"] = _coerce_bool_series(df["regime_up"])

    df = df.sort_values("exit_ts").reset_index(drop=True)
    return df


# ----------------------------
# Equity reconstruction (DAILY)
# ----------------------------
def reconstruct_daily_equity(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    s = trades.set_index("exit_ts")["pnl"].sort_index()
    daily_pnl = s.resample("D").sum()

    if not isinstance(daily_pnl.index, pd.DatetimeIndex):
        daily_pnl.index = pd.to_datetime(daily_pnl.index, utc=True)
    if daily_pnl.index.tz is None:
        daily_pnl.index = daily_pnl.index.tz_localize("UTC")
    else:
        daily_pnl.index = daily_pnl.index.tz_convert("UTC")

    if daily_pnl.empty:
        return pd.DataFrame()
        
    start = daily_pnl.index.min()
    end = daily_pnl.index.max()
    full_idx = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    
    daily_pnl = daily_pnl.reindex(full_idx).fillna(0.0)

    df = pd.DataFrame({"timestamp": daily_pnl.index, "daily_pnl": daily_pnl.values})
    df["cum_pnl"] = df["daily_pnl"].cumsum()
    df["equity"] = initial_capital + df["cum_pnl"]

    peak = df["equity"].cummax()
    denom = peak.replace(0.0, np.nan)
    df["drawdown_pct"] = (df["equity"] / denom - 1.0) * 100.0
    df["drawdown_pct"] = df["drawdown_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df.loc[df["drawdown_pct"] > 0, "drawdown_pct"] = 0.0

    return df


# ----------------------------
# KPIs
# ----------------------------
def calculate_kpis(trades: pd.DataFrame, eq: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty or eq.empty:
        return {}

    start_equity = float(initial_capital)
    end_equity = float(eq["equity"].iloc[-1])
    net_profit = end_equity - start_equity
    total_return_pct = (net_profit / start_equity) * 100.0 if start_equity else 0.0
    max_dd_pct = float(abs(eq["drawdown_pct"].min())) if len(eq) else 0.0

    rets = eq["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = float((rets.mean() / std) * np.sqrt(ANNUALIZATION)) if std > 0 else 0.0

    n = int(len(trades))
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]

    win_rate = (len(wins) / n) * 100.0 if n else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    avg_pnl = float(trades["pnl"].mean()) if n else 0.0

    return {
        "start_date": pd.to_datetime(eq["timestamp"].iloc[0]).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(eq["timestamp"].iloc[-1]).strftime("%Y-%m-%d"),
        "total_trades": n,
        "total_return_pct": float(total_return_pct),
        "net_profit": float(net_profit),
        "max_drawdown_pct": float(max_dd_pct),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.inf,
        "avg_pnl_per_trade": float(avg_pnl),
    }


# ----------------------------
# Charts
# ----------------------------
def chart_equity_and_drawdown(eq: pd.DataFrame) -> go.Figure:
    if eq.empty:
        return _placeholder_figure("Equity Curve & Drawdown", "No equity data available.")

    # FORCE CONVERSION TO PYTHON LISTS
    # This ensures Plotly receives standard types, bypassing any numpy/pandas version issues.
    x_data = eq["timestamp"].dt.strftime('%Y-%m-%d').tolist()
    y_equity = eq["equity"].fillna(0.0).tolist()
    y_drawdown = eq["drawdown_pct"].fillna(0.0).tolist()

    print(f"DEBUG: Charting Equity. Points: {len(y_equity)}. First 5: {y_equity[:5]}")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.70, 0.30],
    )

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_equity,
            mode="lines",
            name="Equity",
            line=dict(width=2, color="#636efa"),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_drawdown,
            mode="lines",
            name="Drawdown (%)",
            fill="tozeroy",
            line=dict(width=1, color="#ef553b"),
        ),
        row=2, col=1
    )

    dd_min = min(y_drawdown) if y_drawdown else 0.0
    y_min = dd_min * 1.1 if dd_min < 0 else -1.0
    fig.update_yaxes(range=[y_min, 0.5], row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title="Equity Curve & Drawdown",
        height=650,
        margin=dict(l=55, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def chart_monthly_pnl_heatmap(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return _placeholder_figure("Monthly PnL ($)", "No trades available.")

    t = trades.copy()
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, errors="coerce")
    t = t.dropna(subset=["exit_ts"])
    if t.empty:
        return _placeholder_figure("Monthly PnL ($)", "All exit_ts failed to parse.")

    t["year"] = t["exit_ts"].dt.year.astype(int)
    t["month"] = t["exit_ts"].dt.month.astype(int)

    monthly = t.groupby(["year", "month"], as_index=False)["pnl"].sum()
    if monthly.empty:
        return _placeholder_figure("Monthly PnL ($)", "No monthly data.")

    pivot = monthly.pivot(index="year", columns="month", values="pnl")
    pivot = pivot.reindex(columns=list(range(1, 13))).fillna(0.0)
    
    y_min, y_max = int(pivot.index.min()), int(pivot.index.max())
    pivot = pivot.reindex(index=list(range(y_min, y_max + 1))).fillna(0.0)
    pivot = pivot.sort_index(ascending=False)

    # FORCE CONVERSION TO PYTHON LISTS
    z_data = pivot.values.tolist()
    y_data = pivot.index.astype(int).tolist()
    x_data = MONTH_NAMES

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_data,
            y=y_data,
            zmid=0,
            colorscale="RdBu",
            hovertemplate="Year=%{y}<br>Month=%{x}<br>PnL=$%{z:,.2f}<extra></extra>",
            colorbar=dict(title="PnL ($)"),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Monthly PnL ($)",
        height=420,
        margin=dict(l=55, r=30, t=60, b=40),
        yaxis=dict(title="Year", dtick=1, type='category'),
        xaxis=dict(title="Month"),
    )
    return fig


def chart_cum_pnl_by_regime(trades: pd.DataFrame, eq: pd.DataFrame) -> go.Figure:
    if trades.empty or eq.empty:
        return _placeholder_figure("Cumulative PnL by Market Regime", "No data available.")

    if "regime_up" not in trades.columns:
        return _placeholder_figure("Cumulative PnL by Market Regime", "Column regime_up not found.")

    t = trades.copy()
    t["regime_up"] = _coerce_bool_series(t["regime_up"])
    t = t.dropna(subset=["regime_up"])
    if t.empty:
        return _placeholder_figure("Cumulative PnL by Market Regime", "No valid regime_up values.")

    t["date"] = pd.to_datetime(t["exit_ts"], utc=True, errors="coerce").dt.floor("D")
    t = t.dropna(subset=["date"])
    
    daily = t.groupby(["date", "regime_up"])["pnl"].sum().unstack(fill_value=0.0)

    idx = pd.DatetimeIndex(pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")).dropna()
    if len(idx) == 0:
        return _placeholder_figure("Cumulative PnL by Market Regime", "Equity timestamps invalid.")
        
    full = pd.date_range(start=idx.min(), end=idx.max(), freq="D", tz="UTC")
    daily = daily.reindex(full, fill_value=0.0)

    cum = daily.cumsum()
    
    # FORCE CONVERSION TO PYTHON LISTS
    x_data = cum.index.strftime('%Y-%m-%d').tolist()

    fig = go.Figure()
    if False in cum.columns:
        y_false = cum[False].fillna(0.0).tolist()
        fig.add_trace(go.Scatter(x=x_data, y=y_false, mode="lines", name="Regime = False (Bear/Chop)"))
    if True in cum.columns:
        y_true = cum[True].fillna(0.0).tolist()
        fig.add_trace(go.Scatter(x=x_data, y=y_true, mode="lines", name="Regime = True (Bull)"))

    if len(fig.data) == 0:
        return _placeholder_figure("Cumulative PnL by Market Regime", "No regime series to plot.")

    fig.update_layout(
        template="plotly_dark",
        title="Cumulative PnL by Market Regime",
        height=420,
        margin=dict(l=55, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Cumulative PnL ($)")
    return fig


# ----------------------------
# HTML Report
# ----------------------------
def build_html(kpis: dict, fig1: go.Figure, fig2: go.Figure, fig3: go.Figure, out_path: Path) -> None:
    css = """
    <style>
        body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
               background: #0e1117; color: #e6e6e6; margin: 0; padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { margin: 0 0 8px 0; font-size: 28px; text-align: center; }
        .meta { text-align: center; color: #9aa4b2; margin: 0 0 18px 0; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                    gap: 14px; margin-bottom: 18px; }
        .kpi-card { background: #161b22; border: 1px solid #2a2f3a; border-radius: 10px; padding: 14px; }
        .kpi-label { font-size: 12px; color: #9aa4b2; letter-spacing: .08em; text-transform: uppercase; }
        .kpi-value { font-size: 22px; margin-top: 6px; font-weight: 700; }
        .pos { color: #2dd4bf; }
        .neg { color: #fb7185; }
        .neutral { color: #e6e6e6; }
        .card { background: #161b22; border: 1px solid #2a2f3a; border-radius: 10px; padding: 10px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        @media (max-width: 980px) { .grid-2 { grid-template-columns: 1fr; } }
    </style>
    """

    total_return = kpis.get("total_return_pct", 0.0)
    net_profit = kpis.get("net_profit", 0.0)
    avg_trade = kpis.get("avg_pnl_per_trade", 0.0)

    total_return_class = "pos" if total_return >= 0 else "neg"
    net_profit_class = "pos" if net_profit >= 0 else "neg"
    avg_trade_class = "pos" if avg_trade >= 0 else "neg"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Strategy Performance Report</title>
  {css}
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <div class="container">
    <h1>Strategy Performance Report</h1>
    <p class="meta">
      {kpis.get("start_date","")} to {kpis.get("end_date","")} &nbsp;|&nbsp;
      Start Capital: {_fmt_money(INITIAL_CAPITAL, 0)} &nbsp;|&nbsp;
      Trades: {kpis.get("total_trades", 0)}
    </p>

    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Total Return</div><div class="kpi-value {total_return_class}">{total_return:.2f}%</div></div>
      <div class="kpi-card"><div class="kpi-label">Net Profit</div><div class="kpi-value {net_profit_class}">{_fmt_money(net_profit, 0)}</div></div>
      <div class="kpi-card"><div class="kpi-label">Max Drawdown</div><div class="kpi-value neg">{kpis.get("max_drawdown_pct",0.0):.2f}%</div></div>
      <div class="kpi-card"><div class="kpi-label">Sharpe (ann.)</div><div class="kpi-value neutral">{kpis.get("sharpe",0.0):.2f}</div></div>
      <div class="kpi-card"><div class="kpi-label">Win Rate</div><div class="kpi-value neutral">{kpis.get("win_rate",0.0):.1f}%</div></div>
      <div class="kpi-card"><div class="kpi-label">Profit Factor</div><div class="kpi-value neutral">{_fmt_pf(kpis.get("profit_factor",0.0))}</div></div>
      <div class="kpi-card"><div class="kpi-label">Avg PnL / Trade</div><div class="kpi-value {avg_trade_class}">{_fmt_money(avg_trade, 2)}</div></div>
    </div>

    <div class="card">
      {fig1.to_html(full_html=False, include_plotlyjs=False)}
    </div>

    <div style="height: 14px"></div>

    <div class="grid-2">
      <div class="card">{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>
      <div class="card">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>

  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading trades from {TRADES_PATH}...")
    trades = load_trades(TRADES_PATH)
    if trades.empty:
        fig = _placeholder_figure("Report", "trades.csv is empty.")
        build_html({}, fig, fig, fig, OUTPUT_FILE)
        print(f"Report generated (empty): {OUTPUT_FILE}")
        return 0

    print(f"Reconstructing equity (Initial Capital: ${INITIAL_CAPITAL})...")
    eq = reconstruct_daily_equity(trades, INITIAL_CAPITAL)
    if eq.empty:
        fig = _placeholder_figure("Report", "Failed to reconstruct equity from trades.")
        build_html({}, fig, fig, fig, OUTPUT_FILE)
        print(f"Report generated (no equity): {OUTPUT_FILE}")
        return 0

    # Debug output
    print("\n--- Equity Head ---")
    print(eq.head())
    print("\n--- Equity Tail ---")
    print(eq.tail())
    print("-------------------\n")

    kpis = calculate_kpis(trades, eq, INITIAL_CAPITAL)

    fig_eq = chart_equity_and_drawdown(eq)
    fig_hm = chart_monthly_pnl_heatmap(trades)
    fig_reg = chart_cum_pnl_by_regime(trades, eq)

    build_html(kpis, fig_eq, fig_hm, fig_reg, OUTPUT_FILE)

    print(f"Trades: {len(trades)}")
    print(f"Exit_ts range: {trades['exit_ts'].min()} -> {trades['exit_ts'].max()}")
    print(f"Sum pnl: {trades['pnl'].sum():.2f}")
    print(f"End equity: {eq['equity'].iloc[-1]:.2f}")
    print(f"Report generated: {OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())