#!/usr/bin/env python3
"""
tools/sweep_policy_settings.py

Policy sweep harness for the existing repo backtester.

Fix included:
- Filters partitioned signals universe to symbols that have OHLCV parquet present in cfg.PARQUET_DIR.
  Prevents backtester crashes on missing parquet (e.g., FORTHUSDT).

Usage:
  python tools/sweep_policy_settings.py --start 2023-01-01 --end latest --jobs 4

Outputs:
  results/policy_sweeps/<run_id>/<setting_name>/
    - run_config.json
    - metrics.json
    - equity_curve.csv   (canonical: timestamp,equity,drawdown)
    - trades.csv         (produced by backtester)
    - logs.txt
    - _DONE
  results/policy_sweeps/<run_id>/summary.csv
  results/policy_sweeps/<run_id>/report.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_ROOT = REPO_ROOT / "results" / "policy_sweeps"


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Policy sweep harness for Donch backtester (repo-native).")
    p.add_argument("--start", type=str, default="2023-01-01", help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default="latest", help="YYYY-MM-DD or 'latest'")
    p.add_argument("--jobs", type=int, default=1, help="Parallel jobs (subprocesses).")
    p.add_argument("--lambda", dest="lam", type=float, default=2.5,
                   help="Penalty weight for negative risk-off PnL in utility.")
    p.add_argument("--mu", dest="mu", type=float, default=1.5,
                   help="Penalty weight for max drawdown (cash) in utility.")
    p.add_argument("--run-id", type=str, default="",
                   help="Optional run id; default is UTC timestamp.")
    p.add_argument("--pstar", type=float, default=0.64,
                   help="Meta gate threshold p* (used for gating modes).")
    p.add_argument("--max-runs", type=int, default=0,
                   help="If >0, cap number of runs (useful for quick tests).")
    p.add_argument("--no-filter-missing-parquet", action="store_true",
                   help="Disable filtering of signals symbols missing parquet OHLCV. (Not recommended.)")
    return p.parse_args(argv)


def _parse_yyyy_mm_dd(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


# -----------------------------
# Latest date resolver (parquet metadata)
# -----------------------------

def _safe_import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq  # type: ignore
        return pq
    except Exception:
        return None


def _latest_ts_in_parquet(parquet_path: Path) -> Optional[pd.Timestamp]:
    pq = _safe_import_pyarrow_parquet()
    if pq is None or not parquet_path.exists():
        return None

    pf = pq.ParquetFile(str(parquet_path))
    if pf.num_row_groups <= 0:
        return None

    schema_names = set(pf.schema.names)
    candidates = ["timestamp", "ts", "__index_level_0__", "index"]
    ts_col = next((c for c in candidates if c in schema_names), None)
    if ts_col is None:
        return None

    rg = pf.num_row_groups - 1
    tbl = pf.read_row_group(rg, columns=[ts_col])
    arr = tbl.column(0)
    try:
        v = pd.to_datetime(arr.to_numpy(zero_copy_only=False)).max()
        if pd.isna(v):
            return None
        return pd.to_datetime(v, utc=True, errors="coerce")
    except Exception:
        try:
            vals = pd.to_datetime(arr.to_pylist(), utc=True, errors="coerce")
            v = pd.to_datetime(vals).max()
            if pd.isna(v):
                return None
            return pd.to_datetime(v, utc=True, errors="coerce")
        except Exception:
            return None


def resolve_end_date(end_arg: str, cfg: Any) -> str:
    end_arg = (end_arg or "").strip()
    if end_arg.lower() != "latest":
        _ = _parse_yyyy_mm_dd(end_arg)
        return end_arg

    parquet_dir = Path(getattr(cfg, "PARQUET_DIR", REPO_ROOT / "parquet")).resolve()
    eth = str(getattr(cfg, "REGIME_ASSET", "ETHUSDT"))
    btc = "BTCUSDT"

    latest_list: List[pd.Timestamp] = []
    for sym in [eth, btc]:
        p = parquet_dir / f"{sym}.parquet"
        ts = _latest_ts_in_parquet(p)
        if ts is not None:
            latest_list.append(ts)

    if not latest_list:
        fallback = str(getattr(cfg, "END_DATE", "") or "")
        if fallback:
            _ = _parse_yyyy_mm_dd(fallback)
            return fallback
        raise RuntimeError("Cannot resolve end=latest: no parquet timestamps found and cfg.END_DATE missing.")

    tmin = min(latest_list)
    return pd.to_datetime(tmin, utc=True).date().isoformat()


# -----------------------------
# Signals availability + filtering
# -----------------------------

def detect_signals_source(cfg: Any) -> Tuple[Optional[Path], str]:
    """
    Returns (signals_path, kind):
      - (cfg.SIGNALS_DIR, "partitioned") if symbol=* exists
      - (cfg.SIGNALS_DIR/"signals.parquet", "file") if file exists
      - (None, "missing") otherwise
    """
    sdir = Path(getattr(cfg, "SIGNALS_DIR", REPO_ROOT / "signals")).resolve()
    part_dirs = list(sdir.glob("symbol=*"))
    if part_dirs:
        return sdir, "partitioned"
    f = sdir / "signals.parquet"
    if f.exists():
        return f, "file"
    return None, "missing"


def _symbol_from_partition_dir(d: Path) -> Optional[str]:
    # expects "symbol=BTCUSDT"
    name = d.name
    if "=" not in name:
        return None
    k, v = name.split("=", 1)
    if k.strip().lower() != "symbol":
        return None
    return v.strip().upper() or None


def build_filtered_signals_partitioned(
    *,
    signals_dir: Path,
    parquet_dir: Path,
    out_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Create out_dir containing symlinks to symbol partitions that have OHLCV parquet available.
    Returns (filtered_dir, manifest).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            # If manifest exists, assume dir already built
            return out_dir, m
        except Exception:
            pass

    part_dirs = sorted([p for p in signals_dir.glob("symbol=*") if p.is_dir()])
    have = {p.stem.upper() for p in parquet_dir.glob("*.parquet")}
    # p.stem is "BTCUSDT" for "BTCUSDT.parquet"

    kept: List[str] = []
    skipped: List[str] = []
    for d in part_dirs:
        sym = _symbol_from_partition_dir(d)
        if sym is None:
            continue
        if sym not in have:
            skipped.append(sym)
            continue
        dst = out_dir / d.name
        if dst.exists():
            kept.append(sym)
            continue
        try:
            os.symlink(str(d), str(dst), target_is_directory=True)
            kept.append(sym)
        except Exception:
            # fallback: copytree (potentially heavy but guarantees correctness)
            shutil.copytree(d, dst)
            kept.append(sym)

    m = {
        "signals_dir": str(signals_dir),
        "parquet_dir": str(parquet_dir),
        "kept_symbols": sorted(set(kept)),
        "skipped_symbols_missing_parquet": sorted(set(skipped)),
        "kept_count": int(len(set(kept))),
        "skipped_count": int(len(set(skipped))),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir, m


# -----------------------------
# Grid definition
# -----------------------------

@dataclass(frozen=True)
class GridPoint:
    meta_mode: str
    regime_block_when_down: bool
    risk_off_probe_mult: float
    slope_filter_enabled: bool

    def name(self) -> str:
        b = "blockDown1" if self.regime_block_when_down else "blockDown0"
        r = f"probe{self.risk_off_probe_mult:.2f}".replace(".", "p")
        s = "slope1" if self.slope_filter_enabled else "slope0"
        return f"{self.meta_mode}__{b}__{r}__{s}"

    def overrides(self, pstar: float) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "REGIME_FILTER_ENABLED": True,
            "REGIME_BLOCK_WHEN_DOWN": bool(self.regime_block_when_down),
            "REGIME_SLOPE_FILTER_ENABLED": bool(self.slope_filter_enabled),
            "RISK_OFF_PROBE_MULT": float(self.risk_off_probe_mult),
        }

        if self.meta_mode == "no_meta_gate":
            d["META_PROB_THRESHOLD"] = None
        elif self.meta_mode == "gate_with_pstar_all":
            d["META_PROB_THRESHOLD"] = float(pstar)
        elif self.meta_mode == "gate_with_pstar_scope":
            # Fail-closed outside risk_on scope via size cap to zero in risk_off.
            d["META_PROB_THRESHOLD"] = float(pstar)
            d["RISK_OFF_PROBE_MULT"] = 0.0
        else:
            raise ValueError(f"Unknown meta_mode: {self.meta_mode}")
        return d


def build_grid() -> List[GridPoint]:
    meta_modes = ["no_meta_gate", "gate_with_pstar_all", "gate_with_pstar_scope"]
    blocks = [False, True]
    probes = [0.0, 0.01, 0.05]
    slopes = [False, True]

    out: List[GridPoint] = []
    for mm in meta_modes:
        for b in blocks:
            for s in slopes:
                if mm == "gate_with_pstar_scope":
                    out.append(GridPoint(mm, b, 0.0, s))
                else:
                    for r in probes:
                        out.append(GridPoint(mm, b, r, s))
    return out


# -----------------------------
# Subprocess runner (repo-native backtester)
# -----------------------------

SUBPROC_CODE = r"""
import json, os, sys
from pathlib import Path

repo_root = Path(os.environ["SWEEP_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg

# Capture original artifact paths BEFORE redirecting RESULTS_DIR
orig_results = Path(getattr(cfg, "RESULTS_DIR", repo_root / "results")).resolve()
orig_meta_pred = Path(getattr(cfg, "META_PRED_PATH", orig_results / "meta_export" / "oos_predictions_calibrated.parquet")).resolve()
orig_meta_model = Path(getattr(cfg, "META_MODEL_DIR", orig_results / "meta_export")).resolve()

# Apply overrides
overrides = json.loads(os.environ["SWEEP_OVERRIDES"])
for k, v in overrides.items():
    setattr(cfg, k, v)

# Dates
cfg.START_DATE = os.environ["SWEEP_START"]
cfg.END_DATE   = os.environ["SWEEP_END"]

# Redirect outputs ONLY (keep meta artifacts pinned to original locations)
out_dir = Path(os.environ["SWEEP_OUTDIR"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
cfg.RESULTS_DIR = out_dir
cfg.META_PRED_PATH = orig_meta_pred
cfg.META_MODEL_DIR = orig_meta_model

# Signals path (dir or file), passed through to backtester as-is
sig_path = os.environ.get("SWEEP_SIGNALSPATH", "").strip()
signals_path = Path(sig_path).resolve() if sig_path else None

import backtester
backtester.run_backtest(signals_path=signals_path)
"""


def run_one_backtest_subprocess(
    *,
    gp: GridPoint,
    run_dir: Path,
    start: str,
    end: str,
    pstar: float,
    signals_path: Optional[Path],
    log_path: Path,
) -> int:
    env = os.environ.copy()
    env["SWEEP_REPO_ROOT"] = str(REPO_ROOT)
    env["SWEEP_START"] = start
    env["SWEEP_END"] = end
    env["SWEEP_OUTDIR"] = str(run_dir.resolve())
    env["SWEEP_SIGNALSPATH"] = str(signals_path.resolve()) if signals_path else ""
    env["SWEEP_OVERRIDES"] = json.dumps(gp.overrides(pstar))

    run_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [sys.executable, "-c", SUBPROC_CODE],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


# -----------------------------
# Metrics
# -----------------------------

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _max_dd_duration_days(ts: pd.Series, equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    equity = pd.to_numeric(equity, errors="coerce")
    ok = ts.notna() & equity.notna()
    ts = ts[ok].reset_index(drop=True)
    equity = equity[ok].reset_index(drop=True)
    if len(equity) < 2:
        return 0.0

    peak = equity.cummax()
    max_dur = pd.Timedelta(0)
    i = 0
    n = len(equity)
    while i < n:
        peak_level = peak.iloc[i]
        peak_time = ts.iloc[i]
        j = i + 1
        if j < n and equity.iloc[j] < peak_level - 1e-12:
            while j < n and equity.iloc[j] < peak_level - 1e-12:
                j += 1
            end_time = ts.iloc[j] if j < n else ts.iloc[n - 1]
            dur = end_time - peak_time
            if dur > max_dur:
                max_dur = dur
            i = j
        else:
            i += 1

    return float(max_dur / pd.Timedelta(days=1))


def _interval_union_fraction(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]], total_start: pd.Timestamp, total_end: pd.Timestamp) -> float:
    if total_end <= total_start or not intervals:
        return 0.0

    clipped: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for a, b in intervals:
        if pd.isna(a) or pd.isna(b):
            continue
        a = max(a, total_start)
        b = min(b, total_end)
        if b > a:
            clipped.append((a, b))
    if not clipped:
        return 0.0

    clipped.sort(key=lambda x: x[0])
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_a, cur_b = clipped[0]
    for a, b in clipped[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))

    covered = sum((b - a for a, b in merged), pd.Timedelta(0))
    return float(covered / (total_end - total_start))


def compute_metrics_for_run(run_dir: Path, cfg: Any, lam: float, mu: float) -> Dict[str, Any]:
    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.csv"

    if not trades_path.exists():
        return {"status": "no_trades_file", "utility": np.nan}

    trades = pd.read_csv(trades_path, low_memory=False)
    if trades.empty:
        return {"status": "no_trades", "utility": np.nan}

    entry_col = _pick_col(trades, ["entry_ts", "entry_time", "entry_timestamp"])
    exit_col = _pick_col(trades, ["exit_ts", "exit_time", "exit_timestamp"])
    pnl_col = _pick_col(trades, ["pnl_cash", "pnl"])

    if entry_col is None or exit_col is None or pnl_col is None:
        raise RuntimeError(f"Trades missing required columns. First cols: {list(trades.columns)[:80]}")

    trades["_entry_ts"] = _coerce_ts(trades[entry_col])
    trades["_exit_ts"] = _coerce_ts(trades[exit_col])
    trades = trades[trades["_entry_ts"].notna() & trades["_exit_ts"].notna()].copy()
    trades.sort_values("_exit_ts", inplace=True)

    pnl_cash = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)

    init_cap = float(getattr(cfg, "INITIAL_CAPITAL", 1000.0))

    # Equity
    if equity_path.exists():
        eq = pd.read_csv(equity_path)
        tcol = _pick_col(eq, ["timestamp", "ts", "time"])
        ecol = _pick_col(eq, ["equity", "balance", "nav"])
        if tcol is not None and ecol is not None:
            eq = eq[[tcol, ecol]].copy()
            eq["timestamp"] = _coerce_ts(eq[tcol])
            eq["equity"] = pd.to_numeric(eq[ecol], errors="coerce")
            eq = eq[eq["timestamp"].notna() & eq["equity"].notna()].sort_values("timestamp")
        else:
            eq = pd.DataFrame({
                "timestamp": trades["_exit_ts"].values,
                "equity": init_cap + pnl_cash.cumsum().values
            })
    else:
        eq = pd.DataFrame({
            "timestamp": trades["_exit_ts"].values,
            "equity": init_cap + pnl_cash.cumsum().values
        })

    eq["peak"] = eq["equity"].cummax()
    eq["drawdown"] = eq["equity"] / eq["peak"] - 1.0

    total_return = float(eq["equity"].iloc[-1] / init_cap - 1.0) if len(eq) else 0.0
    max_dd_frac = float(eq["drawdown"].min()) if len(eq) else 0.0
    max_dd_cash = float(abs(max_dd_frac) * init_cap)
    dd_dur_days = _max_dd_duration_days(eq["timestamp"], eq["equity"])

    # CAGR
    if len(eq) >= 2:
        days = max((eq["timestamp"].iloc[-1] - eq["timestamp"].iloc[0]) / pd.Timedelta(days=1), 1e-9)
        cagr = float((eq["equity"].iloc[-1] / init_cap) ** (365.0 / days) - 1.0)
    else:
        cagr = 0.0

    n_trades = int(len(trades))
    win_rate = float((pnl_cash > 0).mean()) if n_trades else 0.0

    # Exposure
    start_ts = pd.to_datetime(getattr(cfg, "START_DATE", None) or trades["_entry_ts"].min(), utc=True)
    end_ts = pd.to_datetime(getattr(cfg, "END_DATE", None) or trades["_exit_ts"].max(), utc=True) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    intervals_all = list(zip(trades["_entry_ts"], trades["_exit_ts"]))
    exposure = _interval_union_fraction(intervals_all, start_ts, end_ts)

    # Risk split:
    # Prefer backtester-provided tag if present; else leave risk split as NaN (do not guess).
    risk_col = _pick_col(trades, ["risk_on_1", "risk_on"])
    if risk_col is not None:
        risk_on = pd.to_numeric(trades[risk_col], errors="coerce").fillna(0).astype(int)
        pnl_on = float(pnl_cash[risk_on == 1].sum())
        pnl_off = float(pnl_cash[risk_on == 0].sum())

        def _subset_max_dd_cash(mask: pd.Series) -> float:
            sub = pnl_cash[mask].fillna(0.0)
            if sub.empty:
                return 0.0
            cum = sub.cumsum()
            peak = cum.cummax()
            dd = cum - peak
            return float(abs(dd.min()))

        max_dd_on_cash = _subset_max_dd_cash(risk_on == 1)
        max_dd_off_cash = _subset_max_dd_cash(risk_on == 0)

        intervals_on = [(a, b) for a, b, rf in zip(trades["_entry_ts"], trades["_exit_ts"], risk_on) if rf == 1]
        intervals_off = [(a, b) for a, b, rf in zip(trades["_entry_ts"], trades["_exit_ts"], risk_on) if rf == 0]
        exposure_on = _interval_union_fraction(intervals_on, start_ts, end_ts)
        exposure_off = _interval_union_fraction(intervals_off, start_ts, end_ts)

        worst_len = 0
        worst_sum = 0.0
        cur_len = 0
        cur_sum = 0.0
        for rf, p in zip(risk_on.to_list(), pnl_cash.to_list()):
            if rf == 0 and p < 0:
                cur_len += 1
                cur_sum += p
                if cur_len > worst_len:
                    worst_len = cur_len
                    worst_sum = cur_sum
            else:
                cur_len = 0
                cur_sum = 0.0

        bad_regime_damage = float((-pnl_cash[(risk_on == 0) & (pnl_cash < 0)].sum()))
        utility = float(pnl_on - lam * abs(min(pnl_off, 0.0)) - mu * max_dd_cash)

        risk_metrics = {
            "pnl_risk_on_cash": pnl_on,
            "pnl_risk_off_cash": pnl_off,
            "max_dd_risk_on_cash": max_dd_on_cash,
            "max_dd_risk_off_cash": max_dd_off_cash,
            "trade_count_risk_on": int((risk_on == 1).sum()),
            "trade_count_risk_off": int((risk_on == 0).sum()),
            "exposure_risk_on": exposure_on,
            "exposure_risk_off": exposure_off,
            "worst_loss_streak_len_risk_off": int(worst_len),
            "worst_loss_streak_pnl_risk_off": float(worst_sum),
            "bad_regime_damage_cash": bad_regime_damage,
            "utility": utility,
        }
    else:
        # No risk tags available in trades output; avoid fabricating split metrics.
        risk_metrics = {
            "pnl_risk_on_cash": np.nan,
            "pnl_risk_off_cash": np.nan,
            "max_dd_risk_on_cash": np.nan,
            "max_dd_risk_off_cash": np.nan,
            "trade_count_risk_on": np.nan,
            "trade_count_risk_off": np.nan,
            "exposure_risk_on": np.nan,
            "exposure_risk_off": np.nan,
            "worst_loss_streak_len_risk_off": np.nan,
            "worst_loss_streak_pnl_risk_off": np.nan,
            "bad_regime_damage_cash": np.nan,
            "utility": np.nan,
        }

    # Canonical equity_curve.csv
    eq_out = eq[["timestamp", "equity", "drawdown"]].copy()
    eq_out.to_csv(run_dir / "equity_curve.csv", index=False)

    return {
        "status": "ok",
        "total_return": total_return,
        "CAGR": cagr,
        "max_drawdown_frac": max_dd_frac,
        "max_drawdown_cash": max_dd_cash,
        "max_dd_duration_days": dd_dur_days,
        "number_of_trades": n_trades,
        "win_rate": win_rate,
        "exposure_time_in_market": exposure,
        **risk_metrics,
    }


# -----------------------------
# Report / summary
# -----------------------------

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_report_md(df: pd.DataFrame, run_id: str, lam: float, mu: float) -> str:
    cols = [
        "rank", "setting_name", "utility",
        "pnl_risk_on_cash", "pnl_risk_off_cash",
        "max_drawdown_cash", "max_dd_duration_days",
        "total_return", "CAGR",
        "number_of_trades", "win_rate",
        "bad_regime_damage_cash",
        "meta_mode", "REGIME_BLOCK_WHEN_DOWN", "RISK_OFF_PROBE_MULT", "REGIME_SLOPE_FILTER_ENABLED",
    ]
    keep = [c for c in cols if c in df.columns]

    top3 = df.head(3)
    lines: List[str] = []
    lines.append(f"# Policy sweep report: {run_id}")
    lines.append("")
    lines.append(f"Utility = PnL_risk_on  - {lam:g} * abs(min(PnL_risk_off,0))  - {mu:g} * max_drawdown_cash")
    lines.append("")
    lines.append("## Top 3 settings (by utility)")
    for _, r in top3.iterrows():
        lines.append(
            f"- {r.get('setting_name','')}: utility={r.get('utility',np.nan):.2f}, "
            f"PnL_on={r.get('pnl_risk_on_cash',np.nan):.2f}, "
            f"PnL_off={r.get('pnl_risk_off_cash',np.nan):.2f}, "
            f"maxDD={r.get('max_drawdown_cash',np.nan):.2f}"
        )
    lines.append("")
    lines.append("## Ranked table (top 15)")
    lines.append("")
    view = df[keep].head(15).copy()
    for c in view.columns:
        if c in ("utility", "pnl_risk_on_cash", "pnl_risk_off_cash", "max_drawdown_cash", "bad_regime_damage_cash"):
            view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
        if c in ("total_return", "CAGR", "win_rate", "exposure_time_in_market", "exposure_risk_on", "exposure_risk_off"):
            view[c] = pd.to_numeric(view[c], errors="coerce").round(4)
        if c in ("max_dd_duration_days",):
            view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
    lines.append(view.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT))
    import config as cfg  # type: ignore

    run_id = args.run_id.strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_root = DEFAULT_SWEEP_ROOT / run_id
    sweep_root.mkdir(parents=True, exist_ok=True)

    start = args.start
    _ = _parse_yyyy_mm_dd(start)
    end = resolve_end_date(args.end, cfg)
    _ = _parse_yyyy_mm_dd(end)

    sig_path, sig_kind = detect_signals_source(cfg)
    if sig_kind == "missing" or sig_path is None:
        raise RuntimeError(
            f"No signals found under {Path(getattr(cfg,'SIGNALS_DIR',REPO_ROOT/'signals')).resolve()} "
            "(expected signals/symbol=* or signals/signals.parquet)."
        )

    parquet_dir = Path(getattr(cfg, "PARQUET_DIR", REPO_ROOT / "parquet")).resolve()
    if not parquet_dir.exists():
        raise RuntimeError(f"cfg.PARQUET_DIR does not exist: {parquet_dir}")

    # Build filtered signals (partitioned only)
    signals_for_runs: Optional[Path] = sig_path
    filter_manifest: Dict[str, Any] = {}
    if (not args.no_filter_missing_parquet) and sig_kind == "partitioned":
        filtered_dir = sweep_root / "_signals_filtered"
        signals_for_runs, filter_manifest = build_filtered_signals_partitioned(
            signals_dir=sig_path,
            parquet_dir=parquet_dir,
            out_dir=filtered_dir,
        )
        write_json(sweep_root / "signals_filter_manifest.json", filter_manifest)

    grid = build_grid()
    if args.max_runs and args.max_runs > 0:
        grid = grid[: int(args.max_runs)]

    @dataclass
    class RunSpec:
        gp: GridPoint
        run_dir: Path
        setting_name: str

    specs: List[RunSpec] = []
    for gp in grid:
        setting_name = gp.name()
        run_dir = sweep_root / setting_name
        specs.append(RunSpec(gp=gp, run_dir=run_dir, setting_name=setting_name))

    def _run_one(spec: RunSpec) -> Dict[str, Any]:
        done = spec.run_dir / "_DONE"
        log_path = spec.run_dir / "logs.txt"
        cfg_path = spec.run_dir / "run_config.json"
        met_path = spec.run_dir / "metrics.json"

        if done.exists() and met_path.exists() and cfg_path.exists():
            return {"setting_name": spec.setting_name, "status": "skipped_existing"}

        spec.run_dir.mkdir(parents=True, exist_ok=True)

        run_cfg = {
            "start": start,
            "end": end,
            "setting_name": spec.setting_name,
            "grid_point": {
                "meta_mode": spec.gp.meta_mode,
                "REGIME_BLOCK_WHEN_DOWN": spec.gp.regime_block_when_down,
                "RISK_OFF_PROBE_MULT": spec.gp.risk_off_probe_mult,
                "REGIME_SLOPE_FILTER_ENABLED": spec.gp.slope_filter_enabled,
            },
            "overrides": spec.gp.overrides(args.pstar),
            "pstar": args.pstar,
            "signals_kind": sig_kind,
            "signals_path_used": str(signals_for_runs) if signals_for_runs else "",
            "parquet_dir": str(parquet_dir),
            "signals_filter": filter_manifest if filter_manifest else None,
        }
        write_json(cfg_path, run_cfg)

        rc = run_one_backtest_subprocess(
            gp=spec.gp,
            run_dir=spec.run_dir,
            start=start,
            end=end,
            pstar=args.pstar,
            signals_path=signals_for_runs,
            log_path=log_path,
        )
        if rc != 0:
            write_json(met_path, {"status": "error", "returncode": rc})
            return {"setting_name": spec.setting_name, "status": "error", "returncode": rc}

        try:
            metrics = compute_metrics_for_run(spec.run_dir, cfg, lam=float(args.lam), mu=float(args.mu))
            write_json(met_path, metrics)
            if metrics.get("status") == "ok":
                done.write_text(dt.datetime.now(dt.timezone.utc).isoformat(), encoding="utf-8")
            return {"setting_name": spec.setting_name, **metrics}
        except Exception as e:
            write_json(met_path, {"status": "error_metrics", "error": repr(e)})
            return {"setting_name": spec.setting_name, "status": "error_metrics", "error": repr(e)}

    results: List[Dict[str, Any]] = []
    jobs = max(int(args.jobs), 1)
    if jobs == 1:
        for spec in specs:
            results.append(_run_one(spec))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_run_one, spec): spec for spec in specs}
            for fut in as_completed(futs):
                results.append(fut.result())

    # Consolidate from disk
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        met_path = spec.run_dir / "metrics.json"
        cfg_path = spec.run_dir / "run_config.json"
        row: Dict[str, Any] = {"setting_name": spec.setting_name}
        if cfg_path.exists():
            rcfg = load_json(cfg_path)
            row["meta_mode"] = rcfg.get("grid_point", {}).get("meta_mode")
            row["REGIME_BLOCK_WHEN_DOWN"] = rcfg.get("grid_point", {}).get("REGIME_BLOCK_WHEN_DOWN")
            row["RISK_OFF_PROBE_MULT"] = rcfg.get("grid_point", {}).get("RISK_OFF_PROBE_MULT")
            row["REGIME_SLOPE_FILTER_ENABLED"] = rcfg.get("grid_point", {}).get("REGIME_SLOPE_FILTER_ENABLED")
        if met_path.exists():
            met = load_json(met_path)
            if isinstance(met, dict):
                row.update(met)
        rows.append(row)

    df = pd.DataFrame(rows)

    df["utility"] = pd.to_numeric(df.get("utility", np.nan), errors="coerce")
    ok_mask = (df.get("status", "") == "ok") & df["utility"].notna()

    ranked = df[ok_mask].sort_values("utility", ascending=False).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    out_summary = sweep_root / "summary.csv"
    df_out = pd.concat([ranked, df[~ok_mask]], ignore_index=True)
    df_out.to_csv(out_summary, index=False)

    out_report = sweep_root / "report.md"
    if len(ranked):
        report = make_report_md(ranked, run_id=run_id, lam=float(args.lam), mu=float(args.mu))
    else:
        report = f"# Policy sweep report: {run_id}\n\nNo successful runs to rank.\n"
    out_report.write_text(report, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
