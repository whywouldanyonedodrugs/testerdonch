#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sweep_policy_settings_v2 import (  # type: ignore
    find_trades_file,
    import_cfg,
    latest_from_signals,
    load_scoped_signals,
    preflight_symbols,
    resolve_paths,
    run_backtest_subprocess,
    write_signals_file,
)
from telegram_notify import TelegramNotifier


def _parse_date(s: str) -> str:
    if s.lower() == "latest":
        return "latest"
    datetime.strptime(s, "%Y-%m-%d")
    return s


def _utc_run_id(user_id: str) -> str:
    return user_id.strip() or datetime.now(timezone.utc).strftime("jt014_%Y%m%d_%H%M%S")


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("empty float list")
    return out


def _safe_float(x: object, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _safe_div(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def _dd_tail_p05_from_pnl_r(pnl_r: pd.Series) -> float:
    if pnl_r.empty:
        return float("nan")
    eq = pnl_r.fillna(0.0).cumsum()
    dd = eq - eq.cummax()
    if dd.empty:
        return float("nan")
    return float(np.nanquantile(dd.to_numpy(dtype=float), 0.05))


def _max_dd_r_from_pnl_r(pnl_r: pd.Series) -> float:
    if pnl_r.empty:
        return float("nan")
    eq = pnl_r.fillna(0.0).cumsum()
    dd = eq - eq.cummax()
    return float(dd.min()) if not dd.empty else float("nan")


def _downside_dev(pnl: pd.Series) -> float:
    arr = pd.to_numeric(pnl, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(np.minimum(arr, 0.0) ** 2)))


def _rolling_worst_mean(pnl_r: pd.Series, window: int) -> float:
    arr = pd.to_numeric(pnl_r, errors="coerce").dropna()
    if arr.empty:
        return float("nan")
    w = int(max(2, window))
    min_periods = int(min(len(arr), max(2, w // 2)))
    roll = arr.rolling(window=w, min_periods=min_periods).mean()
    return float(roll.min()) if roll.notna().any() else float("nan")


def _pick_ts_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("exit_ts", "closed_at", "timestamp", "ts"):
        if c in df.columns:
            return c
    return None


def _compute_window_rows(df: pd.DataFrame, setting: str, bucket: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "setting",
                "bucket",
                "month",
                "n_trades",
                "sum_pnl_cash",
                "mean_pnl_cash",
                "sum_pnl_R",
                "mean_pnl_R",
                "win_rate",
            ]
        )
    w = df.copy()
    w["month"] = w["exit_ts_use"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    g = w.groupby("month", sort=True)
    out = pd.DataFrame(
        {
            "n_trades": g.size(),
            "sum_pnl_cash": g["pnl_cash_use"].sum(),
            "mean_pnl_cash": g["pnl_cash_use"].mean(),
            "sum_pnl_R": g["pnl_R_use"].sum(),
            "mean_pnl_R": g["pnl_R_use"].mean(),
            "win_rate": g["pnl_cash_use"].apply(lambda s: float((s > 0).mean())),
        }
    ).reset_index()
    out.insert(0, "bucket", bucket)
    out.insert(0, "setting", setting)
    return out


def _compute_variant_metrics(
    run_dir: Path,
    *,
    rolling_trades_window: int,
    initial_capital: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    trades_path = find_trades_file(run_dir)
    df = pd.read_csv(trades_path, low_memory=False)

    ts_col = _pick_ts_col(df)
    if ts_col is None:
        raise RuntimeError(f"No exit timestamp column in {trades_path}")
    w = df.copy()
    w["exit_ts_use"] = pd.to_datetime(w[ts_col], utc=True, errors="coerce")

    pnl_cash_col = None
    for c in ("pnl", "pnl_cash", "pnl_usd", "pnl_net", "pnl_after_fees"):
        if c in w.columns:
            pnl_cash_col = c
            break
    if pnl_cash_col is None:
        raise RuntimeError(f"No pnl cash-like column in {trades_path}")

    pnl_r_col = None
    for c in ("pnl_R", "pnl_r"):
        if c in w.columns:
            pnl_r_col = c
            break
    if pnl_r_col is None:
        raise RuntimeError(f"No pnl_R-like column in {trades_path}")

    w["pnl_cash_use"] = pd.to_numeric(w[pnl_cash_col], errors="coerce")
    w["pnl_R_use"] = pd.to_numeric(w[pnl_r_col], errors="coerce")
    if "risk_on" in w.columns:
        w["risk_on_use"] = pd.to_numeric(w["risk_on"], errors="coerce")
    elif "risk_on_1" in w.columns:
        w["risk_on_use"] = pd.to_numeric(w["risk_on_1"], errors="coerce")
    else:
        w["risk_on_use"] = np.nan

    w = w.dropna(subset=["exit_ts_use", "pnl_cash_use", "pnl_R_use"]).sort_values("exit_ts_use", kind="mergesort")
    if w.empty:
        met = {
            "status": "ok",
            "trades_file": str(trades_path),
            "number_of_trades": 0,
            "total_pnl_cash": 0.0,
            "total_pnl_R": 0.0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "max_dd_pct": float("nan"),
            "sharpe_daily": float("nan"),
            "sortino_daily": float("nan"),
            "calmar": float("nan"),
            "overall_worst_window_mean_pnl_R": float("nan"),
            "overall_worst_month_mean_pnl_R": float("nan"),
            "overall_positive_month_ratio": float("nan"),
            "risk_off_n_trades": 0,
            "risk_off_total_pnl_cash": 0.0,
            "risk_off_total_pnl_R": 0.0,
            "risk_off_worst_window_mean_pnl_R": float("nan"),
            "risk_off_worst_month_mean_pnl_R": float("nan"),
            "risk_off_positive_month_ratio": float("nan"),
            "risk_off_downside_deviation_R": float("nan"),
            "risk_off_drawdown_tail_p05_R": float("nan"),
            "risk_off_max_dd_R": float("nan"),
            "rolling_trades_window": int(rolling_trades_window),
        }
        return met, pd.DataFrame()

    n = int(len(w))
    pnl_cash = w["pnl_cash_use"]
    pnl_r = w["pnl_R_use"]
    total_pnl_cash = float(pnl_cash.sum())
    total_pnl_r = float(pnl_r.sum())
    win_rate = float((pnl_cash > 0).mean())
    gross_pos = float(pnl_cash[pnl_cash > 0].sum())
    gross_neg = float(-pnl_cash[pnl_cash < 0].sum())
    pf = _safe_div(gross_pos, gross_neg) if gross_neg > 0 else float("nan")

    daily = (
        w.groupby(w["exit_ts_use"].dt.floor("D"), as_index=True)["pnl_cash_use"]
        .sum()
        .sort_index()
    )
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")
    if not daily.empty:
        idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
        daily = daily.reindex(idx, fill_value=0.0)
        equity = float(initial_capital) + daily.cumsum()
        max_dd_pct = _max_drawdown_pct(equity)
        rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) > 1:
            mu = float(rets.mean())
            sd = float(rets.std(ddof=1))
            neg = rets[rets < 0]
            sd_down = float(neg.std(ddof=1)) if len(neg) > 1 else float("nan")
            sharpe = _safe_div(mu, sd) * math.sqrt(252.0) if np.isfinite(sd) and sd > 0 else float("nan")
            sortino = _safe_div(mu, sd_down) * math.sqrt(252.0) if np.isfinite(sd_down) and sd_down > 0 else float("nan")
        else:
            sharpe = float("nan")
            sortino = float("nan")
        days = max((daily.index.max() - daily.index.min()).days, 1)
        years = days / 365.25
        e0 = float(equity.iloc[0])
        e1 = float(equity.iloc[-1])
        cagr = float((e1 / e0) ** (1.0 / years) - 1.0) if (e0 > 0 and e1 > 0 and years > 0) else float("nan")
        calmar = _safe_div(cagr, abs(max_dd_pct)) if np.isfinite(max_dd_pct) and max_dd_pct < 0 else float("nan")
    else:
        max_dd_pct = float("nan")
        sharpe = float("nan")
        sortino = float("nan")
        calmar = float("nan")

    m_all = (
        w.assign(month=w["exit_ts_use"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str))
        .groupby("month", sort=True)["pnl_R_use"]
        .mean()
    )
    overall_worst_month = float(m_all.min()) if len(m_all) else float("nan")
    overall_pos_month_ratio = float((m_all > 0).mean()) if len(m_all) else float("nan")
    overall_worst_window = _rolling_worst_mean(pnl_r, rolling_trades_window)

    risk_off = w[w["risk_on_use"] == 0].copy()
    if risk_off.empty:
        risk_off_n = 0
        risk_off_total_cash = 0.0
        risk_off_total_r = 0.0
        risk_off_worst_window = float("nan")
        risk_off_worst_month = float("nan")
        risk_off_pos_month_ratio = float("nan")
        risk_off_down_r = float("nan")
        risk_off_dd_tail = float("nan")
        risk_off_max_dd_r = float("nan")
    else:
        risk_off_n = int(len(risk_off))
        risk_off_total_cash = float(risk_off["pnl_cash_use"].sum())
        risk_off_total_r = float(risk_off["pnl_R_use"].sum())
        risk_off_worst_window = _rolling_worst_mean(risk_off["pnl_R_use"], rolling_trades_window)
        m_off = (
            risk_off.assign(month=risk_off["exit_ts_use"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str))
            .groupby("month", sort=True)["pnl_R_use"]
            .mean()
        )
        risk_off_worst_month = float(m_off.min()) if len(m_off) else float("nan")
        risk_off_pos_month_ratio = float((m_off > 0).mean()) if len(m_off) else float("nan")
        risk_off_down_r = _downside_dev(risk_off["pnl_R_use"])
        risk_off_dd_tail = _dd_tail_p05_from_pnl_r(risk_off["pnl_R_use"])
        risk_off_max_dd_r = _max_dd_r_from_pnl_r(risk_off["pnl_R_use"])

    window_parts = [
        _compute_window_rows(w, setting="", bucket="all"),
        _compute_window_rows(w[w["risk_on_use"] == 1], setting="", bucket="risk_on"),
        _compute_window_rows(w[w["risk_on_use"] == 0], setting="", bucket="risk_off"),
    ]
    window_parts = [p for p in window_parts if not p.empty]
    if window_parts:
        windows = pd.concat(window_parts, ignore_index=True)
    else:
        windows = pd.DataFrame(
            columns=[
                "setting",
                "bucket",
                "month",
                "n_trades",
                "sum_pnl_cash",
                "mean_pnl_cash",
                "sum_pnl_R",
                "mean_pnl_R",
                "win_rate",
            ]
        )

    met = {
        "status": "ok",
        "trades_file": str(trades_path),
        "number_of_trades": n,
        "total_pnl_cash": total_pnl_cash,
        "total_pnl_R": total_pnl_r,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_dd_pct": max_dd_pct,
        "sharpe_daily": sharpe,
        "sortino_daily": sortino,
        "calmar": calmar,
        "overall_worst_window_mean_pnl_R": overall_worst_window,
        "overall_worst_month_mean_pnl_R": overall_worst_month,
        "overall_positive_month_ratio": overall_pos_month_ratio,
        "risk_off_n_trades": risk_off_n,
        "risk_off_total_pnl_cash": risk_off_total_cash,
        "risk_off_total_pnl_R": risk_off_total_r,
        "risk_off_worst_window_mean_pnl_R": risk_off_worst_window,
        "risk_off_worst_month_mean_pnl_R": risk_off_worst_month,
        "risk_off_positive_month_ratio": risk_off_pos_month_ratio,
        "risk_off_downside_deviation_R": risk_off_down_r,
        "risk_off_drawdown_tail_p05_R": risk_off_dd_tail,
        "risk_off_max_dd_R": risk_off_max_dd_r,
        "rolling_trades_window": int(rolling_trades_window),
    }
    return met, windows


def _empty_metrics_payload(run_dir: Path, *, reason: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    met = {
        "status": "ok",
        "trades_file": str(run_dir / "trades.csv"),
        "number_of_trades": 0,
        "total_pnl_cash": 0.0,
        "total_pnl_R": 0.0,
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "max_dd_pct": float("nan"),
        "sharpe_daily": float("nan"),
        "sortino_daily": float("nan"),
        "calmar": float("nan"),
        "overall_worst_window_mean_pnl_R": float("nan"),
        "overall_worst_month_mean_pnl_R": float("nan"),
        "overall_positive_month_ratio": float("nan"),
        "risk_off_n_trades": 0,
        "risk_off_total_pnl_cash": 0.0,
        "risk_off_total_pnl_R": 0.0,
        "risk_off_worst_window_mean_pnl_R": float("nan"),
        "risk_off_worst_month_mean_pnl_R": float("nan"),
        "risk_off_positive_month_ratio": float("nan"),
        "risk_off_downside_deviation_R": float("nan"),
        "risk_off_drawdown_tail_p05_R": float("nan"),
        "risk_off_max_dd_R": float("nan"),
        "empty_reason": str(reason),
    }
    return met, pd.DataFrame()


@dataclass(frozen=True)
class ExitPreset:
    name: str
    enabled: bool
    risk_off_sl: float
    risk_off_tp: float
    risk_off_time: Optional[float]
    partial_tp_enabled: bool = False
    partial_tp_ratio: float = 0.5
    partial_tp1_atr_mult: Optional[float] = None
    move_sl_to_be_on_tp1: bool = False
    trail_after_tp1: bool = False
    trail_atr_mult: float = 1.0
    dyn_exits_enabled: bool = False
    dyn_macd_hist_thresh: float = 0.0
    dyn_tp_mult_pos: float = 1.15
    dyn_sl_mult_pos: float = 0.90
    dyn_tp_mult_neg: float = 0.85
    dyn_sl_mult_neg: float = 1.15
    exit_on_regime_flip: bool = False
    regime_flip_grace_bars: int = 1

    def overrides(
        self,
        *,
        base_sl: float,
        base_tp: float,
        base_time: Optional[float],
        block_when_down: bool,
        size_when_down: float,
        probe_mult: float,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "SL_ATR_MULT": float(base_sl),
            "TP_ATR_MULT": float(base_tp),
            "TIME_EXIT_HOURS": None if base_time is None else float(base_time),
            "REGIME_COND_EXITS_ENABLED": bool(self.enabled),
            "RISK_ON_SL_ATR_MULT": float(base_sl),
            "RISK_ON_TP_ATR_MULT": float(base_tp),
            "RISK_ON_TIME_EXIT_HOURS": None if base_time is None else float(base_time),
            "RISK_OFF_SL_ATR_MULT": float(self.risk_off_sl),
            "RISK_OFF_TP_ATR_MULT": float(self.risk_off_tp),
            "RISK_OFF_TIME_EXIT_HOURS": None if self.risk_off_time is None else float(self.risk_off_time),
            "REGIME_BLOCK_WHEN_DOWN": bool(block_when_down),
            "REGIME_SIZE_WHEN_DOWN": float(size_when_down),
            "RISK_OFF_PROBE_MULT": float(probe_mult),
            "PARTIAL_TP_ENABLED": bool(self.partial_tp_enabled),
            "PARTIAL_TP_RATIO": float(self.partial_tp_ratio),
            "MOVE_SL_TO_BE_ON_TP1": bool(self.move_sl_to_be_on_tp1),
            "TRAIL_AFTER_TP1": bool(self.trail_after_tp1),
            "TRAIL_ATR_MULT": float(self.trail_atr_mult),
            "DYN_EXITS_ENABLED": bool(self.dyn_exits_enabled),
            "DYN_MACD_HIST_THRESH": float(self.dyn_macd_hist_thresh),
            "DYN_TP_MULT_POS": float(self.dyn_tp_mult_pos),
            "DYN_SL_MULT_POS": float(self.dyn_sl_mult_pos),
            "DYN_TP_MULT_NEG": float(self.dyn_tp_mult_neg),
            "DYN_SL_MULT_NEG": float(self.dyn_sl_mult_neg),
            "EXIT_ON_REGIME_FLIP": bool(self.exit_on_regime_flip),
            "EXIT_ON_REGIME_FLIP_GRACE_BARS": int(self.regime_flip_grace_bars),
        }
        if self.partial_tp1_atr_mult is not None:
            out["PARTIAL_TP1_ATR_MULT"] = float(self.partial_tp1_atr_mult)
        return out


def _make_compact_presets(base_sl: float, base_tp: float, base_time: Optional[float]) -> List[ExitPreset]:
    te = float(base_time) if base_time is not None else 72.0
    return [
        ExitPreset(name="baseline_exits", enabled=False, risk_off_sl=float(base_sl), risk_off_tp=float(base_tp), risk_off_time=base_time),
        ExitPreset(name=f"ro_sl150_tp600_te{int(te)}", enabled=True, risk_off_sl=1.5, risk_off_tp=6.0, risk_off_time=te),
        ExitPreset(name=f"ro_sl150_tp400_te{int(te)}", enabled=True, risk_off_sl=1.5, risk_off_tp=4.0, risk_off_time=te),
        ExitPreset(name="ro_sl150_tp400_te48", enabled=True, risk_off_sl=1.5, risk_off_tp=4.0, risk_off_time=48.0),
        ExitPreset(name="ro_sl150_tp400_te24", enabled=True, risk_off_sl=1.5, risk_off_tp=4.0, risk_off_time=24.0),
        ExitPreset(name="ro_sl125_tp400_te24", enabled=True, risk_off_sl=1.25, risk_off_tp=4.0, risk_off_time=24.0),
        ExitPreset(name="ro_sl100_tp300_te24", enabled=True, risk_off_sl=1.0, risk_off_tp=3.0, risk_off_time=24.0),
        ExitPreset(name="ro_sl200_tp600_te24", enabled=True, risk_off_sl=2.0, risk_off_tp=6.0, risk_off_time=24.0),
    ]


def _make_grid_presets(
    *,
    base_sl: float,
    base_tp: float,
    base_time: Optional[float],
    risk_off_sl_values: List[float],
    risk_off_tp_values: List[float],
    risk_off_time_values: List[float],
) -> List[ExitPreset]:
    presets = [ExitPreset(name="baseline_exits", enabled=False, risk_off_sl=float(base_sl), risk_off_tp=float(base_tp), risk_off_time=base_time)]
    for sl in risk_off_sl_values:
        for tp in risk_off_tp_values:
            for te in risk_off_time_values:
                presets.append(
                    ExitPreset(
                        name=f"ro_sl{int(round(sl*100)):03d}_tp{int(round(tp*100)):03d}_te{int(round(te))}",
                        enabled=True,
                        risk_off_sl=float(sl),
                        risk_off_tp=float(tp),
                        risk_off_time=float(te),
                    )
                )
    return presets


def _make_mechanics_presets(base_sl: float, base_tp: float, base_time: Optional[float]) -> List[ExitPreset]:
    te = float(base_time) if base_time is not None else 72.0
    return [
        ExitPreset(name="baseline_exits", enabled=False, risk_off_sl=float(base_sl), risk_off_tp=float(base_tp), risk_off_time=base_time),
        ExitPreset(
            name="mech_ro150_tp400_te24_partial",
            enabled=True,
            risk_off_sl=1.5,
            risk_off_tp=4.0,
            risk_off_time=24.0,
            partial_tp_enabled=True,
            partial_tp1_atr_mult=2.5,
        ),
        ExitPreset(
            name="mech_ro150_tp400_te24_partial_be",
            enabled=True,
            risk_off_sl=1.5,
            risk_off_tp=4.0,
            risk_off_time=24.0,
            partial_tp_enabled=True,
            partial_tp1_atr_mult=2.5,
            move_sl_to_be_on_tp1=True,
        ),
        ExitPreset(
            name="mech_ro150_tp400_te24_partial_be_trail08",
            enabled=True,
            risk_off_sl=1.5,
            risk_off_tp=4.0,
            risk_off_time=24.0,
            partial_tp_enabled=True,
            partial_tp1_atr_mult=2.5,
            move_sl_to_be_on_tp1=True,
            trail_after_tp1=True,
            trail_atr_mult=0.8,
        ),
        ExitPreset(
            name=f"mech_ro150_tp600_te{int(te)}_dynneg",
            enabled=True,
            risk_off_sl=1.5,
            risk_off_tp=6.0,
            risk_off_time=te,
            dyn_exits_enabled=True,
            dyn_macd_hist_thresh=0.0,
            dyn_tp_mult_neg=0.70,
            dyn_sl_mult_neg=0.80,
            dyn_tp_mult_pos=1.05,
            dyn_sl_mult_pos=0.95,
        ),
        ExitPreset(
            name="mech_ro150_tp400_te24_flip_g2",
            enabled=True,
            risk_off_sl=1.5,
            risk_off_tp=4.0,
            risk_off_time=24.0,
            exit_on_regime_flip=True,
            regime_flip_grace_bars=2,
        ),
        ExitPreset(
            name="mech_ro150_tp400_te12_timeflip",
            enabled=True,
            risk_off_sl=1.5,
            risk_off_tp=4.0,
            risk_off_time=12.0,
            exit_on_regime_flip=True,
            regime_flip_grace_bars=2,
        ),
        ExitPreset(
            name="mech_ro200_tp600_te24_flip_g2",
            enabled=True,
            risk_off_sl=2.0,
            risk_off_tp=6.0,
            risk_off_time=24.0,
            exit_on_regime_flip=True,
            regime_flip_grace_bars=2,
        ),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JT-014 regime-conditioned exits sweep.")
    p.add_argument("--start", type=_parse_date, default="2023-01-01")
    p.add_argument("--end", type=_parse_date, default="latest")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--on-missing-symbol", choices=["fail", "skip"], default="skip")
    p.add_argument("--smoke-n", type=int, default=0)
    p.add_argument("--rolling-trades-window", type=int, default=250)
    p.add_argument("--initial-capital", type=float, default=2000.0)

    p.add_argument("--base-sl-atr", type=float, default=2.0)
    p.add_argument("--base-tp-atr", type=float, default=8.0)
    p.add_argument("--base-time-exit-hours", type=float, default=72.0)

    p.add_argument("--policy-block-when-down", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--policy-size-when-down", type=float, default=0.2)
    p.add_argument("--policy-probe-mult", type=float, default=0.25)

    p.add_argument("--preset-mode", choices=["compact", "grid", "mechanics"], default="compact")
    p.add_argument("--risk-off-sl-values", type=str, default="1.0,1.25,1.5,2.0")
    p.add_argument("--risk-off-tp-values", type=str, default="3.0,4.0,6.0,8.0")
    p.add_argument("--risk-off-time-values", type=str, default="24,48,72")
    p.add_argument("--notify-every-variants", type=int, default=2)
    p.add_argument("--notify-every-minutes", type=float, default=20.0)

    p.add_argument("--variant-retries", type=int, default=1)
    p.add_argument("--min-cash-pnl-frac-vs-baseline", type=float, default=0.80)
    p.add_argument("--max-sharpe-drop-vs-baseline", type=float, default=0.20)
    p.add_argument("--max-calmar-drop-vs-baseline", type=float, default=0.80)

    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument("--tg-auto-chat", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cfg = import_cfg()
    signals_dir, parquet_dir, results_dir, _meta_model_dir = resolve_paths(cfg)

    rid = _utc_run_id(a.run_id)
    notifier = TelegramNotifier.from_args(a, run_label=f"jt014:{rid}")
    print(f"[jt014] telegram notify: {notifier.status_line()}", flush=True)

    run_root = (results_dir / "jt014_regime_exits" / rid).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    end = a.end
    if str(end).lower() == "latest":
        end = latest_from_signals(signals_dir)

    sig = load_scoped_signals(signals_dir, a.start, end)
    sig = preflight_symbols(sig, parquet_dir, a.on_missing_symbol)
    if a.smoke_n and int(a.smoke_n) > 0:
        sig = sig.head(int(a.smoke_n)).copy()
    scoped_dir = write_signals_file(sig, run_root / "_scoped_signals")

    (run_root / "scoped_info.json").write_text(
        json.dumps(
            {
                "start": a.start,
                "end": end,
                "rows": int(len(sig)),
                "symbols": int(sig["symbol"].nunique()) if "symbol" in sig.columns else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    base_time = float(a.base_time_exit_hours)
    if a.preset_mode == "grid":
        presets = _make_grid_presets(
            base_sl=float(a.base_sl_atr),
            base_tp=float(a.base_tp_atr),
            base_time=base_time,
            risk_off_sl_values=_parse_float_list(a.risk_off_sl_values),
            risk_off_tp_values=_parse_float_list(a.risk_off_tp_values),
            risk_off_time_values=_parse_float_list(a.risk_off_time_values),
        )
    elif a.preset_mode == "mechanics":
        presets = _make_mechanics_presets(
            base_sl=float(a.base_sl_atr),
            base_tp=float(a.base_tp_atr),
            base_time=base_time,
        )
    else:
        presets = _make_compact_presets(
            base_sl=float(a.base_sl_atr),
            base_tp=float(a.base_tp_atr),
            base_time=base_time,
        )

    perf_overrides = {
        "BT_PROGRESS_ENABLED": False,
        "BT_DECISION_LOG_ENABLED": False,
        "BT_META_REPLAY_ENABLED": False,
        "USE_INTRABAR_1M": False,
        "DYN_EXITS_ENABLED": False,
    }

    notifier.send(
        "STARTED",
        body=(
            f"run_id={rid}\nstart={a.start} end={end}\n"
            f"signals_rows={len(sig)} symbols={int(sig['symbol'].nunique()) if 'symbol' in sig.columns else 0}\n"
            f"variants={len(presets)} jobs={max(1, int(a.jobs))} mode={a.preset_mode}\n"
            f"policy:block={bool(a.policy_block_when_down)} probe={float(a.policy_probe_mult)}"
        ),
    )

    def run_one(preset: ExitPreset) -> Dict[str, Any]:
        run_dir = run_root / preset.name
        run_dir.mkdir(parents=True, exist_ok=True)
        done = run_dir / "_DONE.json"
        metrics_path = run_dir / "metrics.json"

        if done.exists() and metrics_path.exists():
            try:
                prev = json.loads(metrics_path.read_text(encoding="utf-8"))
                if str(prev.get("status", "")).lower() == "ok":
                    return prev
            except Exception:
                pass

        max_attempts = max(1, int(a.variant_retries) + 1)
        last_rc = 999
        attempts: List[Dict[str, Any]] = []
        for i in range(max_attempts):
            overrides = {
                **perf_overrides,
                **preset.overrides(
                    base_sl=float(a.base_sl_atr),
                    base_tp=float(a.base_tp_atr),
                    base_time=base_time,
                    block_when_down=bool(a.policy_block_when_down),
                    size_when_down=float(a.policy_size_when_down),
                    probe_mult=float(a.policy_probe_mult),
                ),
            }
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "start": a.start,
                        "end": end,
                        "signals": str(scoped_dir),
                        "preset": {
                            "name": preset.name,
                            "enabled": preset.enabled,
                            "risk_off_sl": preset.risk_off_sl,
                            "risk_off_tp": preset.risk_off_tp,
                            "risk_off_time": preset.risk_off_time,
                        },
                        "attempt": i,
                        "overrides": overrides,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            rc, _ = run_backtest_subprocess(
                run_dir=run_dir,
                signals_dir=scoped_dir,
                start=a.start,
                end=end,
                overrides=overrides,
            )
            last_rc = int(rc)
            attempts.append({"attempt": i, "returncode": int(rc), "overrides": overrides})
            log_src = run_dir / "logs.txt"
            if log_src.exists():
                try:
                    shutil.copy2(log_src, run_dir / f"logs.attempt_{i}.txt")
                except Exception:
                    pass
            if int(rc) == 0:
                try:
                    met, windows = _compute_variant_metrics(
                        run_dir=run_dir,
                        rolling_trades_window=int(a.rolling_trades_window),
                        initial_capital=float(a.initial_capital),
                    )
                except RuntimeError as exc:
                    if "No trades file found under" in str(exc):
                        met, windows = _empty_metrics_payload(run_dir, reason="no_trades_file_no_executions")
                    else:
                        raise
                met.update(
                    {
                        "setting": preset.name,
                        "status": "ok",
                        "returncode": 0,
                        "REGIME_COND_EXITS_ENABLED": bool(preset.enabled),
                        "risk_off_sl_atr_mult": float(preset.risk_off_sl),
                        "risk_off_tp_atr_mult": float(preset.risk_off_tp),
                        "risk_off_time_exit_hours": float(preset.risk_off_time)
                        if preset.risk_off_time is not None
                        else None,
                        "base_sl_atr_mult": float(a.base_sl_atr),
                        "base_tp_atr_mult": float(a.base_tp_atr),
                        "base_time_exit_hours": float(base_time),
                        "policy_block_when_down": bool(a.policy_block_when_down),
                        "policy_size_when_down": float(a.policy_size_when_down),
                        "policy_probe_mult": float(a.policy_probe_mult),
                        "attempts": attempts,
                    }
                )
                windows = windows.copy()
                if not windows.empty:
                    windows["setting"] = preset.name
                    windows.to_csv(run_dir / "window_metrics.csv", index=False)
                metrics_path.write_text(json.dumps(met, indent=2, sort_keys=True), encoding="utf-8")
                done.write_text(json.dumps({"returncode": 0}, indent=2), encoding="utf-8")
                return met

            time.sleep(1.0)

        out = {
            "setting": preset.name,
            "status": "error",
            "returncode": int(last_rc),
            "REGIME_COND_EXITS_ENABLED": bool(preset.enabled),
            "risk_off_sl_atr_mult": float(preset.risk_off_sl),
            "risk_off_tp_atr_mult": float(preset.risk_off_tp),
            "risk_off_time_exit_hours": float(preset.risk_off_time) if preset.risk_off_time is not None else None,
            "attempts": attempts,
        }
        metrics_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        done.write_text(json.dumps({"returncode": int(last_rc)}, indent=2), encoding="utf-8")
        return out

    jobs = max(1, int(a.jobs))
    n_total = len(presets)
    notify_every = max(1, int(a.notify_every_variants))
    notify_every_sec = max(60.0, float(a.notify_every_minutes) * 60.0)
    t_start = time.time()
    t_last_notify = t_start
    progress_state = {
        "done": 0,
        "ok": 0,
        "err": 0,
    }

    def _emit_progress(force: bool = False) -> None:
        nonlocal t_last_notify
        done = int(progress_state["done"])
        if done <= 0:
            return
        now = time.time()
        by_count = (done % notify_every == 0) or (done >= n_total)
        by_time = (now - t_last_notify) >= notify_every_sec
        if not (force or by_count or by_time):
            return
        elapsed_min = max((now - t_start) / 60.0, 1e-9)
        rate = done / elapsed_min
        eta_min = (n_total - done) / rate if rate > 0 else float("nan")
        prog = {
            "done": done,
            "total": int(n_total),
            "ok": int(progress_state["ok"]),
            "err": int(progress_state["err"]),
            "elapsed_min": float(elapsed_min),
            "eta_min": float(eta_min) if np.isfinite(eta_min) else None,
        }
        (run_root / "_PROGRESS.json").write_text(json.dumps(prog, indent=2), encoding="utf-8")
        notifier.send(
            "PROGRESS",
            body=(
                f"run_id={rid}\nmode={a.preset_mode}\n"
                f"done={done}/{n_total} ok={prog['ok']} err={prog['err']}\n"
                f"elapsed_min={elapsed_min:.1f}\n"
                f"eta_min={(f'{eta_min:.1f}' if np.isfinite(eta_min) else 'n/a')}"
            ),
        )
        t_last_notify = now

    results: List[Dict[str, Any]] = []
    if jobs == 1:
        for p in presets:
            row = run_one(p)
            results.append(row)
            progress_state["done"] += 1
            if str(row.get("status", "")).lower() == "ok":
                progress_state["ok"] += 1
            else:
                progress_state["err"] += 1
            _emit_progress(force=False)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(run_one, p) for p in presets]
            for fut in as_completed(futs):
                row = fut.result()
                results.append(row)
                progress_state["done"] += 1
                if str(row.get("status", "")).lower() == "ok":
                    progress_state["ok"] += 1
                else:
                    progress_state["err"] += 1
                _emit_progress(force=False)
    _emit_progress(force=True)

    summary = pd.DataFrame(results)
    summary_path = run_root / "summary.csv"
    summary.to_csv(summary_path, index=False)

    n_ok = int((summary["status"].astype(str) == "ok").sum()) if "status" in summary.columns else 0
    n_err = int((summary["status"].astype(str) == "error").sum()) if "status" in summary.columns else (n_total - n_ok)

    if n_ok == 0:
        notifier.send(
            "FAILED",
            body=f"run_id={rid}\nno successful variants\nsummary={summary_path}",
        )
        print(str(run_root))
        return 1

    ok = summary[summary["status"] == "ok"].copy()
    base = ok[ok["setting"] == "baseline_exits"]
    if base.empty:
        base = ok.iloc[[0]].copy()
    base_row = base.iloc[0]

    base_cash = _safe_float(base_row.get("total_pnl_cash"), 0.0)
    base_sh = _safe_float(base_row.get("sharpe_daily"))
    base_calmar = _safe_float(base_row.get("calmar"))
    base_off_down = _safe_float(base_row.get("risk_off_downside_deviation_R"))
    base_off_worst = _safe_float(base_row.get("risk_off_worst_month_mean_pnl_R"))
    base_off_tail = _safe_float(base_row.get("risk_off_drawdown_tail_p05_R"))

    ok["improve_off_downside"] = pd.to_numeric(ok["risk_off_downside_deviation_R"], errors="coerce") < base_off_down
    ok["improve_off_worst_month"] = pd.to_numeric(ok["risk_off_worst_month_mean_pnl_R"], errors="coerce") > base_off_worst
    ok["improve_off_tail_dd"] = pd.to_numeric(ok["risk_off_drawdown_tail_p05_R"], errors="coerce") > base_off_tail
    ok["improve_any_off_metric"] = ok[
        ["improve_off_downside", "improve_off_worst_month", "improve_off_tail_dd"]
    ].any(axis=1)

    if base_cash > 0:
        ok["pass_cash_floor"] = pd.to_numeric(ok["total_pnl_cash"], errors="coerce") >= float(a.min_cash_pnl_frac_vs_baseline) * base_cash
    else:
        ok["pass_cash_floor"] = True

    if np.isfinite(base_sh):
        ok["pass_sharpe_floor"] = pd.to_numeric(ok["sharpe_daily"], errors="coerce") >= (base_sh - float(a.max_sharpe_drop_vs_baseline))
    else:
        ok["pass_sharpe_floor"] = True
    if np.isfinite(base_calmar):
        ok["pass_calmar_floor"] = pd.to_numeric(ok["calmar"], errors="coerce") >= (base_calmar - float(a.max_calmar_drop_vs_baseline))
    else:
        ok["pass_calmar_floor"] = True

    ok["candidate"] = ok["improve_any_off_metric"] & ok["pass_cash_floor"] & ok["pass_sharpe_floor"] & ok["pass_calmar_floor"]

    ranked = ok.sort_values(
        by=[
            "candidate",
            "improve_off_downside",
            "improve_off_worst_month",
            "risk_off_downside_deviation_R",
            "risk_off_worst_month_mean_pnl_R",
            "total_pnl_cash",
        ],
        ascending=[False, False, False, True, False, False],
        kind="mergesort",
    )
    base_eval = ok[ok["setting"] == str(base_row.get("setting"))].copy()
    base_eval_row = base_eval.iloc[0] if not base_eval.empty else base_row

    strict_pool = ranked[ranked["candidate"]].copy()
    if not strict_pool.empty:
        selected = strict_pool.iloc[0]
        selection_mode = "strict_candidate"
    else:
        selected = base_eval_row.copy()
        selection_mode = "fallback_baseline_no_strict_candidate"

    eval_path = run_root / "evaluation.csv"
    ranked.to_csv(eval_path, index=False)

    # Build combined walk-forward-by-month table
    win_rows: List[pd.DataFrame] = []
    for setting in ok["setting"].tolist():
        p = run_root / str(setting) / "window_metrics.csv"
        if p.exists():
            w = pd.read_csv(p, low_memory=False)
            win_rows.append(w)
    all_windows = pd.concat(win_rows, ignore_index=True) if win_rows else pd.DataFrame()
    if not all_windows.empty:
        all_windows.to_csv(run_root / "walkforward_windows.csv", index=False)

    sens_cols = [
        "setting",
        "REGIME_COND_EXITS_ENABLED",
        "risk_off_sl_atr_mult",
        "risk_off_tp_atr_mult",
        "risk_off_time_exit_hours",
        "total_pnl_cash",
        "sharpe_daily",
        "calmar",
        "risk_off_total_pnl_R",
        "risk_off_worst_month_mean_pnl_R",
        "risk_off_downside_deviation_R",
        "risk_off_drawdown_tail_p05_R",
        "candidate",
    ]
    for c in sens_cols:
        if c not in ranked.columns:
            ranked[c] = np.nan
    sensitivity = ranked[sens_cols].copy()
    sensitivity.to_csv(run_root / "sensitivity_matrix.csv", index=False)

    rec = {
        "run_id": rid,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "start": a.start,
            "end": end,
            "jobs": int(jobs),
            "signals_rows": int(len(sig)),
            "signals_symbols": int(sig["symbol"].nunique()) if "symbol" in sig.columns else 0,
            "preset_mode": str(a.preset_mode),
            "variant_retries": int(a.variant_retries),
            "policy_block_when_down": bool(a.policy_block_when_down),
            "policy_size_when_down": float(a.policy_size_when_down),
            "policy_probe_mult": float(a.policy_probe_mult),
        },
        "baseline_setting": str(base_row.get("setting")),
        "baseline_metrics": base_eval_row.to_dict(),
        "recommended_setting": str(selected.get("setting")),
        "recommended_metrics": selected.to_dict(),
        "selection_mode": selection_mode,
        "acceptance_check": {
            "improves_risk_off_downside_or_tail": bool(selected.get("improve_any_off_metric")),
            "passes_cash_floor": bool(selected.get("pass_cash_floor")),
            "passes_sharpe_floor": bool(selected.get("pass_sharpe_floor")),
            "passes_calmar_floor": bool(selected.get("pass_calmar_floor")),
            "jt011_hard_gate_note": (
                "JT-011 hard gates are model calibration/lift gates and are not directly "
                "recomputed in this exit-only sweep; proxy guardrails are applied."
            ),
        },
        "pass_stats": {
            "variants_total": int(n_total),
            "variants_ok": int(n_ok),
            "variants_error": int(n_err),
        },
    }
    rec_path = run_root / "recommendation.json"
    rec_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# JT-014 Regime-Conditioned Exits: {rid}")
    lines.append("")
    lines.append("## Runtime")
    lines.append(f"- variants_ok: `{n_ok}/{n_total}`")
    lines.append(f"- variants_error: `{n_err}`")
    lines.append(f"- mode: `{a.preset_mode}`")
    lines.append("")
    lines.append("## Baseline")
    lines.append(f"- setting: `{base_row.get('setting')}`")
    lines.append(f"- total_pnl_cash: `{base_row.get('total_pnl_cash')}`")
    lines.append(f"- sharpe_daily: `{base_row.get('sharpe_daily')}`")
    lines.append(f"- calmar: `{base_row.get('calmar')}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- setting: `{selected.get('setting')}`")
    lines.append(f"- selection_mode: `{selection_mode}`")
    lines.append(f"- improves_risk_off_metric: `{bool(selected.get('improve_any_off_metric'))}`")
    lines.append(f"- pass_cash_floor: `{bool(selected.get('pass_cash_floor'))}`")
    lines.append(f"- pass_sharpe_floor: `{bool(selected.get('pass_sharpe_floor'))}`")
    lines.append(f"- pass_calmar_floor: `{bool(selected.get('pass_calmar_floor'))}`")
    lines.append("")
    lines.append("## Sensitivity Matrix")
    lines.append("")
    lines.append(sensitivity.to_markdown(index=False))
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    if n_err > 0:
        notifier.send(
            "WARN",
            body=f"run_id={rid}\nstatus=partial\nok={n_ok}/{n_total}\nrecommendation={rec_path}",
        )
    else:
        notifier.send(
            "DONE",
            body=f"run_id={rid}\nstatus=ok\nok={n_ok}/{n_total}\nrecommended={selected.get('setting')}\nrecommendation={rec_path}",
        )

    print(str(run_root))
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
