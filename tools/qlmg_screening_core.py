#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


GB = 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    if isinstance(ts, pd.DatetimeIndex):
        raise TypeError("expected scalar timestamp")
    return pd.Timestamp(ts)


@dataclass(frozen=True)
class ResourceSnapshot:
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    mem_total_bytes: int | None
    mem_available_bytes: int | None

    @property
    def free_gb(self) -> float:
        return self.free_bytes / GB


def read_memory_snapshot() -> tuple[int | None, int | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    vals: dict[str, int] = {}
    for line in meminfo.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                vals[parts[0][:-1]] = int(parts[1]) * 1024
            except ValueError:
                pass
    return vals.get("MemTotal"), vals.get("MemAvailable")


def resource_snapshot(path: str | Path) -> ResourceSnapshot:
    usage = shutil.disk_usage(path)
    mem_total, mem_available = read_memory_snapshot()
    return ResourceSnapshot(
        path=str(path),
        total_bytes=int(usage.total),
        used_bytes=int(usage.used),
        free_bytes=int(usage.free),
        mem_total_bytes=mem_total,
        mem_available_bytes=mem_available,
    )


def check_resource_guard(
    snapshot: ResourceSnapshot,
    *,
    estimated_output_gb: float,
    hard_free_gb: float = 5.0,
    warn_free_gb: float = 7.0,
    hard_stage_output_gb: float = 20.0,
    allow_large_output: bool = False,
) -> dict[str, Any]:
    status = "pass"
    reasons: list[str] = []
    warnings: list[str] = []
    if snapshot.free_gb < hard_free_gb:
        status = "hard_stop"
        reasons.append(f"free_disk_gb_below_{hard_free_gb:g}")
    elif snapshot.free_gb < warn_free_gb:
        warnings.append(f"free_disk_gb_below_{warn_free_gb:g}")
    if estimated_output_gb > hard_stage_output_gb and not allow_large_output:
        status = "hard_stop"
        reasons.append(f"estimated_output_gb_above_{hard_stage_output_gb:g}_without_allow_large_output")
    return {
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "free_disk_gb": snapshot.free_gb,
        "estimated_output_gb": float(estimated_output_gb),
        "hard_free_gb": float(hard_free_gb),
        "warn_free_gb": float(warn_free_gb),
        "hard_stage_output_gb": float(hard_stage_output_gb),
        "allow_large_output": bool(allow_large_output),
    }


@dataclass(frozen=True)
class FundingEvent:
    timestamp: pd.Timestamp
    rate: float
    mark_price: float
    source: str = "proxy"


@dataclass(frozen=True)
class ReplayConfig:
    side: str
    decision_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float | None
    max_holding_hours: float
    qty: float = 1.0
    fee_bps_round_trip: float = 0.0
    slippage_bps_round_trip: float = 0.0
    leverage: float = 3.0
    maintenance_margin_fraction: float = 0.005
    trailing_stop_distance: float | None = None
    delist_ts: pd.Timestamp | None = None
    tie_breaker: str = "sl_wins"


@dataclass(frozen=True)
class ReplayResult:
    side: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    holding_hours: float
    gross_pnl: float
    fee_pnl: float
    slippage_pnl: float
    funding_pnl: float
    net_pnl: float
    gross_R: float
    fee_R: float
    slippage_R: float
    funding_R: float
    net_R: float
    liquidation_flag: bool
    delist_flag: bool
    funding_events_crossed: int
    ambiguity_flag: bool
    data_quality_flags: str

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["entry_ts"] = str(self.entry_ts)
        d["exit_ts"] = str(self.exit_ts)
        return d


def _ensure_side(side: str) -> str:
    s = str(side).lower().strip()
    if s not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    return s


def risk_per_unit(side: str, entry_price: float, stop_price: float) -> float:
    _ensure_side(side)
    risk = abs(float(entry_price) - float(stop_price))
    if not math.isfinite(risk) or risk <= 0:
        raise ValueError("stop distance must be positive")
    return risk


def gross_pnl(side: str, entry_price: float, exit_price: float, qty: float = 1.0) -> float:
    side = _ensure_side(side)
    if side == "long":
        return (float(exit_price) - float(entry_price)) * float(qty)
    return (float(entry_price) - float(exit_price)) * float(qty)


def funding_cashflow(side: str, qty: float, mark_price: float, funding_rate: float) -> float:
    side = _ensure_side(side)
    sign = -1.0 if side == "long" else 1.0
    return sign * float(qty) * float(mark_price) * float(funding_rate)


def liquidation_price(entry_price: float, side: str, leverage: float, maintenance_margin_fraction: float = 0.005) -> float:
    side = _ensure_side(side)
    lev = float(leverage)
    mm = float(maintenance_margin_fraction)
    if lev <= 0:
        raise ValueError("leverage must be positive")
    adverse = max((1.0 / lev) - mm, 0.0)
    if side == "long":
        return float(entry_price) * (1.0 - adverse)
    return float(entry_price) * (1.0 + adverse)


def _bar_value(row: pd.Series, preferred: str, fallback: str) -> float:
    val = row.get(preferred, np.nan)
    if pd.isna(val):
        val = row.get(fallback, np.nan)
    return float(val)


def _tp_sl_hit(side: str, high: float, low: float, stop_price: float, target_price: float | None) -> tuple[bool, bool]:
    side = _ensure_side(side)
    if side == "long":
        sl_hit = low <= stop_price
        tp_hit = False if target_price is None else high >= target_price
    else:
        sl_hit = high >= stop_price
        tp_hit = False if target_price is None else low <= target_price
    return tp_hit, sl_hit


def _apply_trailing_stop(side: str, stop_price: float, trail_distance: float | None, best_price: float, high: float, low: float) -> tuple[float, float]:
    if trail_distance is None or trail_distance <= 0:
        return stop_price, best_price
    side = _ensure_side(side)
    if side == "long":
        best_price = max(best_price, high)
        stop_price = max(stop_price, best_price - trail_distance)
    else:
        best_price = min(best_price, low)
        stop_price = min(stop_price, best_price + trail_distance)
    return stop_price, best_price


def replay_trade(
    bars: pd.DataFrame,
    config: ReplayConfig,
    funding_events: Sequence[FundingEvent] | None = None,
) -> ReplayResult:
    side = _ensure_side(config.side)
    if config.tie_breaker not in {"sl_wins", "tp_wins"}:
        raise ValueError("tie_breaker must be sl_wins or tp_wins")
    if not isinstance(bars.index, pd.DatetimeIndex):
        if "timestamp" not in bars.columns:
            raise ValueError("bars must have DatetimeIndex or timestamp column")
        bars = bars.set_index(pd.to_datetime(bars["timestamp"], utc=True))
    bars = bars.sort_index()
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")
    start = to_utc_ts(config.entry_ts)
    max_exit = start + pd.Timedelta(hours=float(config.max_holding_hours))
    scan = bars[(bars.index > start) & (bars.index <= max_exit)]
    if scan.empty:
        raise ValueError("no future bars after entry_ts")

    liq_price = liquidation_price(config.entry_price, side, config.leverage, config.maintenance_margin_fraction)
    stop_price = float(config.stop_price)
    target_price = None if config.target_price is None else float(config.target_price)
    exit_ts = scan.index[-1]
    exit_price = float(scan.iloc[-1].get("close"))
    exit_reason = "time_exit"
    liquidation_flag = False
    delist_flag = False
    ambiguity_flag = False
    best_price = float(config.entry_price)

    for ts, row in scan.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        mark_high = _bar_value(row, "mark_high", "high")
        mark_low = _bar_value(row, "mark_low", "low")
        if config.delist_ts is not None and ts >= to_utc_ts(config.delist_ts):
            exit_ts = ts
            exit_price = float(row.get("close"))
            exit_reason = "delist_settlement"
            delist_flag = True
            break
        if side == "long" and mark_low <= liq_price:
            exit_ts = ts
            exit_price = liq_price
            exit_reason = "liquidation"
            liquidation_flag = True
            break
        if side == "short" and mark_high >= liq_price:
            exit_ts = ts
            exit_price = liq_price
            exit_reason = "liquidation"
            liquidation_flag = True
            break
        stop_price, best_price = _apply_trailing_stop(side, stop_price, config.trailing_stop_distance, best_price, high, low)
        tp_hit, sl_hit = _tp_sl_hit(side, high, low, stop_price, target_price)
        if tp_hit and sl_hit:
            ambiguity_flag = True
            if config.tie_breaker == "tp_wins":
                exit_reason = "target"
                exit_price = float(target_price)
            else:
                exit_reason = "stop"
                exit_price = float(stop_price)
            exit_ts = ts
            break
        if sl_hit:
            exit_ts = ts
            exit_price = float(stop_price)
            exit_reason = "stop"
            break
        if tp_hit:
            exit_ts = ts
            exit_price = float(target_price)
            exit_reason = "target"
            break

    qty = float(config.qty)
    gross = gross_pnl(side, config.entry_price, exit_price, qty)
    notional = abs(float(config.entry_price) * qty)
    fee = -notional * (float(config.fee_bps_round_trip) / 10000.0)
    slip = -notional * (float(config.slippage_bps_round_trip) / 10000.0)
    crossed = 0
    fund = 0.0
    for ev in funding_events or []:
        ev_ts = to_utc_ts(ev.timestamp)
        if start < ev_ts <= exit_ts:
            crossed += 1
            fund += funding_cashflow(side, qty, ev.mark_price, ev.rate)
    risk = risk_per_unit(side, config.entry_price, config.stop_price) * qty
    net = gross + fee + slip + fund
    holding = (exit_ts - start).total_seconds() / 3600.0
    flags = []
    if "mark_high" not in scan.columns or "mark_low" not in scan.columns:
        flags.append("mark_proxy_from_last_ohlc")
    if not funding_events:
        flags.append("funding_events_absent_or_proxy_zero")
    return ReplayResult(
        side=side,
        entry_ts=start,
        exit_ts=exit_ts,
        entry_price=float(config.entry_price),
        exit_price=float(exit_price),
        exit_reason=exit_reason,
        holding_hours=float(holding),
        gross_pnl=float(gross),
        fee_pnl=float(fee),
        slippage_pnl=float(slip),
        funding_pnl=float(fund),
        net_pnl=float(net),
        gross_R=float(gross / risk),
        fee_R=float(fee / risk),
        slippage_R=float(slip / risk),
        funding_R=float(fund / risk),
        net_R=float(net / risk),
        liquidation_flag=bool(liquidation_flag),
        delist_flag=bool(delist_flag),
        funding_events_crossed=int(crossed),
        ambiguity_flag=bool(ambiguity_flag),
        data_quality_flags=";".join(flags),
    )


def summarize_trades(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "net_R": 0.0,
            "gross_R": 0.0,
            "PF": 0.0,
            "max_dd_R": 0.0,
            "median_R": np.nan,
            "funding_R": 0.0,
            "slippage_R": 0.0,
            "liquidation_count": 0,
            "delist_count": 0,
        }
    net = pd.to_numeric(df.get("net_R"), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(df.get("gross_R"), errors="coerce").fillna(0.0)
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    equity = net.cumsum()
    dd = (equity - equity.cummax()).min() if not equity.empty else 0.0
    return {
        "trades": int(len(df)),
        "net_R": float(net.sum()),
        "gross_R": float(gross.sum()),
        "PF": float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0),
        "max_dd_R": float(dd),
        "median_R": float(net.median()) if len(net) else np.nan,
        "funding_R": float(pd.to_numeric(df.get("funding_R", 0.0), errors="coerce").fillna(0.0).sum()),
        "slippage_R": float(pd.to_numeric(df.get("slippage_R", 0.0), errors="coerce").fillna(0.0).sum()),
        "liquidation_count": int(pd.Series(df.get("liquidation_flag", False)).fillna(False).astype(bool).sum()),
        "delist_count": int(pd.Series(df.get("delist_flag", False)).fillna(False).astype(bool).sum()),
    }


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
