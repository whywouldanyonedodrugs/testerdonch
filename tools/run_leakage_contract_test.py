#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from indicators import resample_ohlcv
from shared_utils import load_parquet_data


@dataclass(frozen=True)
class Family:
    name: str
    timeframe: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTF as-of leakage contract test runner.")
    p.add_argument("--out-report", required=True)
    p.add_argument("--out-traces-dir", required=True)
    p.add_argument("--start", default="2025-09-01")
    p.add_argument("--end", default="2026-03-05")
    p.add_argument("--sample-n", type=int, default=10_000)
    p.add_argument("--month", default="2025-11")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,DOGEUSDT")
    return p.parse_args()


def _norm_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out.index = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out = out.drop(columns=["timestamp"], errors="ignore")
        else:
            raise ValueError("df has neither DatetimeIndex nor timestamp column")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _tf_open_from_close(close_ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    return close_ts - pd.Timedelta(timeframe)


def _run_contract_for_symbol(
    symbol: str,
    ts_idx: pd.DatetimeIndex,
    families: List[Family],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    base = load_parquet_data(
        symbol,
        start_date=str(ts_idx.min().date() - pd.Timedelta(days=30)),
        end_date=str(ts_idx.max().date() + pd.Timedelta(days=2)),
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume"],
    )
    base = _norm_index(base)
    if base.empty:
        return [], [], {"rows": 0, "viol_close": 0, "viol_open": 0}

    traces: List[Dict[str, object]] = []
    violations: List[Dict[str, object]] = []
    stats = {"rows": 0, "viol_close": 0, "viol_open": 0}

    tf_cache: Dict[str, pd.DataFrame] = {}
    for fam in families:
        tf = fam.timeframe
        if tf not in tf_cache:
            tf_cache[tf] = resample_ohlcv(base, tf)

    for decision_ts in ts_idx:
        for fam in families:
            tf = fam.timeframe
            tf_df = tf_cache[tf]
            row = tf_df.loc[:decision_ts].iloc[-1:] if not tf_df.empty else pd.DataFrame()
            if row.empty:
                continue
            close_ts = pd.Timestamp(row.index[0]).tz_convert("UTC")
            open_ts = _tf_open_from_close(close_ts, tf)
            stats["rows"] += 1

            viol_close = bool(close_ts > decision_ts)
            viol_open = bool(open_ts >= decision_ts)
            if viol_close:
                stats["viol_close"] += 1
            if viol_open:
                stats["viol_open"] += 1

            r = {
                "symbol": symbol,
                "family": fam.name,
                "timeframe": tf,
                "decision_ts": decision_ts.isoformat(),
                "selected_bar_open_ts": open_ts.isoformat(),
                "selected_bar_close_ts": close_ts.isoformat(),
                "open": float(row["open"].iloc[0]),
                "high": float(row["high"].iloc[0]),
                "low": float(row["low"].iloc[0]),
                "close": float(row["close"].iloc[0]),
                "volume": float(row["volume"].iloc[0]),
                "viol_close_after_decision": int(viol_close),
                "viol_open_not_strictly_before_decision": int(viol_open),
            }
            traces.append(r)
            if viol_close or viol_open:
                violations.append(r)
    return traces, violations, stats


def main() -> int:
    a = _parse_args()
    rng = random.Random(int(a.seed))

    out_report = Path(a.out_report).resolve()
    out_dir = Path(a.out_traces_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in str(a.symbols).split(",") if s.strip()]
    if len(symbols) < 5:
        raise SystemExit("Need at least 5 symbols (ETH,BTC + 3 liquid alts).")

    families = [
        Family("atr_rsi_adx", "1h"),
        Family("macd_1h", "1h"),
        Family("macd_4h", "4h"),
        Family("daily_regime", "1D"),
    ]

    all_stats: List[Dict[str, object]] = []
    all_viol: List[Dict[str, object]] = []
    sampled_trace_pool: List[Dict[str, object]] = []

    # Random sample per symbol
    for sym in symbols:
        df = load_parquet_data(
            sym,
            start_date=a.start,
            end_date=a.end,
            drop_last_partial=True,
            columns=["open", "high", "low", "close", "volume"],
        )
        df = _norm_index(df)
        if df.empty:
            all_stats.append({"symbol": sym, "kind": "random", "rows": 0, "viol_close": 0, "viol_open": 0, "note": "empty"})
            continue
        idx = list(df.index)
        n = int(a.sample_n)
        if len(idx) <= n:
            sampled = pd.DatetimeIndex(idx)
        else:
            sampled = pd.DatetimeIndex(rng.sample(idx, n))
            sampled = sampled.sort_values()

        traces, viol, st = _run_contract_for_symbol(sym, sampled, families)
        all_stats.append({"symbol": sym, "kind": "random", **st})
        all_viol.extend(viol)
        sampled_trace_pool.extend(traces)

    # Exhaustive month for ETH
    eth = "ETHUSDT"
    month_start = pd.Timestamp(f"{a.month}-01", tz="UTC")
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_m = load_parquet_data(
        eth,
        start_date=month_start.isoformat(),
        end_date=month_end.isoformat(),
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume"],
    )
    df_m = _norm_index(df_m)
    traces_m, viol_m, st_m = _run_contract_for_symbol(eth, df_m.index, families)
    all_stats.append({"symbol": eth, "kind": "exhaustive_month", "month": str(a.month), **st_m})
    all_viol.extend(viol_m)
    sampled_trace_pool.extend(traces_m)

    stats_df = pd.DataFrame(all_stats)
    viol_df = pd.DataFrame(all_viol)
    stats_df.to_csv(out_dir / "contract_stats.csv", index=False)
    if not viol_df.empty:
        viol_df.to_csv(out_dir / "contract_violations.csv", index=False)

    # 20-50 human auditable traces
    if sampled_trace_pool:
        # prioritize violations if any, then fill with random non-violating
        viol_rows = [r for r in sampled_trace_pool if (r["viol_close_after_decision"] or r["viol_open_not_strictly_before_decision"])]
        non_viol_rows = [r for r in sampled_trace_pool if not (r["viol_close_after_decision"] or r["viol_open_not_strictly_before_decision"])]
        out_rows: List[Dict[str, object]] = []
        out_rows.extend(viol_rows[:50])
        need = max(20, 40 - len(out_rows))
        if need > 0:
            if len(non_viol_rows) <= need:
                out_rows.extend(non_viol_rows)
            else:
                out_rows.extend(rng.sample(non_viol_rows, need))
        out_rows = out_rows[:50]
        pd.DataFrame(out_rows).to_csv(out_dir / "sample_traces.csv", index=False)

    total_rows = int(stats_df["rows"].sum()) if not stats_df.empty else 0
    total_viol_close = int(stats_df["viol_close"].sum()) if not stats_df.empty else 0
    total_viol_open = int(stats_df["viol_open"].sum()) if not stats_df.empty else 0
    passed = (total_viol_close == 0 and total_viol_open == 0)

    lines: List[str] = []
    lines.append("# Leakage Contract Test v1")
    lines.append("")
    lines.append(f"- start: `{a.start}`")
    lines.append(f"- end: `{a.end}`")
    lines.append(f"- random_sample_n_per_symbol: `{int(a.sample_n)}`")
    lines.append(f"- month_exhaustive_symbol: `{eth}`")
    lines.append(f"- month_exhaustive_window: `{a.month}`")
    lines.append(f"- symbols: `{','.join(symbols)}`")
    lines.append(f"- total_checks: `{total_rows}`")
    lines.append(f"- violations_selected_bar_close_gt_decision: `{total_viol_close}`")
    lines.append(f"- violations_selected_bar_open_ge_decision: `{total_viol_open}`")
    lines.append(f"- verdict: `{'PASS' if passed else 'FAIL'}`")
    lines.append("")
    lines.append("## Stats By Symbol/Scope")
    lines.append("")
    if stats_df.empty:
        lines.append("_no rows_")
    else:
        lines.append(stats_df.to_markdown(index=False))
    lines.append("")
    if not viol_df.empty:
        lines.append("## Worst Violations (first 20)")
        lines.append("")
        lines.append(viol_df.head(20).to_markdown(index=False))
        lines.append("")
    lines.append("## Trace Artifacts")
    lines.append("")
    lines.append(f"- `{(out_dir / 'contract_stats.csv').as_posix()}`")
    if (out_dir / "contract_violations.csv").exists():
        lines.append(f"- `{(out_dir / 'contract_violations.csv').as_posix()}`")
    if (out_dir / "sample_traces.csv").exists():
        lines.append(f"- `{(out_dir / 'sample_traces.csv').as_posix()}`")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_report))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
