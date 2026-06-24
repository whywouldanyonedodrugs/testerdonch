from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BYBIT_BASE_URL = "https://api.bybit.com"
ORDERBOOK_ENDPOINT = "/v5/market/orderbook"
RECENT_TRADE_ENDPOINT = "/v5/market/recent-trade"
DEFAULT_CATEGORY = "linear"
DEFAULT_ORDERBOOK_LIMIT = 50
DEFAULT_TRADE_LIMIT = 1000
DEFAULT_RAW_ROOT = Path("/opt/parquet/bybit_microstructure_raw")
DEFAULT_SILVER_ROOT = Path("/opt/parquet/bybit_microstructure_5m")
MICROSTRUCTURE_CONTRACT_VERSION = "perp_microstructure_v2"


def _api_get(endpoint: str, params: Mapping[str, Any], *, timeout_sec: float = 20.0, retries: int = 3) -> dict[str, Any]:
    url = f"{BYBIT_BASE_URL}{endpoint}?{urlencode(dict(params))}"
    last_error = ""
    for attempt in range(1, int(retries) + 1):
        try:
            req = Request(url, headers={"User-Agent": "testerdonch-perp-microstructure-v2"})
            with urlopen(req, timeout=float(timeout_sec)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            if int(obj.get("retCode", -1)) == 0:
                return obj
            last_error = f"retCode={obj.get('retCode')} retMsg={obj.get('retMsg')}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(4.0, 0.5 * attempt))
    raise RuntimeError(f"Bybit API failed {endpoint} params={dict(params)}: {last_error}")


def fetch_orderbook_snapshot(symbol: str, *, category: str = DEFAULT_CATEGORY, limit: int = DEFAULT_ORDERBOOK_LIMIT) -> pd.DataFrame:
    obj = _api_get(ORDERBOOK_ENDPOINT, {"category": category, "symbol": symbol.upper(), "limit": int(limit)})
    return parse_orderbook_response(obj, observed_available_ts=pd.Timestamp.utcnow())


def fetch_recent_public_trades(symbol: str, *, category: str = DEFAULT_CATEGORY, limit: int = DEFAULT_TRADE_LIMIT) -> pd.DataFrame:
    obj = _api_get(RECENT_TRADE_ENDPOINT, {"category": category, "symbol": symbol.upper(), "limit": int(limit)})
    return parse_recent_trade_response(obj, observed_available_ts=pd.Timestamp.utcnow())


def parse_orderbook_response(response: Mapping[str, Any], *, observed_available_ts: pd.Timestamp) -> pd.DataFrame:
    result = response.get("result") or {}
    symbol = str(result.get("s") or "").upper()
    observed = pd.Timestamp(observed_available_ts).tz_convert("UTC") if pd.Timestamp(observed_available_ts).tzinfo else pd.Timestamp(observed_available_ts, tz="UTC")
    system_ts = _ms_ts(result.get("ts"))
    matching_ts = _ms_ts(result.get("cts"))
    rows: list[dict[str, Any]] = []
    for side_name, rows_raw in (("bid", result.get("b") or []), ("ask", result.get("a") or [])):
        for level, raw in enumerate(rows_raw, start=1):
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "side": side_name,
                    "level": int(level),
                    "price": _float_or_nan(raw[0]),
                    "size": _float_or_nan(raw[1]),
                    "observed_available_ts": observed,
                    "system_ts": system_ts,
                    "matching_engine_ts": matching_ts,
                    "update_id": result.get("u"),
                    "cross_sequence": result.get("seq"),
                    "source_interval": "snapshot",
                }
            )
    return pd.DataFrame(rows)


def parse_recent_trade_response(response: Mapping[str, Any], *, observed_available_ts: pd.Timestamp) -> pd.DataFrame:
    result = response.get("result") or {}
    observed = pd.Timestamp(observed_available_ts).tz_convert("UTC") if pd.Timestamp(observed_available_ts).tzinfo else pd.Timestamp(observed_available_ts, tz="UTC")
    rows: list[dict[str, Any]] = []
    for raw in result.get("list") or []:
        if not isinstance(raw, Mapping):
            continue
        rows.append(
            {
                "exec_id": str(raw.get("execId") or ""),
                "symbol": str(raw.get("symbol") or "").upper(),
                "price": _float_or_nan(raw.get("price")),
                "size": _float_or_nan(raw.get("size")),
                "side": str(raw.get("side") or ""),
                "trade_ts": _ms_ts(raw.get("time")),
                "is_block_trade": bool(raw.get("isBlockTrade", False)),
                "is_rpi_trade": bool(raw.get("isRPITrade", False)),
                "sequence": raw.get("seq"),
                "observed_available_ts": observed,
                "source_interval": "recent_trade",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["trade_ts", "exec_id"], kind="mergesort").drop_duplicates(["exec_id"], keep="last")
    return out.reset_index(drop=True)


def write_raw_capture(symbol: str, *, raw_root: Path | str = DEFAULT_RAW_ROOT, orderbook: pd.DataFrame | None = None, trades: pd.DataFrame | None = None) -> dict[str, str]:
    root = Path(raw_root)
    now = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out: dict[str, str] = {}
    if orderbook is not None and not orderbook.empty:
        path = root / "orderbook" / f"symbol={symbol.upper()}" / f"capture_{now}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        orderbook.to_parquet(path, index=False)
        out["orderbook"] = str(path)
    if trades is not None and not trades.empty:
        path = root / "recent_trades" / f"symbol={symbol.upper()}" / f"capture_{now}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(path, index=False)
        out["recent_trades"] = str(path)
    return out


def aggregate_orderbook_snapshot_features(snapshots: pd.DataFrame, *, max_age: pd.Timedelta = pd.Timedelta(seconds=60)) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=_orderbook_silver_columns())
    df = snapshots.copy()
    df["observed_available_ts"] = pd.to_datetime(df["observed_available_ts"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["observed_available_ts", "price", "size", "symbol", "side", "level"])
    if df.empty:
        return pd.DataFrame(columns=_orderbook_silver_columns())
    grouped = []
    for (symbol, observed_ts), g in df.groupby(["symbol", "observed_available_ts"], dropna=False):
        bids = g[g["side"].astype(str).str.lower().eq("bid")].sort_values("level")
        asks = g[g["side"].astype(str).str.lower().eq("ask")].sort_values("level")
        best_bid = float(bids.iloc[0]["price"]) if len(bids) else np.nan
        best_ask = float(asks.iloc[0]["price"]) if len(asks) else np.nan
        mid = (best_bid + best_ask) / 2.0 if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan
        bid_depth_5 = float(bids.head(5)["size"].sum()) if len(bids) else 0.0
        ask_depth_5 = float(asks.head(5)["size"].sum()) if len(asks) else 0.0
        denom = bid_depth_5 + ask_depth_5
        grouped.append(
            {
                "symbol": symbol,
                "decision_ts": pd.Timestamp(observed_ts).ceil("5min"),
                "orderbook_observed_available_ts": observed_ts,
                "orderbook_source_interval": "snapshot",
                "orderbook_max_age_seconds": float(pd.to_timedelta(max_age).total_seconds()),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid,
                "spread_pct": ((best_ask - best_bid) / mid) if np.isfinite(mid) and mid else np.nan,
                "top5_bid_depth": bid_depth_5,
                "top5_ask_depth": ask_depth_5,
                "top5_depth_imbalance": ((bid_depth_5 - ask_depth_5) / denom) if denom > 0 else np.nan,
                "orderbook_reject_reason": "",
            }
        )
    out = pd.DataFrame(grouped).sort_values(["symbol", "decision_ts", "orderbook_observed_available_ts"], kind="mergesort")
    out = out.drop_duplicates(["symbol", "decision_ts"], keep="last")
    age = (out["decision_ts"] - out["orderbook_observed_available_ts"]).dt.total_seconds()
    stale_after = age < 0
    stale_old = age > float(pd.to_timedelta(max_age).total_seconds())
    # A snapshot captured after a decision cutoff, or too long before it, is not usable.
    out.loc[stale_after, "orderbook_reject_reason"] = "snapshot_after_decision_cutoff"
    out.loc[stale_old, "orderbook_reject_reason"] = "snapshot_stale"
    return out[_orderbook_silver_columns()].reset_index(drop=True)


def aggregate_recent_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=_trade_silver_columns())
    df = trades.copy()
    df["trade_ts"] = pd.to_datetime(df["trade_ts"], utc=True, errors="coerce")
    df["observed_available_ts"] = pd.to_datetime(df["observed_available_ts"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["symbol", "trade_ts", "observed_available_ts", "price", "size"])
    if "exec_id" in df.columns:
        df = df.drop_duplicates("exec_id", keep="last")
    df["decision_ts"] = df["trade_ts"].dt.ceil("5min")
    # Use only trades observed no later than the closed-bar decision timestamp.
    usable = df["observed_available_ts"] <= df["decision_ts"]
    df = df[usable].copy()
    if df.empty:
        return pd.DataFrame(columns=_trade_silver_columns())
    df["signed_size"] = np.where(df["side"].astype(str).str.lower().eq("buy"), df["size"], -df["size"])
    df["notional"] = df["price"] * df["size"]
    df["signed_notional"] = df["price"] * df["signed_size"]
    rows = []
    for (symbol, decision_ts), g in df.groupby(["symbol", "decision_ts"], dropna=False):
        size_sum = float(g["size"].sum())
        notional_sum = float(g["notional"].sum())
        rows.append(
            {
                "symbol": symbol,
                "decision_ts": decision_ts,
                "trade_observed_available_ts": g["observed_available_ts"].max(),
                "trade_source_interval": "recent_trade_5m_bucket",
                "trade_count": int(len(g)),
                "trade_notional": notional_sum,
                "trade_size": size_sum,
                "signed_trade_size": float(g["signed_size"].sum()),
                "signed_trade_notional": float(g["signed_notional"].sum()),
                "trade_size_imbalance": float(g["signed_size"].sum() / size_sum) if size_sum > 0 else np.nan,
                "trade_notional_imbalance": float(g["signed_notional"].sum() / notional_sum) if notional_sum > 0 else np.nan,
                "trade_reject_reason": "",
            }
        )
    return pd.DataFrame(rows)[_trade_silver_columns()].sort_values(["symbol", "decision_ts"], kind="mergesort").reset_index(drop=True)


def microstructure_coverage_valid(
    silver: pd.DataFrame,
    *,
    symbols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    rows = []
    if silver.empty:
        silver = pd.DataFrame(columns=["symbol", "decision_ts"])
    work = silver.copy()
    work["decision_ts"] = pd.to_datetime(work.get("decision_ts"), utc=True, errors="coerce")
    for sym in sorted({str(s).upper() for s in symbols}):
        expected = pd.date_range(pd.Timestamp(start).ceil("5min"), pd.Timestamp(end).floor("5min"), freq="5min", tz="UTC")
        got = work.loc[work["symbol"].astype(str).str.upper().eq(sym), "decision_ts"].dropna().drop_duplicates()
        ratio = float(len(got) / len(expected)) if len(expected) else np.nan
        rows.append({"symbol": sym, "expected_5m_decisions": int(len(expected)), "observed_5m_rows": int(len(got)), "coverage_ratio": ratio, "coverage_valid": bool(ratio >= min_coverage)})
    return pd.DataFrame(rows)


def _orderbook_silver_columns() -> list[str]:
    return [
        "symbol",
        "decision_ts",
        "orderbook_observed_available_ts",
        "orderbook_source_interval",
        "orderbook_max_age_seconds",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread_pct",
        "top5_bid_depth",
        "top5_ask_depth",
        "top5_depth_imbalance",
        "orderbook_reject_reason",
    ]


def _trade_silver_columns() -> list[str]:
    return [
        "symbol",
        "decision_ts",
        "trade_observed_available_ts",
        "trade_source_interval",
        "trade_count",
        "trade_notional",
        "trade_size",
        "signed_trade_size",
        "signed_trade_notional",
        "trade_size_imbalance",
        "trade_notional_imbalance",
        "trade_reject_reason",
    ]


def _ms_ts(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        if value is None or value == "":
            return pd.NaT
        return pd.to_datetime(int(value), unit="ms", utc=True)
    except Exception:
        return pd.NaT


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")
