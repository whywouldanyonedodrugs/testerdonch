#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.dataset as ds  # type: ignore
except Exception:
    pa = None
    ds = None



REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# CLI
# -----------------------------

def _parse_date(s: str) -> str:
    if s.lower() == "latest":
        return "latest"
    datetime.strptime(s, "%Y-%m-%d")
    return s


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=_parse_date, default="2023-01-01")
    p.add_argument("--end", type=_parse_date, default="latest")
    p.add_argument("--jobs", type=int, default=1)

    p.add_argument("--pstar", type=float, default=0.64)

    p.add_argument("--lambda", dest="lam", type=float, default=2.0)
    p.add_argument("--mu", dest="mu", type=float, default=1.0)

    p.add_argument("--on-missing-symbol", choices=["fail", "skip"], default="skip")

    p.add_argument("--smoke-n", type=int, default=500, help="Run a smoke backtest on first N signals (0 disables).")

    p.add_argument("--run-id", type=str, default="", help="Default: UTC timestamp.")
    return p.parse_args()


def utc_run_id(user_id: str) -> str:
    return user_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# -----------------------------
# Repo config access (in parent process)
# -----------------------------

def import_cfg():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import config as cfg  # type: ignore
    return cfg


def resolve_paths(cfg) -> Tuple[Path, Path, Path, Path]:
    signals_dir = Path(getattr(cfg, "SIGNALS_DIR", REPO_ROOT / "signals")).resolve()
    parquet_dir = Path(getattr(cfg, "PARQUET_DIR", REPO_ROOT / "parquet")).resolve()
    results_dir = Path(getattr(cfg, "RESULTS_DIR", REPO_ROOT / "results")).resolve()
    meta_model_dir = Path(getattr(cfg, "META_MODEL_DIR", results_dir / "meta_export")).resolve()
    return signals_dir, parquet_dir, results_dir, meta_model_dir


def latest_from_signals(signals_dir: Path) -> str:
    # Prefer pyarrow.dataset with hive partitioning (handles signals/symbol=XYZ/*.parquet)
    if ds is not None and signals_dir.is_dir():
        dataset = ds.dataset(str(signals_dir), format="parquet", partitioning="hive")
        if "timestamp" not in dataset.schema.names:
            raise RuntimeError("Signals dataset missing 'timestamp' column.")
        tbl = dataset.to_table(columns=["timestamp"])
        ts = pd.to_datetime(tbl.column("timestamp").to_pandas(), utc=True, errors="coerce").dropna()
        if ts.empty:
            raise RuntimeError("Signals have no parseable timestamps.")
        return ts.max().date().isoformat()

    # Fallback: read the last parquet file found
    files = sorted(signals_dir.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No signals parquet files under {signals_dir}")
    df = pd.read_parquet(files[-1], columns=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        raise RuntimeError("Signals have no parseable timestamps.")
    return ts.max().date().isoformat()


def load_scoped_signals(signals_dir: Path, start: str, end: str) -> pd.DataFrame:
    """
    Load FULL signals rows between [start, end], and guarantee a 'symbol' column exists.

    Supports:
      - hive partitioned layout: signals/symbol=BTCUSDT/*.parquet (files may not contain 'symbol' physically)
      - flat layout: signals/*.parquet with physical 'symbol' column
    """
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    # ---------- Helper: fallback for hive-partitioned directories ----------
    def _load_by_symbol_dirs() -> pd.DataFrame:
        parts: List[pd.DataFrame] = []
        sym_dirs = sorted([p for p in signals_dir.glob("symbol=*") if p.is_dir()])
        if not sym_dirs:
            raise RuntimeError(
                "Signals appear to be missing a physical 'symbol' column, and no hive partition dirs "
                "like signals/symbol=XYZ were found."
            )

        for d in sym_dirs:
            sym = d.name.split("=", 1)[1].strip().upper()
            for f in sorted(d.rglob("*.parquet")):
                df0 = pd.read_parquet(f)  # FULL row
                if "timestamp" not in df0.columns:
                    continue
                df0["timestamp"] = pd.to_datetime(df0["timestamp"], utc=True, errors="coerce")
                df0 = df0.dropna(subset=["timestamp"])
                df0 = df0[(df0["timestamp"] >= start_ts) & (df0["timestamp"] <= end_ts)]
                if df0.empty:
                    continue
                # Ensure symbol exists
                if "symbol" not in df0.columns:
                    df0["symbol"] = sym
                else:
                    df0["symbol"] = df0["symbol"].astype(str).str.upper()
                    # If symbol column exists but is empty, fill it
                    df0.loc[df0["symbol"].isin(["", "nan", "None"]), "symbol"] = sym
                parts.append(df0)

        if not parts:
            # return empty but with timestamp/symbol columns at least
            return pd.DataFrame(columns=["timestamp", "symbol"])

        out = pd.concat(parts, ignore_index=True)
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out = out.dropna(subset=["timestamp", "symbol"]).sort_values("timestamp")
        return out

    # ---------- Preferred: pyarrow.dataset with hive partitioning ----------
    if ds is not None and pa is not None and signals_dir.is_dir():
        dataset = ds.dataset(str(signals_dir), format="parquet", partitioning="hive")
        if "timestamp" not in dataset.schema.names:
            raise RuntimeError(f"Signals missing required 'timestamp'. Found: {dataset.schema.names}")

        # Build filter literals with the SAME Arrow timestamp type as the dataset field.
        # This fixes: ArrowInvalid: Cannot compare timestamp[ns] with timestamp[ns, tz=UTC]
        ts_type = dataset.schema.field("timestamp").type

        def _mk_scalar(val, typ):
            # pyarrow version compatibility:
            # - some versions accept pa.scalar(val, type=typ)
            # - some accept pa.scalar(val, typ)
            # - some require pa.scalar(val).cast(typ)
            try:
                return pa.scalar(val, type=typ)
            except TypeError:
                try:
                    return pa.scalar(val, typ)
                except TypeError:
                    s = pa.scalar(val)
                    try:
                        return s.cast(typ)
                    except Exception:
                        return s


        # Convert pandas Timestamps to python datetimes matching Arrow tz-ness
        if pa.types.is_timestamp(ts_type) and getattr(ts_type, "tz", None):
            # Dataset is tz-aware (e.g., timestamp[ns, tz=UTC])
            start_py = pd.Timestamp(start_ts).to_pydatetime()
            end_py = pd.Timestamp(end_ts).to_pydatetime()
        else:
            # Dataset is tz-naive (e.g., timestamp[ns])
            start_py = pd.Timestamp(start_ts).tz_convert(None).to_pydatetime()
            end_py = pd.Timestamp(end_ts).tz_convert(None).to_pydatetime()

        start_lit = _mk_scalar(start_py, ts_type)
        end_lit = _mk_scalar(end_py, ts_type)


        filt = (ds.field("timestamp") >= start_lit) & (ds.field("timestamp") <= end_lit)

        # Read ALL columns (do not guess which strategy columns are needed)
        try:
            tbl = dataset.to_table(filter=filt)
        except Exception:
            # Safety net: read all then filter in pandas
            tbl = dataset.to_table()

        df = tbl.to_pandas()

        # Safety net: always normalize + re-apply time filter in pandas
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]


        # Guarantee symbol column
        if "symbol" not in df.columns:
            # In some environments, partition columns may not surface; fall back to directory parsing.
            return _load_by_symbol_dirs()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df.dropna(subset=["timestamp", "symbol"]).sort_values("timestamp")
        return df

    # ---------- Fallback: no pyarrow.dataset ----------
    # If signals are hive-partitioned and lack 'symbol' physically, use dir parsing.
    return _load_by_symbol_dirs()


def write_signals_file(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signals.parquet").write_bytes(b"")  # ensure path exists even if parquet writer fails early
    df.to_parquet(out_dir / "signals.parquet", index=False)
    return out_dir


def symbol_has_parquet(parquet_dir: Path, sym: str) -> bool:
    p = parquet_dir / f"{sym}.parquet"
    if p.exists():
        return True
    # allow directory formats
    if (parquet_dir / sym).exists():
        return True
    return False


def preflight_symbols(df_signals: pd.DataFrame, parquet_dir: Path, mode: str) -> pd.DataFrame:
    syms = sorted(df_signals["symbol"].unique().tolist())
    missing = [s for s in syms if not symbol_has_parquet(parquet_dir, s)]
    if missing and mode == "fail":
        raise RuntimeError(f"Missing parquet for {len(missing)} symbols. First 25: {missing[:25]}")
    if missing and mode == "skip":
        return df_signals[~df_signals["symbol"].isin(missing)].copy()
    return df_signals


# -----------------------------
# Grid
# -----------------------------

@dataclass(frozen=True)
class GridPoint:
    name: str
    overrides: Dict[str, Any]


def grid(pstar: float) -> List[GridPoint]:
    meta_modes = [
        ("no_meta_gate", {"META_PROB_THRESHOLD": None, "META_GATE_SCOPE": "all", "META_GATE_FAIL_CLOSED": False}),
        ("gate_with_pstar_all", {"META_PROB_THRESHOLD": pstar, "META_GATE_SCOPE": "all", "META_GATE_FAIL_CLOSED": False}),
        ("gate_with_pstar_scope", {"META_PROB_THRESHOLD": pstar, "META_GATE_SCOPE": "RISK_ON_1", "META_GATE_FAIL_CLOSED": True}),
    ]
    block = [("blockdown_F", {"REGIME_BLOCK_WHEN_DOWN": False}), ("blockdown_T", {"REGIME_BLOCK_WHEN_DOWN": True})]
    probe = [("probe_0", {"RISK_OFF_PROBE_MULT": 0.0}), ("probe_001", {"RISK_OFF_PROBE_MULT": 0.01}), ("probe_005", {"RISK_OFF_PROBE_MULT": 0.05})]
    slope = [("slope_F", {"REGIME_SLOPE_FILTER_ENABLED": False}), ("slope_T", {"REGIME_SLOPE_FILTER_ENABLED": True})]

    out: List[GridPoint] = []
    for (mn, mo), (bn, bo), (pn, po), (sn, so) in [(a, b, c, d) for a in meta_modes for b in block for c in probe for d in slope]:
        name = "__".join([mn, bn, pn, sn])
        ov = {}
        ov.update(mo); ov.update(bo); ov.update(po); ov.update(so)
        out.append(GridPoint(name=name, overrides=ov))
    return out


# -----------------------------
# Child runner: calls repo backtester exactly
# -----------------------------

CHILD_CODE = r"""
import json, os, sys
from pathlib import Path

repo_root = Path(os.environ["DONCH_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg

# Preserve original meta artifact paths (do not let RESULTS_DIR override them)
orig_meta_model_dir = None
orig_meta_pred_path = None

try:
    if hasattr(cfg, "META_MODEL_DIR"):
        orig_meta_model_dir = Path(getattr(cfg, "META_MODEL_DIR")).resolve()
except Exception:
    orig_meta_model_dir = None

try:
    if hasattr(cfg, "META_PRED_PATH"):
        orig_meta_pred_path = Path(getattr(cfg, "META_PRED_PATH")).resolve()
except Exception:
    orig_meta_pred_path = None

out_dir = Path(os.environ["DONCH_OUT_DIR"]).resolve()
signals_dir = Path(os.environ["DONCH_SIGNALS_DIR"]).resolve()
start = os.environ["DONCH_START"]
end = os.environ["DONCH_END"]
overrides = json.loads(os.environ["DONCH_OVERRIDES_JSON"])

# IMPORTANT: set these as Path objects (backtester uses .glob() and Path / "file" ops)
setattr(cfg, "RESULTS_DIR", out_dir)
setattr(cfg, "SIGNALS_DIR", signals_dir)

# Also set common alternative names if they exist (safe no-ops otherwise)
for k in ("SIGNALS_PATH",):
    if hasattr(cfg, k):
        try:
            setattr(cfg, k, signals_dir)
        except Exception:
            pass

setattr(cfg, "START_DATE", str(start))
setattr(cfg, "END_DATE", str(end))

# Re-pin meta artifact paths (keep reading the existing bundle/artifacts)
if orig_meta_model_dir is not None:
    setattr(cfg, "META_MODEL_DIR", orig_meta_model_dir)
if orig_meta_pred_path is not None:
    setattr(cfg, "META_PRED_PATH", orig_meta_pred_path)

# Apply overrides
for k, v in overrides.items():
    setattr(cfg, k, v)

import backtester
backtester.run_backtest()
"""



def run_backtest_subprocess(
    run_dir: Path,
    signals_dir: Path,
    start: str,
    end: str,
    overrides: Dict[str, Any],
) -> Tuple[int, str]:
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DONCH_REPO_ROOT"] = str(REPO_ROOT)
    env["DONCH_OUT_DIR"] = str(run_dir.resolve())
    env["DONCH_SIGNALS_DIR"] = str(signals_dir.resolve())
    env["DONCH_START"] = start
    env["DONCH_END"] = end
    env["DONCH_OVERRIDES_JSON"] = json.dumps(overrides, sort_keys=True)

    # Force unbuffered output so logs update continuously.
    env["PYTHONUNBUFFERED"] = "1"

    log_path = run_dir / "logs.txt"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        p = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD_CODE],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = p.wait()

    return int(rc), "logs.txt"



# -----------------------------
# Metrics (robust file finding)
# -----------------------------

def find_trades_file(run_dir: Path) -> Path:
    pats = ["trades.clean.csv", "trades.enriched.filled.csv", "trades.enriched.csv", "trades.csv", "*.trades.csv"]
    for pat in pats:
        hits = sorted(run_dir.rglob(pat))
        if hits:
            return hits[0]
    raise RuntimeError(f"No trades file found under {run_dir}")


def compute_metrics(run_dir: Path, lam: float, mu: float) -> Dict[str, Any]:
    trades_path = find_trades_file(run_dir)
    df = pd.read_csv(trades_path, low_memory=False)

    pnl_col = None
    for c in ("pnl_cash", "pnl", "pnl_usd", "pnl_net", "pnl_after_fees"):
        if c in df.columns:
            pnl_col = c
            break
    if pnl_col is None:
        raise RuntimeError(f"Trades file has no PnL column. Columns: {list(df.columns)[:50]}")
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0).to_numpy()

    # equity curve (event-based)
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = np.where(peak != 0, equity / peak - 1.0, 0.0)
    max_dd = float(np.min(dd)) if dd.size else 0.0

    n = int(len(pnl))
    win_rate = float(np.mean(pnl > 0)) if n else float("nan")
    tail_5 = float(np.quantile(pnl, 0.05)) if n else float("nan")

    # risk split if present (preferred)
    if "risk_on" in df.columns:
        r = pd.to_numeric(df["risk_on"], errors="coerce").fillna(np.nan).to_numpy()
    elif "risk_on_1" in df.columns:
        r = pd.to_numeric(df["risk_on_1"], errors="coerce").fillna(np.nan).to_numpy()
    else:
        r = np.full(n, np.nan)

    pnl_on = float(np.sum(pnl[(r == 1)])) if n else float("nan")
    pnl_off = float(np.sum(pnl[(r == 0)])) if n else float("nan")
    bad_damage = float(np.sum(pnl[(r == 0) & (pnl < 0)])) if n else float("nan")

    util = float(pnl_on - lam * abs(min(pnl_off, 0.0)) - mu * abs(min(max_dd, 0.0))) if np.isfinite(pnl_on) and np.isfinite(pnl_off) else float("nan")

    return {
        "trades_file": str(trades_path),
        "pnl_col": pnl_col,
        "total_pnl": float(np.sum(pnl)),
        "max_drawdown": max_dd,
        "number_of_trades": n,
        "win_rate": win_rate,
        "tail_loss_p05": tail_5,
        "pnl_risk_on": pnl_on,
        "pnl_risk_off": pnl_off,
        "bad_regime_damage": bad_damage,
        "utility": util,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    a = args()
    cfg = import_cfg()
    signals_dir, parquet_dir, results_dir, _meta_model_dir = resolve_paths(cfg)

    rid = utc_run_id(a.run_id)
    sweep_root = (results_dir / "policy_sweeps" / rid).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    end = a.end
    if end.lower() == "latest":
        end = latest_from_signals(signals_dir)

    # Build scoped signals once
    sig = load_scoped_signals(signals_dir, a.start, end)
    sig = preflight_symbols(sig, parquet_dir, a.on_missing_symbol)

    scoped_dir = write_signals_file(sig, sweep_root / "_scoped_signals")
    (sweep_root / "scoped_info.json").write_text(
        json.dumps({"start": a.start, "end": end, "rows": int(len(sig)), "symbols": int(sig["symbol"].nunique())}, indent=2),
        encoding="utf-8",
    )

    # Smoke test (fast)
    if a.smoke_n and a.smoke_n > 0:
        smoke_df = sig.head(int(a.smoke_n)).copy()
        smoke_dir = write_signals_file(smoke_df, sweep_root / "_smoke_signals")
        smoke_run = sweep_root / "_smoke_run"
        smoke_run.mkdir(parents=True, exist_ok=True)


        (sweep_root / "_STAGE.txt").write_text("SMOKE_RUNNING\n", encoding="utf-8")



        rc, _ = run_backtest_subprocess(
            run_dir=smoke_run,
            signals_dir=smoke_dir,
            start=a.start,
            end=end,
            overrides={
                "BT_META_ONLINE_ENABLED": True,
                "META_STRICT_SCHEMA": True,
            },
        )
        if rc != 0:
            raise RuntimeError(f"Smoke test failed (rc={rc}). See {smoke_run / 'logs.txt'}")

        (sweep_root / "_STAGE.txt").write_text("SMOKE_DONE\n", encoding="utf-8")


    # Grid
    gps = grid(a.pstar)

    def run_one(gp: GridPoint) -> Dict[str, Any]:
        run_dir = sweep_root / gp.name
        done = run_dir / "_DONE.json"
        if done.exists():
            try:
                return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            except Exception:
                pass

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.json").write_text(
            json.dumps({"start": a.start, "end": end, "signals": str(scoped_dir), "overrides": gp.overrides}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        (sweep_root / "_STAGE.txt").write_text(f"RUNNING {gp.name}\n", encoding="utf-8")


        rc, _log = run_backtest_subprocess(
            run_dir=run_dir,
            signals_dir=scoped_dir,
            start=a.start,
            end=end,
            overrides=gp.overrides,
        )
        if rc != 0:
            (run_dir / "metrics.json").write_text(json.dumps({"status": "error", "returncode": rc}, indent=2), encoding="utf-8")
            (run_dir / "_DONE.json").write_text(json.dumps({"returncode": rc}, indent=2), encoding="utf-8")
            return {"setting": gp.name, "status": "error", "returncode": rc}

        met = compute_metrics(run_dir, a.lam, a.mu)
        met["setting"] = gp.name
        met["status"] = "ok"
        (run_dir / "metrics.json").write_text(json.dumps(met, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "_DONE.json").write_text(json.dumps({"returncode": 0}, indent=2), encoding="utf-8")
        return met

    results: List[Dict[str, Any]] = []
    jobs = max(int(a.jobs), 1)

    if jobs == 1:
        for gp in gps:
            results.append(run_one(gp))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(run_one, gp) for gp in gps]
            for fut in as_completed(futs):
                results.append(fut.result())

    df = pd.DataFrame(results)
    df.to_csv(sweep_root / "summary.csv", index=False)

    df_ok = df[df.get("status") == "ok"].copy()
    df_ok["utility"] = pd.to_numeric(df_ok["utility"], errors="coerce")
    df_ok = df_ok.sort_values("utility", ascending=False)

    top = df_ok.head(3)
    lines = []
    lines.append(f"# Policy sweep report: {rid}")
    lines.append("")
    lines.append(f"Utility = PnL_risk_on - {a.lam:g}*abs(min(PnL_risk_off,0)) - {a.mu:g}*max_drawdown")
    lines.append("")
    lines.append("## Top 3")
    for _, r in top.iterrows():
        lines.append(f"- {r['setting']}: utility={r.get('utility')}, pnl_on={r.get('pnl_risk_on')}, bad_damage={r.get('bad_regime_damage')}, max_dd={r.get('max_drawdown')}")
    lines.append("")
    lines.append("## Ranked (top 15)")
    lines.append("")
    show = df_ok[["setting","utility","pnl_risk_on","pnl_risk_off","bad_regime_damage","max_drawdown","number_of_trades","win_rate"]].head(15)
    lines.append(show.to_markdown(index=False))
    (sweep_root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(str(sweep_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
