#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from telegram_notify import TelegramNotifier


REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_WARMUP_MISSING_FIELDS = frozenset({"days_since_prev_break", "S6_fresh_x_compress"})


SCOUT_CODE = r"""
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["DONCH_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg
from scout import run_scout

cfg.PARQUET_DIR = Path(os.environ["DONCH_PARQUET_DIR"]).resolve()
cfg.SIGNALS_DIR = Path(os.environ["DONCH_SIGNALS_DIR"]).resolve()
cfg.RESULTS_DIR = Path(os.environ["DONCH_AUX_RESULTS_DIR"]).resolve()
cfg.START_DATE = str(os.environ["DONCH_START"])
cfg.END_DATE = str(os.environ["DONCH_END"])
cfg.N_WORKERS = int(os.environ.get("DONCH_SCOUT_WORKERS", "2"))
cfg.SCOUT_BACKEND = str(os.environ.get("DONCH_SCOUT_BACKEND", "thread"))
cfg.SCOUT_CLEAN_OUTPUT_DIR = bool(int(os.environ.get("DONCH_SCOUT_CLEAN", "1")))

sym_file = os.environ.get("DONCH_SYMBOLS_FILE", "").strip()
if sym_file:
    cfg.SYMBOLS_FILE = Path(sym_file).resolve()

cfg.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

n = run_scout()
print(f"[autopar] scout_done rows={int(n)} signals_dir={cfg.SIGNALS_DIR}", flush=True)
"""


BACKTEST_CODE = r"""
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["DONCH_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg
import backtester

cfg.PARQUET_DIR = Path(os.environ["DONCH_PARQUET_DIR"]).resolve()
cfg.SIGNALS_DIR = Path(os.environ["DONCH_SIGNALS_DIR"]).resolve()
cfg.RESULTS_DIR = Path(os.environ["DONCH_BT_OUT_DIR"]).resolve()
cfg.START_DATE = str(os.environ["DONCH_START"])
cfg.END_DATE = str(os.environ["DONCH_END"])
cfg.META_MODEL_DIR = Path(os.environ["DONCH_META_MODEL_DIR"]).resolve()

parq1m = os.environ.get("DONCH_PARQUET_1M_DIR", "").strip()
if parq1m:
    cfg.PARQUET_1M_DIR = Path(parq1m).resolve()

for k, v in json.loads(os.environ["DONCH_OVERRIDES_JSON"]).items():
    setattr(cfg, k, v)

cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
backtester.run_backtest(signals_path=cfg.SIGNALS_DIR)
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily live-vs-backtest parity runner (donch_autopar phase-1)."
    )
    p.add_argument("--run-id", default="", help="Default: UTC timestamp.")
    p.add_argument("--results-root", default="results/donch_autopar")
    p.add_argument("--live-input", required=True, help="Live package dir or zip.")
    p.add_argument("--live-decisions", default="", help="Override decisions CSV path.")
    p.add_argument("--live-trades", default="", help="Override trades CSV path.")
    p.add_argument("--symbols-file", default="", help="Override symbols file path.")

    p.add_argument("--start", default="", help="UTC start override.")
    p.add_argument("--end", default="", help="UTC end override.")
    p.add_argument("--window-days", type=int, default=3, help="Fallback window if start/end are not inferable.")

    p.add_argument("--parquet-dir", default="/opt/fader2/parquet")
    p.add_argument("--parquet-1m-dir", default="")
    p.add_argument("--model-dir", default="results/offline_releases/20260209_meta_release_v2_live_safe/meta_export_pstar_042")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-scout", action="store_true")
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--scout-workers", type=int, default=2)
    p.add_argument("--scout-backend", choices=["thread", "process"], default="thread")
    p.add_argument("--bt-overrides-json", default="", help="Extra cfg overrides JSON for backtest stage.")

    p.add_argument("--min-overlap-rate", type=float, default=0.50)
    p.add_argument("--min-enter-agreement", type=float, default=0.85)
    p.add_argument("--sla-stale-max-hours", type=float, default=6.0, help="SLA: max allowed staleness gap from expected window end.")
    p.add_argument("--sla-min-ohlcv-coverage", type=float, default=0.90, help="SLA: minimum required OHLCV completeness in window.")
    p.add_argument("--sla-min-oi-coverage", type=float, default=0.50, help="SLA: minimum required OI completeness in window.")
    p.add_argument("--sla-min-funding-coverage", type=float, default=0.50, help="SLA: minimum required funding completeness in window.")
    p.add_argument("--sla-top-n", type=int, default=8, help="Top-N symbols to surface in SLA diagnostics.")
    p.add_argument("--sla-max-symbols", type=int, default=0, help="Optional debug cap for SLA symbol checks (0=all).")

    p.add_argument("--tg-bot-token", default="")
    p.add_argument("--tg-chat-id", default="")
    p.add_argument(
        "--tg-auto-chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try Telegram getUpdates to auto-discover chat id when not provided.",
    )
    return p.parse_args()


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_abs(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p.resolve()


def _run_cmd(cmd: List[str], *, cwd: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[autopar] running: {' '.join(cmd)}", flush=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    run_env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=run_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(p.wait())


def _extract_live_package(live_input: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if live_input.is_file() and live_input.suffix.lower() == ".zip":
        with zipfile.ZipFile(live_input, "r") as zf:
            zf.extractall(dst_dir)
        # If zip has a top-level folder, use it.
        kids = [p for p in dst_dir.iterdir() if p.is_dir()]
        return kids[0] if len(kids) == 1 else dst_dir
    if live_input.is_dir():
        return live_input
    raise FileNotFoundError(f"Unsupported live input: {live_input}")


def _find_first(base: Path, candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = base / c
        if p.exists() and p.is_file():
            return p
    return None


def _infer_window(
    decisions_path: Optional[Path],
    start_arg: str,
    end_arg: str,
    window_days: int,
) -> Tuple[str, str]:
    start_arg = (start_arg or "").strip()
    end_arg = (end_arg or "").strip()
    if start_arg and end_arg:
        return start_arg, end_arg

    if decisions_path and decisions_path.exists():
        try:
            d = pd.read_csv(decisions_path, low_memory=False)
            ts_col = None
            for c in ("decision_ts", "ts_effective", "signal_ts", "timestamp"):
                if c in d.columns:
                    ts_col = c
                    break
            if ts_col is not None:
                ts = pd.to_datetime(d[ts_col], utc=True, errors="coerce").dropna()
                if not ts.empty:
                    s = ts.min().floor("D")
                    e = ts.max().ceil("D")
                    return str(s.date()), str(e.date())
        except Exception:
            pass

    today = datetime.now(timezone.utc).date()
    end_d = today - timedelta(days=1)
    start_d = end_d - timedelta(days=max(1, int(window_days)) - 1)
    return str(start_d), str(end_d)


def _infer_window_from_context(package_dir: Path) -> Optional[Tuple[str, str]]:
    rc = package_dir / "run_context.json"
    if not rc.exists():
        return None
    try:
        obj = json.loads(rc.read_text(encoding="utf-8"))
        s = str(obj.get("window_start_utc", "")).strip()
        e = str(obj.get("window_end_utc", "")).strip()
        if not s or not e:
            return None
        s_ts = pd.to_datetime([s], utc=True, errors="coerce")
        e_ts = pd.to_datetime([e], utc=True, errors="coerce")
        if s_ts.isna().any() or e_ts.isna().any():
            return None
        s_d = s_ts[0].floor("D")
        e_d = e_ts[0].ceil("D")
        return str(s_d.date()), str(e_d.date())
    except Exception:
        return None


def _default_bt_overrides(model_dir: Path) -> Dict[str, object]:
    ov: Dict[str, object] = {
        "BT_DECISION_LOG_ENABLED": True,
        "BT_PROGRESS_ENABLED": False,
        "BT_META_REPLAY_ENABLED": False,
        "BT_META_ONLINE_ENABLED": True,
        "META_STRICT_SCHEMA": True,
    }
    dep = model_dir / "deployment_config.json"
    if dep.exists():
        try:
            obj = json.loads(dep.read_text(encoding="utf-8"))
            thr = (((obj.get("decision") or {}).get("threshold")))
            scope = (((obj.get("decision") or {}).get("scope")))
            if thr is not None:
                ov["META_PROB_THRESHOLD"] = float(thr)
            if scope is not None:
                ov["META_GATE_SCOPE"] = str(scope)
                ov["META_GATE_FAIL_CLOSED"] = False
        except Exception:
            pass
    return ov


def _to_bool_opt(x: object) -> Optional[bool]:
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def _safe_float(x: object, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _parse_missing_fields_from_err(err: object) -> List[str]:
    s = str(err).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return []

    fields: List[str] = []
    low = s.lower()

    if "missing_required" in low:
        rest = s.split("missing_required", 1)[1].lstrip(":=").strip()
        if rest.startswith("[") or rest.startswith("("):
            try:
                parsed = ast.literal_eval(rest)
                if isinstance(parsed, (list, tuple, set)):
                    for v in parsed:
                        txt = str(v).strip()
                        if txt:
                            fields.append(txt)
            except Exception:
                pass
        if not fields:
            fields.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rest))

    if (not fields) and ("feature_missing" in low):
        m = re.search(r"feature_missing[:=]([A-Za-z_][A-Za-z0-9_]*)", s, flags=re.IGNORECASE)
        if m:
            fields.append(m.group(1))

    if not fields:
        fields.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s))

    deny = {
        "missing_required",
        "feature_missing",
        "missing",
        "required",
        "none",
        "null",
        "nan",
        "true",
        "false",
    }
    out: List[str] = []
    seen = set()
    for f in fields:
        k = str(f).strip()
        if not k:
            continue
        if k.lower() in deny:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _collect_schema_fails_from_live_log(package_dir: Path) -> Tuple[Counter, Counter, int]:
    field_counter: Counter = Counter()
    symbol_counter: Counter = Counter()
    rows = 0

    candidates: List[Path] = []
    p0 = package_dir / "live.log"
    if p0.exists():
        candidates.append(p0)
    p_logs = package_dir / "logs"
    if p_logs.exists():
        for p in sorted(p_logs.glob("*.log*")):
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return field_counter, symbol_counter, rows

    seen = set()
    for fp in candidates:
        marker = "META_DECISION "
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if marker not in line:
                        continue
                    if "reason=schema_fail" not in line:
                        continue
                    rows += 1
                    m_sym = re.search(r"\bsymbol=([A-Za-z0-9_.:\-]+)", line)
                    sym = m_sym.group(1).upper() if m_sym else ""
                    m_ts = re.search(r"\bdecision_ts=([0-9T:\-+]+)", line)
                    ts = m_ts.group(1) if m_ts else ""
                    if sym:
                        key = f"{sym}|{ts}" if ts else f"{sym}|{rows}"
                        if key not in seen:
                            symbol_counter[sym] += 1
                            seen.add(key)
                    m_err = re.search(r"\berr=(.*)$", line.strip())
                    err_txt = m_err.group(1).strip() if m_err else ""
                    for fld in _parse_missing_fields_from_err(err_txt):
                        field_counter[fld] += 1
        except Exception:
            continue
    return field_counter, symbol_counter, rows


def _top_counter_rows(counter: Counter, key_name: str, top_n: int = 8) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    n = max(1, int(top_n))
    for k, v in counter.most_common(n):
        out.append({key_name: str(k), "count": int(v)})
    return out


def _build_live_schema_diag(
    live_decisions_path: Path,
    package_dir: Path,
    top_n: int = 8,
) -> Dict[str, object]:
    diag: Dict[str, object] = {
        "rows_total": 0,
        "schema_fail_rows": 0,
        "schema_fail_rate": 0.0,
        "schema_ok_rows": 0,
        "top_schema_fail_symbols": [],
        "top_missing_fields": [],
        "missing_fields_source": "none",
        "missing_field_mentions": 0,
        "warmup_expected_fields": sorted(_EXPECTED_WARMUP_MISSING_FIELDS),
        "warmup_field_mentions": 0,
        "non_warmup_field_mentions": 0,
        "expected_warmup_only": False,
        "alert_level": "ok",
    }
    if not live_decisions_path.exists():
        diag["alert_level"] = "unknown"
        diag["notes"] = f"missing_live_decisions:{live_decisions_path}"
        return diag

    try:
        d = pd.read_csv(live_decisions_path, low_memory=False)
    except Exception as e:
        diag["alert_level"] = "unknown"
        diag["notes"] = f"read_error:{type(e).__name__}:{e}"
        return diag

    diag["rows_total"] = int(len(d))
    if d.empty:
        return diag

    col_l = {c.lower(): c for c in d.columns}
    sym_col = col_l.get("symbol")
    reason_col = col_l.get("reason")
    schema_ok_col = col_l.get("schema_ok")
    err_col = col_l.get("err")

    fail_mask = pd.Series([False] * len(d), index=d.index)
    if reason_col is not None:
        fail_mask = fail_mask | (d[reason_col].astype(str).str.strip().str.lower() == "schema_fail")
    if schema_ok_col is not None:
        b = d[schema_ok_col].map(_to_bool_opt)
        fail_mask = fail_mask | (b == False)  # noqa: E712

    schema_fail_rows = int(fail_mask.sum())
    rows_total = int(len(d))
    schema_ok_rows = int(max(0, rows_total - schema_fail_rows))
    schema_fail_rate = float(schema_fail_rows / rows_total) if rows_total > 0 else 0.0

    diag["schema_fail_rows"] = schema_fail_rows
    diag["schema_fail_rate"] = schema_fail_rate
    diag["schema_ok_rows"] = schema_ok_rows

    symbol_counter: Counter = Counter()
    if sym_col is not None and schema_fail_rows > 0:
        for s in d.loc[fail_mask, sym_col].astype(str):
            sym = s.upper().strip()
            if sym:
                symbol_counter[sym] += 1

    fields_counter: Counter = Counter()
    if err_col is not None and schema_fail_rows > 0:
        for e in d.loc[fail_mask, err_col]:
            for fld in _parse_missing_fields_from_err(e):
                fields_counter[fld] += 1
        if fields_counter:
            diag["missing_fields_source"] = "live_decisions.err"

    if not fields_counter:
        log_fields, log_symbols, _ = _collect_schema_fails_from_live_log(package_dir)
        if log_fields:
            fields_counter.update(log_fields)
            diag["missing_fields_source"] = "live.log"
        if not symbol_counter and log_symbols:
            symbol_counter.update(log_symbols)

    diag["top_schema_fail_symbols"] = _top_counter_rows(symbol_counter, "symbol", top_n=top_n)
    diag["top_missing_fields"] = _top_counter_rows(fields_counter, "field", top_n=top_n)

    total_mentions = int(sum(fields_counter.values()))
    warmup_mentions = int(
        sum(v for k, v in fields_counter.items() if str(k) in _EXPECTED_WARMUP_MISSING_FIELDS)
    )
    non_warmup_mentions = int(max(0, total_mentions - warmup_mentions))

    diag["missing_field_mentions"] = total_mentions
    diag["warmup_field_mentions"] = warmup_mentions
    diag["non_warmup_field_mentions"] = non_warmup_mentions
    diag["expected_warmup_only"] = bool(schema_fail_rows > 0 and total_mentions > 0 and non_warmup_mentions == 0)

    if schema_fail_rows <= 0:
        diag["alert_level"] = "ok"
    elif diag["expected_warmup_only"]:
        diag["alert_level"] = "info"
    else:
        diag["alert_level"] = "warn"
    return diag


def _fmt_top_counts(rows: object, key: str, n: int = 3) -> str:
    if not isinstance(rows, list):
        return "none"
    out: List[str] = []
    for row in rows[: max(1, int(n))]:
        if not isinstance(row, dict):
            continue
        k = str(row.get(key, "")).strip()
        c = _safe_int(row.get("count"), 0)
        if k:
            out.append(f"{k}:{c}")
    return ", ".join(out) if out else "none"


def _schema_alert_lines(diag: object) -> List[str]:
    if not isinstance(diag, dict):
        return []
    total = _safe_int(diag.get("rows_total"), 0)
    sf = _safe_int(diag.get("schema_fail_rows"), 0)
    rate = _safe_float(diag.get("schema_fail_rate"), float("nan"))
    lvl = str(diag.get("alert_level", "")).strip() or "unknown"
    if total <= 0:
        return [f"schema_fail=n/a level={lvl}"]
    if rate == rate:
        rate_txt = f"{100.0 * rate:.2f}%"
    else:
        rate_txt = "n/a"
    lines = [f"schema_fail={sf}/{total} ({rate_txt}) level={lvl}"]
    top_fields = _fmt_top_counts(diag.get("top_missing_fields"), "field", n=3)
    top_symbols = _fmt_top_counts(diag.get("top_schema_fail_symbols"), "symbol", n=3)
    lines.append(f"schema_missing_fields_top={top_fields}")
    lines.append(f"schema_fail_symbols_top={top_symbols}")
    if bool(diag.get("expected_warmup_only")):
        lines.append("schema_fail_mode=expected_warmup_only")
    return lines


def _read_symbols_file(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    p = Path(path)
    if (not p.exists()) or (not p.is_file()):
        return []
    out: List[str] = []
    try:
        for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = raw.strip().upper()
            if not s:
                continue
            if s.startswith("#"):
                continue
            out.append(s)
    except Exception:
        return []
    # de-dup preserving order
    seen = set()
    ded: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        ded.append(s)
    return ded


def _read_symbols_from_decisions(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        d = pd.read_csv(path, usecols=["symbol"], low_memory=False)
    except Exception:
        try:
            d = pd.read_csv(path, low_memory=False)
        except Exception:
            return []
        if "symbol" not in d.columns:
            return []
    vals = d["symbol"].astype(str).str.upper().str.strip()
    vals = vals[vals != ""]
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for s in vals.tolist():
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _collect_sla_symbols(
    symbols_file: Optional[Path],
    live_decisions_path: Path,
    max_symbols: int = 0,
) -> List[str]:
    sym_file = _read_symbols_file(symbols_file)
    sym_live = _read_symbols_from_decisions(live_decisions_path)
    seen = set(sym_file)
    merged = list(sym_file)
    for s in sym_live:
        if s in seen:
            continue
        seen.add(s)
        merged.append(s)
    if max_symbols and max_symbols > 0:
        merged = merged[: int(max_symbols)]
    return merged


def _window_bounds_from_args(start_s: str, end_s: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    s = pd.to_datetime(str(start_s), utc=True, errors="coerce")
    e = pd.to_datetime(str(end_s), utc=True, errors="coerce")
    if pd.isna(s) or pd.isna(e):
        raise ValueError(f"Invalid SLA window bounds: start={start_s} end={end_s}")
    s = pd.Timestamp(s).floor("D")
    e = pd.Timestamp(e)
    # If date-only / midnight boundary, treat end as full day.
    if e.hour == 0 and e.minute == 0 and e.second == 0 and e.microsecond == 0:
        e = e + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    else:
        e = e.floor("5min")
    if e < s:
        e = s
    return s, e


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _float_quantile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = pd.Series(vals, dtype="float64")
    s = s[s.notna()]
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def _append_data_sla_to_report(report_html: Path, diag: Dict[str, object]) -> None:
    if not report_html.exists():
        return
    try:
        txt = report_html.read_text(encoding="utf-8")
    except Exception:
        return
    if "Data Freshness & Completeness (SLA)" in txt:
        return

    issues = diag.get("issues", {}) if isinstance(diag, dict) else {}
    cov = diag.get("coverage", {}) if isinstance(diag, dict) else {}
    top = diag.get("top_issue_symbols", []) if isinstance(diag, dict) else []
    top_rows = "".join(
        [
            f"<tr><td>{r.get('symbol')}</td><td>{r.get('issues')}</td><td>{r.get('latest_ts')}</td><td>{r.get('staleness_hours')}</td><td>{r.get('ohlcv_cov')}</td><td>{r.get('oi_cov')}</td><td>{r.get('funding_cov')}</td></tr>"
            for r in top
            if isinstance(r, dict)
        ]
    ) or "<tr><td colspan='7'>none</td></tr>"

    section = f"""
<h2>Data Freshness &amp; Completeness (SLA)</h2>
<table>
<tr><th>status</th><th>symbols_total</th><th>symbols_checked</th><th>window_start</th><th>window_end</th></tr>
<tr><td>{diag.get('status')}</td><td>{diag.get('symbols_total')}</td><td>{diag.get('symbols_checked')}</td><td>{diag.get('window_start')}</td><td>{diag.get('window_end')}</td></tr>
</table>
<table>
<tr><th>missing_file</th><th>stale</th><th>low_ohlcv</th><th>low_oi</th><th>low_funding</th><th>incident_recommended</th></tr>
<tr><td>{issues.get('missing_file')}</td><td>{issues.get('stale')}</td><td>{issues.get('low_ohlcv_coverage')}</td><td>{issues.get('low_oi_coverage')}</td><td>{issues.get('low_funding_coverage')}</td><td>{diag.get('incident_recommended')}</td></tr>
</table>
<table>
<tr><th>ohlcv_cov_median</th><th>oi_cov_median</th><th>funding_cov_median</th><th>staleness_p90_h</th></tr>
<tr><td>{cov.get('ohlcv_cov_median')}</td><td>{cov.get('oi_cov_median')}</td><td>{cov.get('funding_cov_median')}</td><td>{cov.get('staleness_hours_p90')}</td></tr>
</table>
<h3>Top Issue Symbols</h3>
<table>
<tr><th>symbol</th><th>issues</th><th>latest_ts</th><th>staleness_hours</th><th>ohlcv_cov</th><th>oi_cov</th><th>funding_cov</th></tr>
{top_rows}
</table>
"""

    if "</body>" in txt:
        txt = txt.replace("</body>", section + "\n</body>")
    else:
        txt = txt + "\n" + section
    try:
        report_html.write_text(txt, encoding="utf-8")
    except Exception:
        return


def _data_sla_alert_lines(diag: object) -> List[str]:
    if not isinstance(diag, dict):
        return []
    issues = diag.get("issues", {}) if isinstance(diag.get("issues"), dict) else {}
    status = str(diag.get("status", "unknown"))
    n = _safe_int(diag.get("symbols_checked"), 0)
    if n <= 0:
        return [f"data_sla=status:{status} symbols=0"]
    lines = [
        f"data_sla=status:{status} symbols={n}",
        "data_sla_issues="
        + ",".join(
            [
                f"missing_file:{_safe_int(issues.get('missing_file'),0)}",
                f"stale:{_safe_int(issues.get('stale'),0)}",
                f"low_ohlcv:{_safe_int(issues.get('low_ohlcv_coverage'),0)}",
                f"low_oi:{_safe_int(issues.get('low_oi_coverage'),0)}",
                f"low_funding:{_safe_int(issues.get('low_funding_coverage'),0)}",
            ]
        ),
    ]
    top = diag.get("top_issue_symbols", [])
    if isinstance(top, list) and top:
        parts: List[str] = []
        for r in top[:3]:
            if not isinstance(r, dict):
                continue
            sym = str(r.get("symbol", "")).strip()
            iss = str(r.get("issues", "")).strip()
            if sym:
                parts.append(f"{sym}:{iss}")
        if parts:
            lines.append("data_sla_top=" + "; ".join(parts))
    if bool(diag.get("incident_recommended")):
        lines.append("data_sla_incident_recommended=true")
    return lines


def _build_data_sla_diag(
    parquet_dir: Path,
    symbols_file: Optional[Path],
    live_decisions_path: Path,
    start_s: str,
    end_s: str,
    *,
    stale_max_hours: float = 6.0,
    min_ohlcv_coverage: float = 0.90,
    min_oi_coverage: float = 0.50,
    min_funding_coverage: float = 0.50,
    top_n: int = 8,
    max_symbols: int = 0,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        diag = {
            "status": "unknown",
            "notes": f"pyarrow_unavailable:{type(e).__name__}:{e}",
            "symbols_total": 0,
            "symbols_checked": 0,
            "issues": {},
            "coverage": {},
            "top_issue_symbols": [],
            "incident_recommended": False,
        }
        return diag, pd.DataFrame()

    syms = _collect_sla_symbols(symbols_file, live_decisions_path, max_symbols=max_symbols)
    window_start, window_end = _window_bounds_from_args(start_s, end_s)
    exp_rows = int(((window_end - window_start) / pd.Timedelta(minutes=5)) + 1)
    exp_rows = max(exp_rows, 1)
    stale_gap = pd.Timedelta(hours=max(0.0, float(stale_max_hours)))

    rows: List[Dict[str, object]] = []
    for sym in syms:
        p = parquet_dir / f"{sym}.parquet"
        row: Dict[str, object] = {
            "symbol": sym,
            "file_exists": bool(p.exists()),
            "latest_ts": "",
            "staleness_hours": float("nan"),
            "rows_window": 0,
            "expected_rows": exp_rows,
            "ohlcv_cov": 0.0,
            "oi_cov": 0.0,
            "funding_cov": 0.0,
            "has_open_interest_col": False,
            "has_funding_rate_col": False,
            "issues": "",
        }
        issues: List[str] = []

        if not p.exists():
            issues.append("missing_file")
            row["issues"] = "|".join(issues)
            rows.append(row)
            continue

        try:
            pf = pq.ParquetFile(str(p))
            schema_cols = list(pf.schema.names)
        except Exception:
            issues.append("read_error")
            row["issues"] = "|".join(issues)
            rows.append(row)
            continue

        ts_col = None
        for c in ("timestamp", "ts", "datetime", "date", "__index_level_0__"):
            if c in schema_cols:
                ts_col = c
                break
        if ts_col is None:
            issues.append("missing_timestamp_col")
            row["issues"] = "|".join(issues)
            rows.append(row)
            continue

        try:
            last_rg = max(0, int(pf.num_row_groups) - 1)
            t_last = pf.read_row_group(last_rg, columns=[ts_col]).to_pandas()
            if ts_col in t_last.columns:
                latest_ts = pd.to_datetime(t_last[ts_col], utc=True, errors="coerce").max()
            else:
                latest_ts = pd.NaT
        except Exception:
            latest_ts = pd.NaT

        if pd.isna(latest_ts):
            issues.append("stale")
            row["latest_ts"] = ""
            row["staleness_hours"] = float("inf")
        else:
            latest_ts = pd.Timestamp(latest_ts).tz_convert("UTC")
            row["latest_ts"] = latest_ts.isoformat()
            st_hours = float((window_end - latest_ts) / pd.Timedelta(hours=1))
            row["staleness_hours"] = st_hours
            if latest_ts < (window_end - stale_gap):
                issues.append("stale")

        use_cols = [ts_col]
        req_ohlcv = ["open", "high", "low", "close", "volume"]
        use_cols.extend([c for c in req_ohlcv if c in schema_cols])
        has_oi = "open_interest" in schema_cols
        has_fr = "funding_rate" in schema_cols
        row["has_open_interest_col"] = bool(has_oi)
        row["has_funding_rate_col"] = bool(has_fr)
        if has_oi:
            use_cols.append("open_interest")
        if has_fr:
            use_cols.append("funding_rate")

        try:
            tab = pq.read_table(
                str(p),
                columns=use_cols,
                filters=[(ts_col, ">=", window_start.to_pydatetime()), (ts_col, "<=", window_end.to_pydatetime())],
            )
            d = tab.to_pandas()
        except Exception:
            d = pd.DataFrame(columns=use_cols)

        if d.empty:
            row["rows_window"] = 0
            row["ohlcv_cov"] = 0.0
            row["oi_cov"] = 0.0
            row["funding_cov"] = 0.0
            issues.append("low_ohlcv_coverage")
            issues.append("low_oi_coverage")
            issues.append("low_funding_coverage")
            row["issues"] = "|".join(sorted(set(issues)))
            rows.append(row)
            continue

        row["rows_window"] = int(len(d))
        # OHLCV validity requires all five OHLCV cols present and non-null
        if all(c in d.columns for c in req_ohlcv):
            valid_ohlcv = (
                pd.to_numeric(d["open"], errors="coerce").notna()
                & pd.to_numeric(d["high"], errors="coerce").notna()
                & pd.to_numeric(d["low"], errors="coerce").notna()
                & pd.to_numeric(d["close"], errors="coerce").notna()
                & pd.to_numeric(d["volume"], errors="coerce").notna()
            )
            ohlcv_cov = _safe_ratio(float(valid_ohlcv.sum()), float(exp_rows))
        else:
            ohlcv_cov = 0.0
        row["ohlcv_cov"] = float(max(0.0, min(1.0, ohlcv_cov)))

        if has_oi and "open_interest" in d.columns:
            oi_cov = _safe_ratio(float(pd.to_numeric(d["open_interest"], errors="coerce").notna().sum()), float(exp_rows))
        else:
            oi_cov = 0.0
        row["oi_cov"] = float(max(0.0, min(1.0, oi_cov)))

        if has_fr and "funding_rate" in d.columns:
            fr_cov = _safe_ratio(float(pd.to_numeric(d["funding_rate"], errors="coerce").notna().sum()), float(exp_rows))
        else:
            fr_cov = 0.0
        row["funding_cov"] = float(max(0.0, min(1.0, fr_cov)))

        if row["ohlcv_cov"] < float(min_ohlcv_coverage):
            issues.append("low_ohlcv_coverage")
        if row["oi_cov"] < float(min_oi_coverage):
            issues.append("low_oi_coverage")
        if row["funding_cov"] < float(min_funding_coverage):
            issues.append("low_funding_coverage")

        row["issues"] = "|".join(sorted(set(issues)))
        rows.append(row)

    dres = pd.DataFrame(rows)
    if dres.empty:
        diag = {
            "status": "unknown",
            "notes": "no_symbols_to_check",
            "symbols_total": 0,
            "symbols_checked": 0,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "issues": {},
            "coverage": {},
            "top_issue_symbols": [],
            "incident_recommended": False,
        }
        return diag, dres

    def _count_issue(name: str) -> int:
        return int(dres["issues"].astype(str).str.contains(name, regex=False).sum())

    issues = {
        "missing_file": _count_issue("missing_file"),
        "stale": _count_issue("stale"),
        "low_ohlcv_coverage": _count_issue("low_ohlcv_coverage"),
        "low_oi_coverage": _count_issue("low_oi_coverage"),
        "low_funding_coverage": _count_issue("low_funding_coverage"),
    }

    # critical signal: file/staleness/ohlcv failures
    critical = int(issues["missing_file"] + issues["stale"] + issues["low_ohlcv_coverage"])
    warning_only = int(issues["low_oi_coverage"] + issues["low_funding_coverage"])
    if critical > 0:
        status = "warn"
    elif warning_only > 0:
        status = "info"
    else:
        status = "ok"

    # top issue symbols by number of issue tags then staleness
    tmp = dres.copy()
    tmp["_issue_n"] = tmp["issues"].astype(str).apply(lambda x: 0 if x.strip() == "" else len([t for t in x.split("|") if t]))
    tmp = tmp.sort_values(["_issue_n", "staleness_hours"], ascending=[False, False], kind="mergesort")
    top_rows: List[Dict[str, object]] = []
    for _, r in tmp[tmp["_issue_n"] > 0].head(max(1, int(top_n))).iterrows():
        top_rows.append(
            {
                "symbol": str(r.get("symbol", "")),
                "issues": str(r.get("issues", "")),
                "latest_ts": str(r.get("latest_ts", "")),
                "staleness_hours": _safe_float(r.get("staleness_hours"), float("nan")),
                "ohlcv_cov": _safe_float(r.get("ohlcv_cov"), float("nan")),
                "oi_cov": _safe_float(r.get("oi_cov"), float("nan")),
                "funding_cov": _safe_float(r.get("funding_cov"), float("nan")),
            }
        )

    cov = {
        "ohlcv_cov_median": _float_quantile(dres["ohlcv_cov"].astype(float).tolist(), 0.5),
        "oi_cov_median": _float_quantile(dres["oi_cov"].astype(float).tolist(), 0.5),
        "funding_cov_median": _float_quantile(dres["funding_cov"].astype(float).tolist(), 0.5),
        "staleness_hours_p90": _float_quantile(
            [float(x) for x in dres["staleness_hours"].astype(float).tolist() if pd.notna(x)], 0.9
        ),
    }
    incident_recommended = bool(
        issues["missing_file"] > 0 or issues["stale"] > 0 or issues["low_ohlcv_coverage"] > 0
    )

    diag: Dict[str, object] = {
        "status": status,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "symbols_total": int(len(syms)),
        "symbols_checked": int(len(dres)),
        "thresholds": {
            "stale_max_hours": float(stale_max_hours),
            "min_ohlcv_coverage": float(min_ohlcv_coverage),
            "min_oi_coverage": float(min_oi_coverage),
            "min_funding_coverage": float(min_funding_coverage),
        },
        "issues": issues,
        "coverage": cov,
        "top_issue_symbols": top_rows,
        "incident_recommended": incident_recommended,
    }
    return diag, dres


def _build_telegram_body(run_dir: Path, summary: Dict[str, object]) -> str:
    row = summary.get("row_stats", {}) if isinstance(summary, dict) else {}
    agr = summary.get("agreement", {}) if isinstance(summary, dict) else {}
    tr = summary.get("trade_stats", {}) if isinstance(summary, dict) else {}
    s_live = (tr.get("live") or {}) if isinstance(tr, dict) else {}
    s_bt = (tr.get("backtest") or {}) if isinstance(tr, dict) else {}
    schema_diag = summary.get("live_schema_diag", {}) if isinstance(summary, dict) else {}
    data_sla_diag = summary.get("data_sla_diag", {}) if isinstance(summary, dict) else {}
    lines = [
        f"status={summary.get('status')}",
        f"overlap={row.get('overlap_rows')}/{row.get('live_rows')} ({row.get('overlap_rate_live')})",
        f"enter_agreement={agr.get('enter_agreement')}",
        f"reason_agreement={agr.get('reason_agreement')}",
        f"live_trades_rows={s_live.get('rows')} bt_trades_rows={s_bt.get('rows')}",
        f"live_total_pnl={s_live.get('total_pnl')} bt_total_pnl={s_bt.get('total_pnl')}",
        f"report={run_dir / 'compare' / 'report.html'}",
    ]
    lines.extend(_schema_alert_lines(schema_diag))
    lines.extend(_data_sla_alert_lines(data_sla_diag))
    return "\n".join(lines)


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _read_trade_stats(path: Optional[Path], side: str) -> Dict[str, object]:
    out: Dict[str, object] = {"side": side, "present": False, "rows": 0}
    if path is None:
        return out
    p = Path(path)
    if not p.exists():
        return out
    out["present"] = True
    try:
        if p.suffix.lower() == ".parquet":
            d = pd.read_parquet(p)
        else:
            d = pd.read_csv(p, low_memory=False)
        out["rows"] = int(len(d))
        if "pnl" in d.columns:
            pnl = pd.to_numeric(d["pnl"], errors="coerce").dropna()
            if not pnl.empty:
                out["total_pnl"] = float(pnl.sum())
    except Exception as e:
        out["error"] = f"{type(e).__name__}:{e}"
    return out


def _classify_live_skip_reason(reason: object) -> str:
    s = str(reason or "").strip().lower()
    if "schema_fail" in s or "feature_missing" in s:
        return "schema_fail"
    if "scope_fail" in s:
        return "scope_fail"
    if ("below_pstar" in s) or ("meta_prob" in s) or ("below_threshold" in s):
        return "below_pstar"
    return "gate_fail"


def _build_no_signals_failure_bucket_audit(live_decisions: Path) -> Dict[str, object]:
    canonical = {
        "no_signals": 0,
        "schema_fail": 0,
        "scope_fail": 0,
        "below_pstar": 0,
        "gate_fail": 0,
        "other": 0,
    }
    reason_bucket_counts = {
        "schema_fail": 0,
        "scope_fail": 0,
        "below_pstar": 0,
        "gate_fail": 0,
    }
    rows = 0
    skip_rows = 0
    try:
        d = pd.read_csv(live_decisions, low_memory=False)
        rows = int(len(d))
        if rows <= 0:
            return {
                "status": "ok",
                "skip_rows_live": 0,
                "canonical_bucket_counts": canonical,
                "reason_bucket_counts_live": reason_bucket_counts,
                "assigned_rate_live": float("nan"),
                "pipeline_failure_rows": 0,
                "no_edge_rows": 0,
            }
        dec_col = None
        for c in ("decision", "meta_ok"):
            if c in d.columns:
                dec_col = c
                break
        if dec_col is not None:
            if dec_col == "decision":
                v = d[dec_col].astype(str).str.strip().str.lower()
                skip_mask = ~(v.isin(["taken", "enter", "open", "true", "1", "yes"]))
            else:
                v = d[dec_col].astype(str).str.strip().str.lower()
                skip_mask = ~(v.isin(["true", "1", "yes", "taken", "open"]))
        else:
            skip_mask = pd.Series([True] * rows, index=d.index)
        skip_rows = int(skip_mask.sum())
        canonical["no_signals"] = skip_rows
        reason_col = "reason" if "reason" in d.columns else ("decision_reason" if "decision_reason" in d.columns else None)
        if reason_col is not None and skip_rows > 0:
            for r in d.loc[skip_mask, reason_col].tolist():
                b = _classify_live_skip_reason(r)
                reason_bucket_counts[b] = int(reason_bucket_counts.get(b, 0) + 1)
    except Exception:
        pass
    pipeline_rows = int(canonical["no_signals"] + canonical["schema_fail"])
    no_edge_rows = int(canonical["scope_fail"] + canonical["below_pstar"] + canonical["gate_fail"])
    assigned_rate = float(skip_rows > 0)
    return {
        "status": "ok",
        "skip_rows_live": skip_rows,
        "canonical_bucket_counts": canonical,
        "reason_bucket_counts_live": reason_bucket_counts,
        "assigned_rate_live": assigned_rate,
        "pipeline_failure_rows": pipeline_rows,
        "no_edge_rows": no_edge_rows,
    }


def _write_no_signals_report(compare_dir: Path, run_id: str, summary: Dict[str, object]) -> Path:
    compare_dir.mkdir(parents=True, exist_ok=True)
    html_path = compare_dir / "report.html"
    row = summary.get("row_stats", {}) if isinstance(summary, dict) else {}
    tr = summary.get("trade_stats", {}) if isinstance(summary, dict) else {}
    live_t = tr.get("live", {}) if isinstance(tr, dict) else {}
    bt_t = tr.get("backtest", {}) if isinstance(tr, dict) else {}
    checks = summary.get("checks", {}) if isinstance(summary, dict) else {}
    fb_diag = summary.get("failure_bucket_audit", {}) if isinstance(summary, dict) else {}
    schema_diag = summary.get("live_schema_diag", {}) if isinstance(summary, dict) else {}
    data_sla_diag = summary.get("data_sla_diag", {}) if isinstance(summary, dict) else {}
    data_sla_issues = (data_sla_diag.get("issues") or {}) if isinstance(data_sla_diag, dict) else {}
    schema_rows = "".join(
        [
            f"<tr><td>{r.get('field')}</td><td>{r.get('count')}</td></tr>"
            for r in (schema_diag.get("top_missing_fields") if isinstance(schema_diag, dict) else [])
            if isinstance(r, dict)
        ]
    ) or "<tr><td colspan='2'>none</td></tr>"
    schema_sym_rows = "".join(
        [
            f"<tr><td>{r.get('symbol')}</td><td>{r.get('count')}</td></tr>"
            for r in (schema_diag.get("top_schema_fail_symbols") if isinstance(schema_diag, dict) else [])
            if isinstance(r, dict)
        ]
    ) or "<tr><td colspan='2'>none</td></tr>"
    fb_rows = "".join(
        [
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in (fb_diag.get("canonical_bucket_counts") or {}).items()
        ]
    ) or "<tr><td colspan='2'>none</td></tr>"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Donch Autopar {run_id} (No Signals)</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
h1,h2 {{ margin: 0.4em 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
</style></head><body>
<h1>Donch Autopar {run_id}</h1>
<p>Status: {summary.get("status")}</p>
<p>{summary.get("notes", "")}</p>
<h2>Rows</h2>
<table>
<tr><th>live_rows</th><th>bt_rows</th><th>overlap_rows</th></tr>
<tr><td>{row.get("live_rows")}</td><td>{row.get("bt_rows")}</td><td>{row.get("overlap_rows")}</td></tr>
</table>
<h2>Trade Check</h2>
<table>
<tr><th>live_trades_rows</th><th>bt_trades_rows</th><th>no_trade_parity_pass</th></tr>
<tr><td>{live_t.get("rows")}</td><td>{bt_t.get("rows")}</td><td>{checks.get("no_trade_parity_pass")}</td></tr>
</table>
<h2>Failure Buckets (No-Signals Window)</h2>
<p>assigned_rate_live={fb_diag.get("assigned_rate_live")} pipeline_failure_rows={fb_diag.get("pipeline_failure_rows")} no_edge_rows={fb_diag.get("no_edge_rows")}</p>
<table>
<tr><th>bucket</th><th>count</th></tr>
{fb_rows}
</table>
<h2>Live Schema Diagnostics</h2>
<table>
<tr><th>schema_fail_rows</th><th>rows_total</th><th>schema_fail_rate</th><th>alert_level</th><th>expected_warmup_only</th></tr>
<tr><td>{schema_diag.get("schema_fail_rows")}</td><td>{schema_diag.get("rows_total")}</td><td>{schema_diag.get("schema_fail_rate")}</td><td>{schema_diag.get("alert_level")}</td><td>{schema_diag.get("expected_warmup_only")}</td></tr>
</table>
<h2>Top Missing Fields</h2>
<table>
<tr><th>field</th><th>count</th></tr>
{schema_rows}
</table>
<h2>Top Schema-Fail Symbols</h2>
<table>
<tr><th>symbol</th><th>count</th></tr>
{schema_sym_rows}
</table>
<h2>Data Freshness &amp; Completeness (SLA)</h2>
<table>
<tr><th>status</th><th>symbols_checked</th><th>missing_file</th><th>stale</th><th>low_ohlcv</th><th>low_oi</th><th>low_funding</th><th>incident_recommended</th></tr>
<tr>
<td>{(data_sla_diag.get("status") if isinstance(data_sla_diag, dict) else "")}</td>
<td>{((data_sla_diag.get("symbols_checked") if isinstance(data_sla_diag, dict) else ""))}</td>
<td>{(data_sla_issues.get("missing_file") if isinstance(data_sla_issues, dict) else "")}</td>
<td>{(data_sla_issues.get("stale") if isinstance(data_sla_issues, dict) else "")}</td>
<td>{(data_sla_issues.get("low_ohlcv_coverage") if isinstance(data_sla_issues, dict) else "")}</td>
<td>{(data_sla_issues.get("low_oi_coverage") if isinstance(data_sla_issues, dict) else "")}</td>
<td>{(data_sla_issues.get("low_funding_coverage") if isinstance(data_sla_issues, dict) else "")}</td>
<td>{(data_sla_diag.get("incident_recommended") if isinstance(data_sla_diag, dict) else "")}</td>
</tr>
</table>
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _build_no_signals_summary(
    live_decisions: Path,
    live_trades: Optional[Path],
    bt_trades: Optional[Path],
    schema_diag: Optional[Dict[str, object]] = None,
    data_sla_diag: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    live_rows = 0
    try:
        d = pd.read_csv(live_decisions, low_memory=False)
        live_rows = int(len(d))
    except Exception:
        live_rows = 0
    live_trade_stats = _read_trade_stats(live_trades, "live")
    bt_trade_stats = _read_trade_stats(bt_trades, "backtest")
    live_trade_rows = _safe_int(live_trade_stats.get("rows"), 0)
    bt_trade_rows = _safe_int(bt_trade_stats.get("rows"), 0)
    no_trade_parity_pass = bool(live_trade_rows == bt_trade_rows == 0)
    failure_bucket_audit = _build_no_signals_failure_bucket_audit(live_decisions)
    return {
        "status": "no_signals",
        "bucket": "5min",
        "row_stats": {
            "live_rows": live_rows,
            "bt_rows": 0,
            "overlap_rows": 0,
            "live_only_rows": live_rows,
            "bt_only_rows": 0,
            "overlap_rate_live": 0.0 if live_rows > 0 else 1.0,
            "overlap_rate_bt": float("nan"),
        },
        "agreement": {
            "enter_agreement": float("nan"),
            "reason_agreement": float("nan"),
            "confusion": {},
            "p_abs_err_mean": float("nan"),
            "p_abs_err_p90": float("nan"),
            "size_abs_err_mean": float("nan"),
            "size_abs_err_p90": float("nan"),
        },
        "trade_stats": {"live": live_trade_stats, "backtest": bt_trade_stats},
        "failure_bucket_audit": failure_bucket_audit,
        "live_schema_diag": schema_diag or {},
        "data_sla_diag": data_sla_diag or {},
        "checks": {
            "no_trade_parity_pass": no_trade_parity_pass,
            "live_trades_rows": live_trade_rows,
            "bt_trades_rows": bt_trade_rows,
            "failure_bucket_assigned_rate_live": failure_bucket_audit.get("assigned_rate_live"),
            "failure_bucket_assigned_rate_live_min": 0.95,
        },
        "notes": "No reference scout signals were generated for this window; decision parity is unavailable. Trade-count parity was evaluated.",
    }


def main() -> int:
    a = parse_args()
    rid = a.run_id.strip() or _utc_id()
    run_dir = (_to_abs(a.results_root) / rid).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    notifier = TelegramNotifier.from_args(a, run_label=f"autopar:{rid}")
    print(f"[autopar] telegram notify: {notifier.status_line()}", flush=True)
    notifier.send("STARTED", body=f"run_id={rid}\nlive_input={a.live_input}")

    live_input = _to_abs(a.live_input)
    model_dir = _to_abs(a.model_dir)
    parquet_dir = Path(a.parquet_dir).expanduser().resolve()
    parq1m = Path(a.parquet_1m_dir).expanduser().resolve() if a.parquet_1m_dir.strip() else None

    live_stage = run_dir / "live_stage"
    package_dir = _extract_live_package(live_input, live_stage)
    (run_dir / "package_source.txt").write_text(str(package_dir), encoding="utf-8")

    live_decisions = _to_abs(a.live_decisions) if a.live_decisions.strip() else None
    if live_decisions is None:
        live_decisions = _find_first(package_dir, ["live_decisions.csv", "decisions.csv", "meta_decisions.csv"])

    live_trades = _to_abs(a.live_trades) if a.live_trades.strip() else None
    if live_trades is None:
        live_trades = _find_first(package_dir, ["live_trades.csv", "trades.csv"])

    symbols_file = _to_abs(a.symbols_file) if a.symbols_file.strip() else None
    if symbols_file is None:
        symbols_file = _find_first(package_dir, ["symbols_active.txt", "symbols.txt"])

    # If no decisions csv, try extracting from logs.
    if live_decisions is None:
        log_candidate = _find_first(package_dir, ["live.log", "bot.log"])
        if log_candidate is None:
            logs_dir = package_dir / "logs"
            if logs_dir.exists():
                log_candidate = logs_dir
        if log_candidate is not None:
            extracted = run_dir / "live_extracted" / "live_decisions.csv"
            extract_cmd = [
                a.python,
                str((REPO_ROOT / "tools" / "donch_autopar_extract_live_meta_decisions.py").resolve()),
                "--input",
                str(log_candidate),
                "--out",
                str(extracted),
            ]
            rc = _run_cmd(extract_cmd, cwd=REPO_ROOT, log_path=run_dir / "logs" / "00_extract.log")
            if rc == 0 and extracted.exists():
                live_decisions = extracted

    if live_decisions is None or (not live_decisions.exists()):
        notifier.send("FAILED", body="live decisions file not found")
        raise SystemExit("Could not locate or extract live decisions CSV.")

    live_schema_diag = _build_live_schema_diag(live_decisions, package_dir, top_n=8)
    live_schema_diag_path = run_dir / "live_schema_diag.json"
    live_schema_diag_path.write_text(json.dumps(live_schema_diag, indent=2), encoding="utf-8")

    start_s = (a.start or "").strip()
    end_s = (a.end or "").strip()
    if not (start_s and end_s):
        ctx_win = _infer_window_from_context(package_dir)
        if ctx_win is not None:
            start_s, end_s = ctx_win
    if not (start_s and end_s):
        start_s, end_s = _infer_window(live_decisions, a.start, a.end, a.window_days)
    print(f"[autopar] window start={start_s} end={end_s}", flush=True)

    data_sla_diag, data_sla_details = _build_data_sla_diag(
        parquet_dir=parquet_dir,
        symbols_file=symbols_file,
        live_decisions_path=live_decisions,
        start_s=start_s,
        end_s=end_s,
        stale_max_hours=float(a.sla_stale_max_hours),
        min_ohlcv_coverage=float(a.sla_min_ohlcv_coverage),
        min_oi_coverage=float(a.sla_min_oi_coverage),
        min_funding_coverage=float(a.sla_min_funding_coverage),
        top_n=int(max(1, a.sla_top_n)),
        max_symbols=int(max(0, a.sla_max_symbols)),
    )
    data_sla_diag_path = run_dir / "data_sla_diag.json"
    data_sla_diag_path.write_text(json.dumps(data_sla_diag, indent=2), encoding="utf-8")
    data_sla_details_path = run_dir / "data_sla_details.csv"
    try:
        data_sla_details.to_csv(data_sla_details_path, index=False)
    except Exception:
        pass

    ref_signals = run_dir / "reference" / "_scoped_signals"
    ref_aux = run_dir / "reference" / "_aux_results"
    ref_bt = run_dir / "reference" / "backtest"

    if (not a.skip_scout) and (not (a.resume and ref_signals.exists() and any(ref_signals.glob("symbol=*")))):
        env_scout = {
            "DONCH_REPO_ROOT": str(REPO_ROOT),
            "DONCH_PARQUET_DIR": str(parquet_dir),
            "DONCH_SIGNALS_DIR": str(ref_signals),
            "DONCH_AUX_RESULTS_DIR": str(ref_aux),
            "DONCH_START": str(start_s),
            "DONCH_END": str(end_s),
            "DONCH_SCOUT_WORKERS": str(max(1, int(a.scout_workers))),
            "DONCH_SCOUT_BACKEND": str(a.scout_backend),
            "DONCH_SCOUT_CLEAN": "1",
        }
        if symbols_file is not None:
            env_scout["DONCH_SYMBOLS_FILE"] = str(symbols_file)
        rc = _run_cmd(
            [a.python, "-u", "-c", SCOUT_CODE],
            cwd=REPO_ROOT,
            log_path=run_dir / "logs" / "01_scout.log",
            env=env_scout,
        )
        if rc != 0:
            notifier.send("FAILED", body=f"scout failed rc={rc}\nlog={run_dir / 'logs' / '01_scout.log'}")
            raise SystemExit(f"Scout failed rc={rc}")

    need_signals = not bool(a.skip_backtest)
    no_signals = bool(need_signals and (not ref_signals.exists() or not any(ref_signals.glob("symbol=*"))))
    if no_signals:
        # Let backtester run on an empty signals dir so it can still emit empty trades outputs.
        ref_signals.mkdir(parents=True, exist_ok=True)

    bt_resume_ready = (ref_bt / "signal_decisions.csv").exists() or (
        no_signals and (ref_bt / "trades.csv").exists()
    )
    if (not a.skip_backtest) and (not (a.resume and bt_resume_ready)):
        ov = _default_bt_overrides(model_dir)
        if a.bt_overrides_json.strip():
            extra = json.loads(a.bt_overrides_json)
            if not isinstance(extra, dict):
                raise SystemExit("--bt-overrides-json must parse to an object")
            ov.update(extra)
        env_bt = {
            "DONCH_REPO_ROOT": str(REPO_ROOT),
            "DONCH_PARQUET_DIR": str(parquet_dir),
            "DONCH_PARQUET_1M_DIR": str(parq1m) if parq1m else "",
            "DONCH_SIGNALS_DIR": str(ref_signals),
            "DONCH_BT_OUT_DIR": str(ref_bt),
            "DONCH_META_MODEL_DIR": str(model_dir),
            "DONCH_START": str(start_s),
            "DONCH_END": str(end_s),
            "DONCH_OVERRIDES_JSON": json.dumps(ov, sort_keys=True),
        }
        rc = _run_cmd(
            [a.python, "-u", "-c", BACKTEST_CODE],
            cwd=REPO_ROOT,
            log_path=run_dir / "logs" / "02_backtest.log",
            env=env_bt,
        )
        if rc != 0:
            notifier.send("FAILED", body=f"backtest failed rc={rc}\nlog={run_dir / 'logs' / '02_backtest.log'}")
            raise SystemExit(f"Backtest failed rc={rc}")

    bt_decisions = ref_bt / "signal_decisions.csv"
    bt_trades = ref_bt / "trades.csv"
    if not bt_decisions.exists():
        if no_signals:
            compare_dir = run_dir / "compare"
            compare_dir.mkdir(parents=True, exist_ok=True)
            summary = _build_no_signals_summary(
                live_decisions,
                live_trades,
                bt_trades,
                schema_diag=live_schema_diag,
                data_sla_diag=data_sla_diag,
            )
            summary_path = compare_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            report_html = _write_no_signals_report(compare_dir, rid, summary)

            manifest = {
                "run_id": rid,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "no_signals",
                "window_start": start_s,
                "window_end": end_s,
                "live_input": str(live_input),
                "live_package_dir": str(package_dir),
                "live_decisions": str(live_decisions),
                "live_trades": str(live_trades) if live_trades else "",
                "symbols_file": str(symbols_file) if symbols_file else "",
                "parquet_dir": str(parquet_dir),
                "model_dir": str(model_dir),
                "run_dir": str(run_dir),
                "summary_json": str(summary_path),
                "report_html": str(report_html),
                "live_schema_diag_json": str(live_schema_diag_path),
                "data_sla_diag_json": str(data_sla_diag_path),
                "data_sla_details_csv": str(data_sla_details_path),
                "note": "No scout signals for the selected window; decision parity unavailable; trade-count parity evaluated.",
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            body = (
                "status=no_signals\n"
                f"live_rows={_safe_int((summary.get('row_stats') or {}).get('live_rows'))}\n"
                f"live_trades_rows={_safe_int(((summary.get('trade_stats') or {}).get('live') or {}).get('rows'))}\n"
                f"bt_trades_rows={_safe_int(((summary.get('trade_stats') or {}).get('backtest') or {}).get('rows'))}\n"
                f"no_trade_parity_pass={bool((summary.get('checks') or {}).get('no_trade_parity_pass'))}\n"
                f"window={start_s}..{end_s}\nrun_dir={run_dir}"
            )
            schema_lines = _schema_alert_lines(summary.get("live_schema_diag", {}))
            if schema_lines:
                body = body + "\n" + "\n".join(schema_lines)
            sla_lines = _data_sla_alert_lines(summary.get("data_sla_diag", {}))
            if sla_lines:
                body = body + "\n" + "\n".join(sla_lines)
            notifier.send("WARN", body=body)
            print(f"[autopar] done run_id={rid} status=no_signals", flush=True)
            print(f"[autopar] manifest={run_dir / 'run_manifest.json'}", flush=True)
            return 0
        notifier.send("FAILED", body=f"missing {bt_decisions}")
        raise SystemExit(f"Missing {bt_decisions}")

    compare_dir = run_dir / "compare"
    compare_cmd = [
        a.python,
        str((REPO_ROOT / "tools" / "donch_autopar_compare.py").resolve()),
        "--live-decisions",
        str(live_decisions),
        "--bt-decisions",
        str(bt_decisions),
        "--outdir",
        str(compare_dir),
        "--min-overlap-rate",
        str(a.min_overlap_rate),
        "--min-enter-agreement",
        str(a.min_enter_agreement),
        "--html-title",
        f"Donch Autopar {rid}",
    ]
    if live_trades is not None and live_trades.exists():
        compare_cmd += ["--live-trades", str(live_trades)]
    if bt_trades.exists():
        compare_cmd += ["--bt-trades", str(bt_trades)]

    rc_cmp = _run_cmd(compare_cmd, cwd=REPO_ROOT, log_path=run_dir / "logs" / "03_compare.log")
    if rc_cmp not in (0, 2):
        notifier.send("FAILED", body=f"compare failed rc={rc_cmp}\nlog={run_dir / 'logs' / '03_compare.log'}")
        raise SystemExit(f"Compare failed rc={rc_cmp}")

    summary_path = compare_dir / "summary.json"
    if not summary_path.exists():
        notifier.send("FAILED", body=f"missing compare summary: {summary_path}")
        raise SystemExit(f"Missing compare summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["live_schema_diag"] = live_schema_diag
    summary["data_sla_diag"] = data_sla_diag
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _append_data_sla_to_report(compare_dir / "report.html", data_sla_diag)

    manifest = {
        "run_id": rid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": summary.get("status"),
        "window_start": start_s,
        "window_end": end_s,
        "live_input": str(live_input),
        "live_package_dir": str(package_dir),
        "live_decisions": str(live_decisions),
        "live_trades": str(live_trades) if live_trades else "",
        "symbols_file": str(symbols_file) if symbols_file else "",
        "parquet_dir": str(parquet_dir),
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "report_html": str(compare_dir / "report.html"),
        "live_schema_diag_json": str(live_schema_diag_path),
        "data_sla_diag_json": str(data_sla_diag_path),
        "data_sla_details_csv": str(data_sla_details_path),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    body = _build_telegram_body(run_dir, summary)
    if summary.get("status") == "ok":
        notifier.send("DONE", body=body)
    else:
        notifier.send("WARN", body=body)

    print(f"[autopar] done run_id={rid} status={summary.get('status')}", flush=True)
    print(f"[autopar] manifest={run_dir / 'run_manifest.json'}", flush=True)
    return 0 if summary.get("status") != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
