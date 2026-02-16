#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _parse_utc_ts(text: str) -> Optional[pd.Timestamp]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def _to_bool(v: object) -> Optional[bool]:
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def _to_float(v: object) -> Optional[float]:
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _iter_files(path: Path, pattern: str) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob(pattern)):
        if p.is_file():
            yield p


def _line_ts_utc(line: str) -> Optional[pd.Timestamp]:
    # logging format example: "2026-02-10 12:34:56 [INFO] ..."
    if len(line) < 19:
        return None
    head = line[:19]
    try:
        dt = datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt)
    except Exception:
        return None


def _parse_meta_decision_payload(payload: str) -> Dict[str, str]:
    # Payload shape in live_trader.py:
    # META_DECISION bundle=%s symbol=%s decision_ts=%s schema_ok=%s ...
    out: Dict[str, str] = {}
    for tok in payload.strip().split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract META_DECISION rows from live logs.")
    p.add_argument("--input", required=True, help="Log file or directory.")
    p.add_argument("--glob", default="*.log*", help="Glob when --input is a directory.")
    p.add_argument("--start", default="", help="Optional UTC lower bound for decision_ts.")
    p.add_argument("--end", default="", help="Optional UTC upper bound for decision_ts.")
    p.add_argument("--out", required=True, help="Output CSV path (live_decisions.csv).")
    p.add_argument("--summary-out", default="", help="Optional JSON summary path.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    in_path = Path(a.input).expanduser().resolve()
    out_path = Path(a.out).expanduser().resolve()
    sum_path = Path(a.summary_out).expanduser().resolve() if a.summary_out.strip() else out_path.with_suffix(".summary.json")

    if not in_path.exists():
        raise SystemExit(f"Input path not found: {in_path}")

    start_ts = _parse_utc_ts(a.start) if a.start.strip() else None
    end_ts = _parse_utc_ts(a.end) if a.end.strip() else None

    rows: List[Dict[str, object]] = []
    files_scanned = 0
    lines_scanned = 0
    lines_matched = 0

    for fp in _iter_files(in_path, a.glob):
        files_scanned += 1
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    lines_scanned += 1
                    mark = "META_DECISION "
                    j = line.find(mark)
                    if j < 0:
                        continue
                    lines_matched += 1
                    payload = line[j + len(mark) :]
                    kv = _parse_meta_decision_payload(payload)
                    if not kv:
                        continue

                    decision_ts = _parse_utc_ts(kv.get("decision_ts", ""))
                    if decision_ts is None:
                        continue
                    if (start_ts is not None) and (decision_ts < start_ts):
                        continue
                    if (end_ts is not None) and (decision_ts > end_ts):
                        continue

                    schema_ok = _to_bool(kv.get("schema_ok"))
                    scope_ok = _to_bool(kv.get("scope_ok"))
                    meta_ok = _to_bool(kv.get("meta_ok"))
                    strat_ok = _to_bool(kv.get("strat_ok"))
                    p_cal = _to_float(kv.get("p_cal"))
                    pstar = _to_float(kv.get("pstar"))

                    taken = bool(meta_ok) and bool(strat_ok)
                    row: Dict[str, object] = {
                        "log_ts": _line_ts_utc(line),
                        "bundle": kv.get("bundle"),
                        "symbol": (kv.get("symbol") or "").upper(),
                        "decision_ts": decision_ts,
                        "decision_ts_5m": decision_ts.floor("5min"),
                        "decision": "taken" if taken else "skipped",
                        "reason": kv.get("reason"),
                        "schema_ok": schema_ok,
                        "scope_ok": scope_ok,
                        "meta_ok": meta_ok,
                        "strat_ok": strat_ok,
                        "p_cal": p_cal,
                        "pstar": pstar,
                        "pstar_scope": kv.get("pstar_scope"),
                        "risk_on_1": kv.get("risk_on_1"),
                        "risk_on": kv.get("risk_on"),
                        "scope_val": kv.get("scope_val"),
                        "scope_src": kv.get("scope_src"),
                        "err": kv.get("err"),
                        "source_file": str(fp),
                    }
                    rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["symbol"].astype(str) != ""].copy()
        # Keep latest row per (symbol, decision_ts_5m), if duplicates exist.
        df = df.sort_values(["symbol", "decision_ts_5m", "log_ts"], kind="mergesort")
        df = df.drop_duplicates(subset=["symbol", "decision_ts_5m"], keep="last")
        df = df.sort_values(["decision_ts", "symbol"], kind="mergesort").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary: Dict[str, object] = {
        "input_path": str(in_path),
        "files_scanned": files_scanned,
        "lines_scanned": lines_scanned,
        "lines_with_meta_decision": lines_matched,
        "rows_output": int(len(df)),
        "start_filter": str(start_ts) if start_ts is not None else "",
        "end_filter": str(end_ts) if end_ts is not None else "",
        "out_csv": str(out_path),
    }
    if not df.empty:
        summary["decision_counts"] = df["decision"].value_counts(dropna=False).to_dict()
        summary["min_decision_ts"] = str(pd.to_datetime(df["decision_ts"], utc=True).min())
        summary["max_decision_ts"] = str(pd.to_datetime(df["decision_ts"], utc=True).max())
        if "reason" in df.columns:
            summary["top_reasons"] = df["reason"].value_counts(dropna=False).head(12).to_dict()

    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[autopar] wrote {out_path}")
    print(f"[autopar] wrote {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

