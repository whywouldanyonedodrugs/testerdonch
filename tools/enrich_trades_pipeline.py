#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CsvStats:
    path: str
    rows: int
    cols: int
    has_trade_id: bool
    trade_id_non_null: Optional[int]
    trade_id_unique: Optional[int]


@dataclass
class StageReport:
    stage: str
    status: str
    input_path: str
    output_path: str
    seconds: float
    rows_in: int
    rows_out: int
    command: Optional[List[str]] = None
    note: Optional[str] = None


def _resolve_path(p: str) -> Path:
    out = Path(p).expanduser()
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    return out


def _row_count(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def _header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _csv_stats(path: Path) -> CsvStats:
    cols = _header(path)
    rows = _row_count(path)
    has_trade_id = "trade_id" in cols
    trade_id_non_null = None
    trade_id_unique = None
    if has_trade_id:
        ser = pd.read_csv(path, usecols=["trade_id"], low_memory=False)["trade_id"]
        ser = pd.to_numeric(ser, errors="coerce")
        trade_id_non_null = int(ser.notna().sum())
        trade_id_unique = int(ser.nunique(dropna=True))
    return CsvStats(
        path=str(path),
        rows=rows,
        cols=len(cols),
        has_trade_id=has_trade_id,
        trade_id_non_null=trade_id_non_null,
        trade_id_unique=trade_id_unique,
    )


def _validate_stage_io(
    stage: str,
    before: CsvStats,
    after: CsvStats,
    *,
    allow_row_drop: bool,
    required_cols: List[str],
) -> None:
    missing = [c for c in required_cols if c not in _header(Path(after.path))]
    if missing:
        raise RuntimeError(f"[{stage}] output is missing required columns: {missing}")

    if (not allow_row_drop) and before.rows != after.rows:
        raise RuntimeError(
            f"[{stage}] row count changed: in={before.rows} out={after.rows}. "
            "Pass --allow-row-drop if intentional."
        )

    if before.has_trade_id and after.has_trade_id and (not allow_row_drop):
        if before.trade_id_non_null != after.trade_id_non_null:
            raise RuntimeError(
                f"[{stage}] trade_id non-null count changed: "
                f"in={before.trade_id_non_null} out={after.trade_id_non_null}"
            )
        if before.trade_id_unique != after.trade_id_unique:
            raise RuntimeError(
                f"[{stage}] trade_id unique count changed: "
                f"in={before.trade_id_unique} out={after.trade_id_unique}"
            )


def _run_cmd(cmd: List[str]) -> None:
    print(f"[pipeline] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_stage(
    *,
    stage: str,
    script: str,
    in_path: Path,
    out_path: Path,
    python_exe: str,
    script_args: List[str],
    required_cols: List[str],
    allow_row_drop: bool,
    resume: bool,
    allow_failure: bool = False,
) -> tuple[Path, StageReport]:
    if resume and out_path.exists():
        before = _csv_stats(in_path)
        after = _csv_stats(out_path)
        _validate_stage_io(
            stage,
            before,
            after,
            allow_row_drop=allow_row_drop,
            required_cols=required_cols,
        )
        rep = StageReport(
            stage=stage,
            status="skipped_resume",
            input_path=str(in_path),
            output_path=str(out_path),
            seconds=0.0,
            rows_in=before.rows,
            rows_out=after.rows,
            command=None,
            note="reused existing stage output",
        )
        print(f"[pipeline] {stage}: reusing {out_path}", flush=True)
        return out_path, rep

    before = _csv_stats(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, str(REPO_ROOT / script), *script_args]
    t0 = time.time()
    try:
        _run_cmd(cmd)
    except Exception as exc:
        dt = time.time() - t0
        if allow_failure:
            rep = StageReport(
                stage=stage,
                status="failed_optional",
                input_path=str(in_path),
                output_path=str(in_path),
                seconds=dt,
                rows_in=before.rows,
                rows_out=before.rows,
                command=cmd,
                note=f"{type(exc).__name__}: {exc}",
            )
            print(f"[pipeline] {stage}: optional stage failed, continuing with input file", flush=True)
            return in_path, rep
        raise

    if not out_path.exists():
        raise RuntimeError(f"[{stage}] expected output does not exist: {out_path}")

    after = _csv_stats(out_path)
    _validate_stage_io(
        stage,
        before,
        after,
        allow_row_drop=allow_row_drop,
        required_cols=required_cols,
    )
    dt = time.time() - t0
    rep = StageReport(
        stage=stage,
        status="ok",
        input_path=str(in_path),
        output_path=str(out_path),
        seconds=dt,
        rows_in=before.rows,
        rows_out=after.rows,
        command=cmd,
        note=None,
    )
    print(f"[pipeline] {stage}: rows {before.rows} -> {after.rows} ({dt:.1f}s)", flush=True)
    return out_path, rep


def _write_parquet(csv_path: Path) -> Optional[Path]:
    pq_path = csv_path.with_suffix(".parquet")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df.to_parquet(pq_path, index=False)
        print(f"[pipeline] wrote parquet: {pq_path}", flush=True)
        return pq_path
    except Exception as exc:
        print(f"[pipeline] warning: failed to write parquet for {csv_path}: {exc}", flush=True)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all trade enrichment stages in one robust pipeline."
    )
    p.add_argument("--in", dest="infile", default="results/trades.csv")
    p.add_argument("--out", dest="outfile", default="results/trades.clean.csv")
    p.add_argument("--workdir", default="results/.enrich_pipeline")
    p.add_argument("--python", dest="python_exe", default=sys.executable)
    p.add_argument("--resume", action="store_true", help="Reuse stage outputs if already present.")
    p.add_argument(
        "--allow-row-drop",
        action="store_true",
        help="Allow stages to change row count/trade_id counts.",
    )
    p.add_argument("--skip-funding", action="store_true")
    p.add_argument("--skip-backfill", action="store_true")
    p.add_argument("--skip-fill", action="store_true")
    p.add_argument(
        "--funding-optional",
        action="store_true",
        help="If funding API stage fails, continue with existing columns.",
    )
    p.add_argument("--funding-throttle", type=float, default=0.0)
    p.add_argument(
        "--write-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write parquet for final/compat outputs.",
    )
    p.add_argument(
        "--write-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write results/trades.enriched.filled.csv and results/trades.clean.csv.",
    )
    p.add_argument("--compat-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = _resolve_path(args.infile)
    out_path = _resolve_path(args.outfile)
    workdir = _resolve_path(args.workdir)
    compat_dir = _resolve_path(args.compat_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"input trades file not found: {in_path}")

    workdir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compat_dir.mkdir(parents=True, exist_ok=True)

    reports: List[StageReport] = []
    cur = in_path

    if not args.skip_funding:
        cur, rep = _run_stage(
            stage="funding",
            script="enrich_trades_funding_bybit.py",
            in_path=cur,
            out_path=workdir / "01_funding.csv",
            python_exe=args.python_exe,
            script_args=[
                "--trades",
                str(cur),
                "--out",
                str(workdir / "01_funding.csv"),
                "--throttle",
                str(args.funding_throttle),
            ],
            required_cols=["symbol", "entry_ts"],
            allow_row_drop=args.allow_row_drop,
            resume=args.resume,
            allow_failure=args.funding_optional,
        )
        reports.append(rep)

    if not args.skip_backfill:
        cur, rep = _run_stage(
            stage="backfill",
            script="backfill_trade_features.py",
            in_path=cur,
            out_path=workdir / "02_backfill.csv",
            python_exe=args.python_exe,
            script_args=[
                "--trades",
                str(cur),
                "--out",
                str(workdir / "02_backfill.csv"),
            ],
            required_cols=["symbol", "entry_ts"],
            allow_row_drop=args.allow_row_drop,
            resume=args.resume,
        )
        reports.append(rep)

    if not args.skip_fill:
        cur, rep = _run_stage(
            stage="fill_entry_quality",
            script="fill_entry_quality_features.py",
            in_path=cur,
            out_path=workdir / "03_filled.csv",
            python_exe=args.python_exe,
            script_args=[
                "--infile",
                str(cur),
                "--outfile",
                str(workdir / "03_filled.csv"),
            ],
            required_cols=["symbol", "entry_ts"],
            allow_row_drop=args.allow_row_drop,
            resume=args.resume,
        )
        reports.append(rep)

    shutil.copy2(cur, out_path)
    print(f"[pipeline] final csv: {out_path}", flush=True)

    outputs = {"final_csv": str(out_path)}
    if args.write_parquet:
        pq = _write_parquet(out_path)
        if pq is not None:
            outputs["final_parquet"] = str(pq)

    if args.write_compat:
        compat_targets = [
            compat_dir / "trades.clean.csv",
            compat_dir / "trades.enriched.filled.csv",
        ]
        for tgt in compat_targets:
            if tgt.resolve() == out_path.resolve():
                continue
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, tgt)
            print(f"[pipeline] compat csv: {tgt}", flush=True)
            if args.write_parquet:
                _write_parquet(tgt)

    manifest = {
        "input": str(in_path),
        "output": str(out_path),
        "workdir": str(workdir),
        "reports": [asdict(r) for r in reports],
        "outputs": outputs,
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    man_path = workdir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[pipeline] manifest: {man_path}", flush=True)


if __name__ == "__main__":
    main()
