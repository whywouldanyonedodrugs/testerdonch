#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from telegram_notify import TelegramNotifier


REPO_ROOT = Path(__file__).resolve().parents[1]


CHILD_BACKTEST_CODE = r"""
import json, os, sys
from pathlib import Path

repo_root = Path(os.environ["DONCH_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root))

import config as cfg
import backtester

out_dir = Path(os.environ["DONCH_OUT_DIR"]).resolve()
signals_path = Path(os.environ["DONCH_SIGNALS_PATH"]).resolve()
start = os.environ["DONCH_START"]
end = os.environ["DONCH_END"]
overrides = json.loads(os.environ["DONCH_OVERRIDES_JSON"])

out_dir.mkdir(parents=True, exist_ok=True)
setattr(cfg, "RESULTS_DIR", out_dir)
setattr(cfg, "START_DATE", str(start))
setattr(cfg, "END_DATE", str(end))

if signals_path.is_dir():
    setattr(cfg, "SIGNALS_DIR", signals_path)
else:
    setattr(cfg, "SIGNALS_DIR", signals_path.parent)

for k, v in overrides.items():
    setattr(cfg, k, v)

backtester.run_backtest(signals_path=signals_path)
"""


@dataclass
class Stage:
    name: str
    cmd: List[str]
    cwd: Path
    log_path: Path
    outputs: Sequence[Path]


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _to_abs(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p.resolve()


def _run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_s = " ".join(cmd)
    print(f"[offline-release] running: {cmd_s}", flush=True)
    if dry_run:
        log_path.write_text(f"[dry-run] {cmd_s}\n", encoding="utf-8")
        return 0

    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    run_env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        p = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=run_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = int(p.wait())
    return rc


def _stage_done(outputs: Sequence[Path]) -> bool:
    if not outputs:
        return False
    return all(p.exists() for p in outputs)


def _winner_no_meta_overrides() -> Dict[str, Any]:
    return {
        "META_PROB_THRESHOLD": None,
        "META_GATE_SCOPE": "all",
        "META_GATE_FAIL_CLOSED": False,
        "REGIME_BLOCK_WHEN_DOWN": False,
        "RISK_OFF_PROBE_MULT": 0.05,
        "REGIME_SLOPE_FILTER_ENABLED": False,
        "BT_META_REPLAY_ENABLED": False,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline release orchestrator: source backtest/enrich -> retrain -> export bundle -> parity fixtures."
    )

    p.add_argument("--release-id", default="", help="Release id. Default: UTC timestamp.")
    p.add_argument("--python", default=sys.executable, help="Python executable.")

    src = p.add_argument_group("Source data")
    src.add_argument("--trades-in", default="", help="Prepared trades.clean.csv (skips source backtest/enrich).")
    src.add_argument("--signals-path", default="", help="Signals parquet/directory to run source backtest from.")
    src.add_argument("--start", default="2023-01-01")
    src.add_argument("--end", default="latest")
    src.add_argument(
        "--bt-preset",
        choices=["winner_no_meta", "none"],
        default="winner_no_meta",
        help="Backtest override preset if --signals-path is used.",
    )
    src.add_argument(
        "--bt-overrides-json",
        default="",
        help="JSON object with additional cfg overrides for source backtest.",
    )
    src.add_argument("--skip-enrich", action="store_true")
    src.add_argument("--funding-stage", choices=["on", "off", "optional"], default="optional")
    src.add_argument("--funding-throttle", type=float, default=0.0)

    out = p.add_argument_group("Outputs")
    out.add_argument("--release-root", default="results/offline_releases")
    out.add_argument("--publish-dir", default="", help="Optional copy/symlink target for final bundle.")
    out.add_argument(
        "--publish-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to publish to --publish-dir.",
    )

    model = p.add_argument_group("Modeling")
    model.add_argument("--target", default="y_good_05")
    model.add_argument("--train-scope", choices=["ALL", "RISK_ON_1"], default="ALL")
    model.add_argument("--fit-scope", choices=["ALL", "RISK_ON_1"], default="ALL")
    model.add_argument(
        "--decision-scope",
        default="",
        help="Threshold decision scope for 07 export. Default auto.",
    )
    model.add_argument("--criterion", choices=["mean", "sum"], default="mean")
    model.add_argument("--include-regimes-as-features", action="store_true")
    model.add_argument(
        "--live-safe-features",
        action="store_true",
        help="Enable decision-time-only feature filtering in 04_models_cv.",
    )
    model.add_argument("--skip-step06", action="store_true")
    model.add_argument("--bundle-tar", action="store_true")

    parity = p.add_argument_group("Parity package")
    parity.add_argument("--skip-golden", action="store_true")
    parity.add_argument("--golden-rows", type=int, default=1200)
    parity.add_argument("--golden-min-symbols", type=int, default=10)
    parity.add_argument("--golden-max-symbols", type=int, default=25)
    parity.add_argument("--golden-no-score", action="store_true")

    notify = p.add_argument_group("Notifications")
    notify.add_argument("--tg-bot-token", default="")
    notify.add_argument("--tg-chat-id", default="")
    notify.add_argument(
        "--tg-auto-chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try Telegram getUpdates to auto-discover chat id when not provided.",
    )

    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _build_source_trades(args: argparse.Namespace, release_dir: Path, dry_run: bool) -> Path:
    if bool(args.resume):
        existing_clean = release_dir / "inputs" / "trades.clean.csv"
        if existing_clean.exists():
            print(f"[offline-release] resume: using existing source trades at {existing_clean}", flush=True)
            return existing_clean

    if args.trades_in:
        in_path = _to_abs(args.trades_in)
        if not in_path.exists():
            raise FileNotFoundError(f"--trades-in not found: {in_path}")
        out_path = release_dir / "inputs" / "trades.clean.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"[offline-release] dry-run copy: {in_path} -> {out_path}", flush=True)
        else:
            shutil.copy2(in_path, out_path)
        return out_path

    if not args.signals_path:
        raise ValueError("Provide either --trades-in or --signals-path.")

    signals_path = _to_abs(args.signals_path)
    if not signals_path.exists():
        raise FileNotFoundError(f"--signals-path not found: {signals_path}")

    source_bt_dir = release_dir / "source_backtest"
    source_bt_dir.mkdir(parents=True, exist_ok=True)
    bt_log = release_dir / "logs" / "00_source_backtest.log"

    ov: Dict[str, Any] = {}
    if args.bt_preset == "winner_no_meta":
        ov.update(_winner_no_meta_overrides())
    if args.bt_overrides_json.strip():
        extra = json.loads(args.bt_overrides_json)
        if not isinstance(extra, dict):
            raise ValueError("--bt-overrides-json must parse to a JSON object.")
        ov.update(extra)

    cmd = [args.python, "-u", "-c", CHILD_BACKTEST_CODE]
    env = {
        "DONCH_REPO_ROOT": str(REPO_ROOT),
        "DONCH_OUT_DIR": str(source_bt_dir),
        "DONCH_SIGNALS_PATH": str(signals_path),
        "DONCH_START": str(args.start),
        "DONCH_END": str(args.end),
        "DONCH_OVERRIDES_JSON": json.dumps(ov, sort_keys=True),
    }
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=bt_log, env=env, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Source backtest failed rc={rc}. See {bt_log}")

    source_trades = source_bt_dir / "trades.csv"
    if not dry_run and not source_trades.exists():
        raise FileNotFoundError(f"Source backtest finished but trades.csv missing at {source_trades}")

    if args.skip_enrich:
        out_path = release_dir / "inputs" / "trades.clean.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"[offline-release] dry-run copy: {source_trades} -> {out_path}", flush=True)
        else:
            shutil.copy2(source_trades, out_path)
        return out_path

    enrich_log = release_dir / "logs" / "01_enrich_source_trades.log"
    out_clean = release_dir / "inputs" / "trades.clean.csv"
    workdir = release_dir / ".enrich_pipeline"
    enrich_cmd: List[str] = [
        args.python,
        str((REPO_ROOT / "tools" / "enrich_trades_pipeline.py").resolve()),
        "--in",
        str(source_trades),
        "--out",
        str(out_clean),
        "--workdir",
        str(workdir),
        "--compat-dir",
        str(source_bt_dir),
        "--funding-throttle",
        str(float(args.funding_throttle)),
    ]
    if args.resume:
        enrich_cmd.append("--resume")
    if args.funding_stage == "off":
        enrich_cmd.append("--skip-funding")
    elif args.funding_stage == "optional":
        enrich_cmd.append("--funding-optional")

    rc = _run_cmd(enrich_cmd, cwd=REPO_ROOT, log_path=enrich_log, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Enrichment failed rc={rc}. See {enrich_log}")
    if not dry_run and not out_clean.exists():
        raise FileNotFoundError(f"Enrichment finished but output missing: {out_clean}")
    return out_clean


def _research_stages(
    *,
    args: argparse.Namespace,
    release_dir: Path,
    trades_clean: Path,
) -> List[Stage]:
    research_root = release_dir / "research_outputs"
    logs_dir = release_dir / "logs"
    bundle_dir = release_dir / "meta_export"

    s00_out = research_root / "00_qa" / "qa_report.html"
    s01_out = research_root / "01_targets" / "targets.parquet"
    s02_out = research_root / "02_regimes" / "regimes.parquet"
    s03_out = research_root / "03_univariate" / "rankings.csv.gz"
    s04_out = research_root / "04_models_cv" / "oof_predictions.parquet"
    s05_out = research_root / "05_calibration_ev" / "summary.json"
    s06_out = research_root / "06_tp_sl_diagnostics" / "exit_stats.csv"
    s07_out = bundle_dir / "deployment_config.json"
    s08_out = research_root / "EXEC_REPORT" / "report.html"

    stages: List[Stage] = [
        Stage(
            name="00_load_qa",
            cmd=[
                args.python,
                str((REPO_ROOT / "research" / "00_load_qa.py").resolve()),
                "--infile",
                str(trades_clean),
                "--outdir",
                str(research_root / "00_qa"),
            ],
            cwd=REPO_ROOT,
            log_path=logs_dir / "10_00_load_qa.log",
            outputs=[s00_out],
        ),
        Stage(
            name="01_make_targets",
            cmd=[
                args.python,
                str((REPO_ROOT / "research" / "01_make_targets.py").resolve()),
                "--infile",
                str(trades_clean),
                "--outdir",
                str(research_root / "01_targets"),
                "--outfile",
                "targets.parquet",
            ],
            cwd=REPO_ROOT,
            log_path=logs_dir / "11_01_make_targets.log",
            outputs=[s01_out],
        ),
        Stage(
            name="02_make_regimes",
            cmd=[
                args.python,
                str((REPO_ROOT / "research" / "02_make_regimes.py").resolve()),
                "--infile",
                str(trades_clean),
                "--outdir",
                str(research_root / "02_regimes"),
                "--outfile",
                "regimes.parquet",
            ],
            cwd=REPO_ROOT,
            log_path=logs_dir / "12_02_make_regimes.log",
            outputs=[s02_out, research_root / "02_regimes" / "regimes_report.json"],
        ),
        Stage(
            name="03_univariate_screens",
            cmd=[
                args.python,
                str((REPO_ROOT / "research" / "03_univariate_screens.py").resolve()),
                "--trades",
                str(trades_clean),
                "--targets",
                str(research_root / "01_targets" / "targets.parquet"),
                "--regimes",
                str(research_root / "02_regimes" / "regimes.parquet"),
                "--outdir",
                str(research_root / "03_univariate"),
            ],
            cwd=REPO_ROOT,
            log_path=logs_dir / "13_03_univariate.log",
            outputs=[s03_out],
        ),
        Stage(
            name="04_models_cv",
            cmd=[
                args.python,
                str((REPO_ROOT / "research" / "04_models_cv.py").resolve()),
                "--trades",
                str(trades_clean),
                "--targets",
                str(research_root / "01_targets" / "targets.parquet"),
                "--regimes",
                str(research_root / "02_regimes" / "regimes.parquet"),
                "--outdir",
                str(research_root / "04_models_cv"),
                "--target",
                str(args.target),
                "--train-scope",
                str(args.train_scope),
            ]
            + (["--include-regimes-as-features"] if args.include_regimes_as_features else [])
            + (["--live-safe-features"] if args.live_safe_features else []),
            cwd=REPO_ROOT,
            log_path=logs_dir / "14_04_models_cv.log",
            outputs=[s04_out, research_root / "04_models_cv" / "manifest.json"],
        ),
        Stage(
            name="05_calibration_ev",
            cmd=[
                args.python,
                str((REPO_ROOT / "research" / "05_calibration_ev.py").resolve()),
                "--oof",
                str(research_root / "04_models_cv" / "oof_predictions.parquet"),
                "--outdir",
                str(research_root / "05_calibration_ev"),
                "--target",
                str(args.target),
                "--fit-scope",
                str(args.fit_scope),
            ],
            cwd=REPO_ROOT,
            log_path=logs_dir / "15_05_calibration_ev.log",
            outputs=[s05_out],
        ),
    ]

    if not args.skip_step06:
        stages.append(
            Stage(
                name="06_tp_sl_diagnostics",
                cmd=[
                    args.python,
                    str((REPO_ROOT / "research" / "06_tp_sl_diagnostics.py").resolve()),
                    "--trades",
                    str(trades_clean),
                    "--outdir",
                    str(research_root / "06_tp_sl_diagnostics"),
                ],
                cwd=REPO_ROOT,
                log_path=logs_dir / "16_06_tp_sl.log",
                outputs=[s06_out],
            )
        )

    export_cmd = [
        args.python,
        str((REPO_ROOT / "research" / "07_export_deployment_artifacts.py").resolve()),
        "--cvdir",
        str(research_root / "04_models_cv"),
        "--caldir",
        str(research_root / "05_calibration_ev"),
        "--oof",
        str(research_root / "04_models_cv" / "oof_predictions.parquet"),
        "--outdir",
        str(bundle_dir),
        "--target",
        str(args.target),
        "--fit-scope",
        str(args.fit_scope),
        "--criterion",
        str(args.criterion),
    ]
    if args.decision_scope.strip():
        export_cmd.extend(["--decision-scope", str(args.decision_scope.strip())])
    if args.bundle_tar:
        export_cmd.append("--bundle-tar")

    stages.append(
        Stage(
            name="07_export_deployment_artifacts",
            cmd=export_cmd,
            cwd=REPO_ROOT,
            log_path=logs_dir / "17_07_export.log",
            outputs=[s07_out, bundle_dir / "checksums_sha256.json", bundle_dir / "feature_manifest.json"],
        )
    )
    stages.append(
        Stage(
            name="08_make_executive_report",
            cmd=[
                args.python,
                str((REPO_ROOT / "research" / "08_make_executive_report.py").resolve()),
                "--trades",
                str(trades_clean),
                "--root",
                str(research_root),
                "--outdir",
                str(research_root / "EXEC_REPORT"),
            ],
            cwd=REPO_ROOT,
            log_path=logs_dir / "18_08_exec_report.log",
            outputs=[s08_out],
        )
    )
    return stages


def _copy_regime_artifacts(release_dir: Path, dry_run: bool) -> None:
    src = release_dir / "research_outputs" / "02_regimes" / "regimes_report.json"
    dst = release_dir / "meta_export" / "regimes_report.json"
    if (not dry_run) and (not src.exists()):
        raise FileNotFoundError(f"Missing regimes_report.json at {src}")
    if dry_run:
        print(f"[offline-release] dry-run copy: {src} -> {dst}", flush=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _run_golden_export(args: argparse.Namespace, release_dir: Path, trades_clean: Path, dry_run: bool) -> Path:
    out_path = release_dir / "meta_export" / "golden_features.parquet"
    if args.skip_golden:
        return out_path

    cmd: List[str] = [
        args.python,
        str((REPO_ROOT / "export_golden_features.py").resolve()),
        "--bundle-id",
        str(args.release_id_effective),
        "--artifacts-dir",
        str(release_dir / "meta_export"),
        "--trade-store",
        str(trades_clean),
        "--regimes-report",
        str(release_dir / "research_outputs" / "02_regimes" / "regimes_report.json"),
        "--out",
        str(out_path),
        "--n-rows",
        str(int(args.golden_rows)),
        "--min-symbols",
        str(int(args.golden_min_symbols)),
        "--max-symbols",
        str(int(args.golden_max_symbols)),
    ]
    if args.golden_no_score:
        cmd.append("--no-score")

    log_path = release_dir / "logs" / "19_golden_export.log"
    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Golden export failed rc={rc}. See {log_path}")
    return out_path


def _run_bundle_smoke(args: argparse.Namespace, release_dir: Path, dry_run: bool) -> Dict[str, Any]:
    out_json = release_dir / "meta_export" / "bundle_smoke.json"
    smoke_code = r"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from winprob_loader import WinProbScorer

bundle = Path(sys.argv[1]).resolve()
golden = Path(sys.argv[2]).resolve()

sc = WinProbScorer(bundle)
out = {
    "bundle": str(bundle),
    "is_loaded": bool(sc.is_loaded),
    "raw_cols": int(len(sc.raw_cols)),
}
if golden.exists():
    df = pd.read_parquet(golden)
    if len(df) > 0 and "p_cal" in df.columns:
        raw_cols = list(sc.raw_cols)
        chk = df[raw_cols + ["p_cal"]].dropna(subset=raw_cols).head(100).reset_index(drop=True)
        errs = []
        for i in range(len(chk)):
            row = chk.loc[i, raw_cols].to_dict()
            p = float(sc.score(row))
            errs.append(abs(p - float(chk.loc[i, "p_cal"])))
        if errs:
            out["golden_rows_checked"] = int(len(errs))
            out["golden_abs_err_max"] = float(np.max(errs))
            out["golden_abs_err_p95"] = float(np.quantile(errs, 0.95))
        else:
            out["golden_rows_checked"] = 0
    else:
        out["golden_rows_checked"] = 0
else:
    out["golden_rows_checked"] = 0

print(json.dumps(out, indent=2))
"""

    cmd = [
        args.python,
        "-c",
        "import sys\n" + smoke_code,
        str((release_dir / "meta_export").resolve()),
        str((release_dir / "meta_export" / "golden_features.parquet").resolve()),
    ]
    log_path = release_dir / "logs" / "20_bundle_smoke.log"
    if dry_run:
        _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=True)
        return {"status": "dry_run"}

    rc = _run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, dry_run=False)
    if rc != 0:
        raise RuntimeError(f"Bundle smoke failed rc={rc}. See {log_path}")

    txt = log_path.read_text(encoding="utf-8", errors="replace")
    start = txt.find("{")
    end = txt.rfind("}")
    data: Dict[str, Any] = {"status": "parse_failed"}
    if start >= 0 and end > start:
        try:
            data = json.loads(txt[start : end + 1])
        except Exception:
            data = {"status": "parse_failed", "raw_log": txt[-2000:]}
    _write_json(out_json, data)
    return data


def _publish_bundle(args: argparse.Namespace, release_dir: Path, dry_run: bool) -> Optional[Path]:
    if not args.publish_dir.strip():
        return None
    dest = _to_abs(args.publish_dir.strip())
    src = release_dir / "meta_export"
    if dry_run:
        print(f"[offline-release] dry-run publish: {src} -> {dest} ({args.publish_mode})", flush=True)
        return dest

    if args.publish_mode == "copy":
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src, target_is_directory=True)
    return dest


def main() -> int:
    args = parse_args()
    rid = args.release_id.strip() or utc_id()
    setattr(args, "release_id_effective", rid)
    notifier = TelegramNotifier.from_args(args, run_label=f"offline_release:{rid}")
    print(f"[offline-release] telegram notify: {notifier.status_line()}", flush=True)
    notifier.send(
        "STARTED",
        body=(
            f"release_id={rid}\n"
            f"start={args.start} end={args.end}\n"
            f"dry_run={bool(args.dry_run)} resume={bool(args.resume)}"
        ),
    )

    release_root = _to_abs(args.release_root)
    release_dir = release_root / rid
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "logs").mkdir(parents=True, exist_ok=True)

    stage_file = release_dir / "_STAGE.txt"
    stage_file.write_text("INIT\n", encoding="utf-8")

    manifest: Dict[str, Any] = {
        "release_id": rid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "dry_run": bool(args.dry_run),
        "args": vars(args),
    }

    try:
        stage_file.write_text("SOURCE_TRADES\n", encoding="utf-8")
        trades_clean = _build_source_trades(args, release_dir, dry_run=bool(args.dry_run))
        manifest["trades_clean"] = str(trades_clean)

        stage_file.write_text("RESEARCH_PIPELINE\n", encoding="utf-8")
        stages = _research_stages(args=args, release_dir=release_dir, trades_clean=trades_clean)
        stage_status: List[Dict[str, Any]] = []
        for s in stages:
            done = bool(args.resume) and _stage_done(s.outputs)
            if done:
                print(f"[offline-release] resume: skipping {s.name}", flush=True)
                stage_status.append({"name": s.name, "status": "skipped_resume", "log": str(s.log_path)})
                continue
            rc = _run_cmd(s.cmd, cwd=s.cwd, log_path=s.log_path, dry_run=bool(args.dry_run))
            if rc != 0:
                raise RuntimeError(f"Stage {s.name} failed rc={rc}. See {s.log_path}")
            stage_status.append({"name": s.name, "status": "ok", "log": str(s.log_path)})
        manifest["stages"] = stage_status

        stage_file.write_text("POST_EXPORT_ARTIFACTS\n", encoding="utf-8")
        _copy_regime_artifacts(release_dir, dry_run=bool(args.dry_run))

        stage_file.write_text("PARITY_PACKAGE\n", encoding="utf-8")
        golden_path = _run_golden_export(args, release_dir, trades_clean, dry_run=bool(args.dry_run))
        manifest["golden_path"] = str(golden_path)

        stage_file.write_text("SMOKE_CHECK\n", encoding="utf-8")
        smoke = _run_bundle_smoke(args, release_dir, dry_run=bool(args.dry_run))
        manifest["bundle_smoke"] = smoke

        stage_file.write_text("PUBLISH\n", encoding="utf-8")
        published = _publish_bundle(args, release_dir, dry_run=bool(args.dry_run))
        if published is not None:
            manifest["published_bundle"] = str(published)

        if not args.dry_run:
            dep_cfg = release_dir / "meta_export" / "deployment_config.json"
            checksums = release_dir / "meta_export" / "checksums_sha256.json"
            thresholds = release_dir / "meta_export" / "thresholds.json"
            manifest["deployment_config"] = _read_json(dep_cfg) if dep_cfg.exists() else None
            manifest["thresholds"] = _read_json(thresholds) if thresholds.exists() else None
            manifest["checksums"] = _read_json(checksums) if checksums.exists() else None

        stage_file.write_text("DONE\n", encoding="utf-8")
        _write_json(release_dir / "release_manifest.json", manifest)
        print(f"[offline-release] DONE: {release_dir}", flush=True)
        notifier.send(
            "FINISHED",
            body=(
                f"release_id={rid}\nstatus=ok\n"
                f"release_dir={release_dir}\n"
                f"bundle={release_dir / 'meta_export'}"
            ),
        )
        return 0

    except KeyboardInterrupt:
        stage_file.write_text("INTERRUPTED\n", encoding="utf-8")
        manifest["error"] = "KeyboardInterrupt"
        _write_json(release_dir / "release_manifest.json", manifest)
        print("[offline-release] INTERRUPTED by KeyboardInterrupt", flush=True)
        notifier.send(
            "INTERRUPTED",
            body=f"release_id={rid}\nstatus=interrupted (KeyboardInterrupt)\nrelease_dir={release_dir}",
        )
        return 130
    except Exception as e:
        stage_file.write_text("FAILED\n", encoding="utf-8")
        manifest["error"] = f"{type(e).__name__}: {e}"
        _write_json(release_dir / "release_manifest.json", manifest)
        print(f"[offline-release] FAILED: {type(e).__name__}: {e}", flush=True)
        notifier.send(
            "FAILED",
            body=f"release_id={rid}\nstatus=failed\nerror={type(e).__name__}: {e}\nrelease_dir={release_dir}",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
