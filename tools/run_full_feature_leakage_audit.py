#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from feature_registry import ACTIVE_FEATURE_REGISTRY  # noqa: E402
from tools.ci_check_leakage_guards import run_checks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full feature leakage audit bundle.")
    p.add_argument("--out-dir", default="results/leakage_audit/full_feature_audit")
    p.add_argument("--start", default="2025-09-01")
    p.add_argument("--end", default="2026-03-05")
    p.add_argument("--month", default="2025-11")
    p.add_argument("--sample-n", type=int, default=5000)
    return p.parse_args()


def main() -> int:
    a = _parse_args()
    out_dir = (REPO / a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(ACTIVE_FEATURE_REGISTRY).sort_values(["test_type", "family", "name"]).to_csv(
        out_dir / "feature_registry_coverage.csv",
        index=False,
    )

    static_report = run_checks(REPO)
    (out_dir / "static_guard_report.json").write_text(
        json.dumps(static_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dyn_report = out_dir / "dynamic_contract_report.md"
    dyn_traces = out_dir / "dynamic_traces"
    cmd = [
        sys.executable,
        str(REPO / "tools" / "run_leakage_contract_test.py"),
        "--out-report",
        str(dyn_report),
        "--out-traces-dir",
        str(dyn_traces),
        "--start",
        a.start,
        "--end",
        a.end,
        "--month",
        a.month,
        "--sample-n",
        str(a.sample_n),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    (out_dir / "dynamic_contract_stdout.log").write_text(
        (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""),
        encoding="utf-8",
    )

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "unit_tests.test_jt008_leakage_guards",
        "unit_tests.test_jt008_htf_asof_semantics",
        "unit_tests.test_feature_leakage_contracts",
    ]
    unit_proc = subprocess.run(unit_cmd, cwd=str(REPO), capture_output=True, text=True)
    (out_dir / "unit_contract_stdout.log").write_text(
        (unit_proc.stdout or "") + "\n--- STDERR ---\n" + (unit_proc.stderr or ""),
        encoding="utf-8",
    )

    verdict = {
        "static_status": static_report.get("status"),
        "dynamic_contract_rc": int(proc.returncode),
        "unit_contract_rc": int(unit_proc.returncode),
    }
    (out_dir / "summary.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ok = (
        static_report.get("status") == "ok"
        and int(proc.returncode) == 0
        and int(unit_proc.returncode) == 0
    )
    print(str(out_dir))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
