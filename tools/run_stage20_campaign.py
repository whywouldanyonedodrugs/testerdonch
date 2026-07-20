#!/usr/bin/env python3
"""Launch-control and campaign entrypoint for approved Stage 20 work."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qlmg_stage20_campaign import (
    CAMPAIGN_ID, FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256, STAGE19_REL,
    Stage20Error, atomic_json, authority_audit, build_preoutcome_event_tapes, utc_now,
)
from tools.qlmg_stage19_funding import Stage19FundingEngine
from tools.validate_stage19_campaign_packet import validate as validate_stage19


def preflight(args: argparse.Namespace) -> int:
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "preflight").mkdir(exist_ok=True)
    audit = authority_audit(ROOT, args.approval)
    audit["actual_starting_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if audit["actual_starting_commit"] != "245b375b00167f1b4a81f6a4449e7de1d1db83a2":
        raise Stage20Error("repository commit drift")
    readiness = validate_stage19(ROOT / STAGE19_REL)
    if readiness.get("synthetic_canary") != "pass" or readiness.get("registered_cells") != 186:
        raise Stage20Error("Stage 19 launch validator/canary failed")
    funding_contract = json.loads((ROOT / STAGE19_REL / "FUNDING_COST_AND_COVERAGE_CONTRACT.json").read_text())
    funding = Stage19FundingEngine(
        FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256,
        ROOT / STAGE19_REL / "FUNDING_GAP_ALLOWANCE_TABLE.csv",
        funding_contract["gap_allowance_table_sha256"],
    )
    if len(funding.allowances) != 187:
        raise Stage20Error("Stage 19 funding runtime coverage mismatch")
    atomic_json(args.run_root / "PREOUTCOME_AUTHORITY_AND_HASH_AUDIT.json", audit)
    atomic_json(args.run_root / "preflight" / "STAGE19_LAUNCH_READINESS.json", readiness)
    shutil.copyfile(args.approval, args.run_root / "HUMAN_APPROVAL.json")
    launch = {
        "status": "preoutcome_pending_review_and_telegram",
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": utc_now(),
        "starting_commit": audit["actual_starting_commit"],
        "approval_sha256": audit["approval_sha256"],
        "packet_sha256": audit["approval_packet_sha256"],
        "manifest_sha256": audit["campaign_manifest_sha256"],
        "registered_cells": 186,
        "workers_maximum": 4,
        "wall_seconds_maximum": 14400,
        "aggregate_rss_bytes_maximum": 5368709120,
        "output_bytes_maximum": 5368709120,
        "protected_rows_opened": 0,
        "Capitalcom_payload_opened": False,
        "economic_outcome_reader_opened": False,
    }
    atomic_json(args.run_root / "CAMPAIGN_LAUNCH_MANIFEST.json", launch)
    atomic_json(args.run_root / "CAMPAIGN_STATE.json", {**launch, "generation": 1})
    print(json.dumps({"status": "preflight_materialized", "run_root": str(args.run_root),
                      "funding_runtime": "pass", "funding_symbols": len(funding.allowances)}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--approval", type=Path, required=True)
    pre.add_argument("--run-root", type=Path, required=True)
    pre.set_defaults(func=preflight)
    events = sub.add_parser("build-events")
    events.add_argument("--output", type=Path, required=True)
    events.add_argument("--thresholds-source", type=Path)
    events.set_defaults(func=lambda args: (print(json.dumps(build_preoutcome_event_tapes(ROOT, args.output, args.thresholds_source), sort_keys=True)), 0)[1])
    return result


if __name__ == "__main__":
    try:
        ns = parser().parse_args()
        raise SystemExit(ns.func(ns))
    except Stage20Error as exc:
        print(json.dumps({"status": "blocked_preoutcome", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
