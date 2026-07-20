#!/usr/bin/env python3
"""Materialize cryptographically bound Stage 20 review and launch gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qlmg_stage20_campaign import Stage20Error
from tools.qlmg_stage20_launch_gates import (
    build_source_manifest, validate_gate, validate_source_manifest, write_gate,
)


CRITERIA = [
    "cryptographic_gate_binding", "atomic_final_launch_revalidation",
    "bounded_lazy_scheduling", "health_release",
    "empty_inner_fold_preservation", "synthetic_end_to_end_canary",
    "supported_direct_invocation",
]


def source(args: argparse.Namespace) -> int:
    result = build_source_manifest(args.output, args.file)
    print(json.dumps({"status": result["status"],
                      "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    return 0


def review(args: argparse.Namespace) -> int:
    validate_source_manifest(args.source_manifest)
    validate_gate(args.deterministic_replay, "deterministic_event_replay")
    validate_gate(args.synthetic_canary, "synthetic_supervisor_canary")
    evidence = json.loads(args.review_evidence.read_text())
    criteria = evidence.get("acceptance_criteria", {})
    if (evidence.get("status") != "pass" or set(criteria) != set(CRITERIA)
            or any(criteria[name] != "pass" for name in CRITERIA)
            or evidence.get("blocking_findings") not in ([], 0)):
        raise Stage20Error("independent review did not pass all seven fixed criteria")
    result = write_gate(
        args.output, "independent_preoutcome_review",
        [args.approval, args.source_manifest,
         args.event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json",
         args.deterministic_replay, args.synthetic_canary, args.review_evidence],
        {"acceptance_criteria": criteria, "blocking_findings": 0,
         "reviewer_independent": True},
    )
    print(json.dumps({"status": result["status"],
                      "binding_sha256": result["binding_sha256"]}, sort_keys=True))
    return 0


def final(args: argparse.Namespace) -> int:
    validate_source_manifest(args.source_manifest)
    validate_gate(args.deterministic_replay, "deterministic_event_replay")
    validate_gate(args.synthetic_canary, "synthetic_supervisor_canary")
    validate_gate(args.preoutcome_review, "independent_preoutcome_review")
    validate_gate(args.telegram_validation, "telegram_preflight")
    limits = {"max_workers": 4, "max_wall_seconds": 14400,
              "max_rss_bytes": 5 * 1024**3, "max_output_bytes": 5 * 1024**3,
              "heartbeat_seconds": 1800}
    result = write_gate(
        args.output, "final_launch_authority",
        [args.approval, args.source_manifest,
         args.event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json",
         args.deterministic_replay, args.synthetic_canary,
         args.preoutcome_review, args.telegram_validation],
        {"runtime_limits": limits, "approved_cells": 186,
         "protected_rows_opened": 0, "Capitalcom_payload_opened": False,
         "controls_executed": False},
    )
    print(json.dumps({"status": result["status"],
                      "binding_sha256": result["binding_sha256"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("source-manifest")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--file", type=Path, action="append", required=True)
    freeze.set_defaults(func=source)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--approval", type=Path, required=True)
    common.add_argument("--source-manifest", type=Path, required=True)
    common.add_argument("--event-root", type=Path, required=True)
    common.add_argument("--deterministic-replay", type=Path, required=True)
    common.add_argument("--synthetic-canary", type=Path, required=True)
    reviewed = sub.add_parser("review-gate", parents=[common])
    reviewed.add_argument("--review-evidence", type=Path, required=True)
    reviewed.add_argument("--output", type=Path, required=True)
    reviewed.set_defaults(func=review)
    launch = sub.add_parser("final-gate", parents=[common])
    launch.add_argument("--preoutcome-review", type=Path, required=True)
    launch.add_argument("--telegram-validation", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.set_defaults(func=final)
    return result


if __name__ == "__main__":
    try:
        namespace = parser().parse_args()
        raise SystemExit(namespace.func(namespace))
    except Stage20Error as exc:
        print(json.dumps({"status": "blocked_preoutcome", "reason": str(exc)}, sort_keys=True),
              file=sys.stderr)
        raise SystemExit(2)
