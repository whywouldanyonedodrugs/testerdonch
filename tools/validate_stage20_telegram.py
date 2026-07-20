#!/usr/bin/env python3
"""Secret-safe real Telegram preflight for the approved Stage 20 campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qlmg_stage20_campaign import atomic_json, utc_now
from tools.qlmg_stage20_launch_gates import validate_gate, validate_source_manifest, write_gate
from tools.telegram_notify import TelegramNotifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--deterministic-replay", type=Path, required=True)
    parser.add_argument("--synthetic-canary", type=Path, required=True)
    parser.add_argument("--preoutcome-review", type=Path, required=True)
    parser.add_argument("--tg-bot-token", default="")
    parser.add_argument("--tg-chat-id", default="")
    parser.add_argument("--tg-auto-chat", action="store_true", default=True)
    args = parser.parse_args()
    validate_source_manifest(args.source_manifest)
    validate_gate(args.deterministic_replay, "deterministic_event_replay")
    validate_gate(args.synthetic_canary, "synthetic_supervisor_canary")
    validate_gate(args.preoutcome_review, "independent_preoutcome_review")
    notifier = TelegramNotifier.from_args(args, run_label="stage20-preoutcome-validation")
    checks = {
        "dry_run_preflight": notifier.enabled,
        "synthetic_heartbeat": False,
        "synthetic_stop": False,
    }
    if notifier.enabled:
        checks["synthetic_heartbeat"] = notifier.send(
            "SYNTHETIC HEARTBEAT", "pre-outcome notifier validation; no economic results"
        )
        checks["synthetic_stop"] = notifier.send(
            "SYNTHETIC STOP", "pre-outcome notifier validation; no campaign stop occurred"
        )
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "validated_at_utc": utc_now(), "checks": checks,
        "secrets_printed_or_archived": False, "economic_outputs_in_messages": False,
    }
    atomic_json(args.evidence_output, result)
    if result["status"] == "pass":
        gate = write_gate(
            args.output, "telegram_preflight",
            [args.approval, args.source_manifest,
             args.event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json",
             args.deterministic_replay, args.synthetic_canary, args.preoutcome_review,
             args.evidence_output], result,
        )
        result["binding_sha256"] = gate["binding_sha256"]
    print(json.dumps({"status": result["status"], "checks": checks}, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
