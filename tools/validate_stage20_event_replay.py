#!/usr/bin/env python3
"""Hash-only deterministic replay validator for Stage 20 pre-outcome event tapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qlmg_stage20_campaign import file_sha256
from tools.qlmg_stage20_launch_gates import validate_source_manifest, write_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    validate_source_manifest(args.source_manifest)
    left_manifest = args.primary / "PREOUTCOME_EVENT_TAPE_MANIFEST.json"
    right_manifest = args.replay / "PREOUTCOME_EVENT_TAPE_MANIFEST.json"
    left = json.loads(left_manifest.read_text())
    right = json.loads(right_manifest.read_text())
    left_files = {row["symbol"]: row for row in left["files"]}
    right_files = {row["symbol"]: row for row in right["files"]}
    symbols_match = set(left_files) == set(right_files) and len(left_files) == 187 and None not in left_files
    files_match = symbols_match and all(
        left_files[symbol]["rows"] == right_files[symbol]["rows"]
        and left_files[symbol]["sha256"] == right_files[symbol]["sha256"]
        and file_sha256(Path(left_files[symbol]["path"])) == left_files[symbol]["sha256"]
        and file_sha256(Path(right_files[symbol]["path"])) == right_files[symbol]["sha256"]
        for symbol in left_files
    )
    thresholds_match = file_sha256(args.primary / "FOLD_LOCAL_THRESHOLDS.json") == file_sha256(
        args.replay / "FOLD_LOCAL_THRESHOLDS.json"
    )
    skips_match = file_sha256(args.primary / "MECHANICAL_CELL_SKIPS.json") == file_sha256(
        args.replay / "MECHANICAL_CELL_SKIPS.json"
    )
    firewall = all(
        manifest.get("protected_rows_opened") == 0
        and manifest.get("economic_outcome_reader_opened") is False
        and manifest.get("Capitalcom_payload_opened") is False
        for manifest in (left, right)
    )
    passed = files_match and thresholds_match and skips_match and firewall
    assertions = {
        "native_symbol_partitions": len(left_files), "symbol_sets_match": symbols_match,
        "event_file_rows_and_sha256_match": files_match,
        "fold_local_threshold_file_sha256_match": thresholds_match,
        "mechanical_skip_file_sha256_match": skips_match,
        "protected_rows_opened": 0 if firewall else "failed",
        "economic_outcome_reader_opened": False if firewall else "failed",
        "Capitalcom_payload_opened": False if firewall else "failed",
    }
    if not passed:
        print(json.dumps({"status": "fail", **assertions}, sort_keys=True))
        return 2
    result = write_gate(
        args.output, "deterministic_event_replay",
        [args.approval, args.source_manifest, left_manifest, right_manifest,
         args.primary / "FOLD_LOCAL_THRESHOLDS.json", args.replay / "FOLD_LOCAL_THRESHOLDS.json",
         args.primary / "MECHANICAL_CELL_SKIPS.json", args.replay / "MECHANICAL_CELL_SKIPS.json"],
        assertions,
    )
    print(json.dumps({"status": result["status"], "gate_id": result["gate_id"],
                      "binding_sha256": result["binding_sha256"]}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
