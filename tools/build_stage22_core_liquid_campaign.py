#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.core_liquid_campaign.compiler import SourcePaths
from tools.core_liquid_campaign.packet import build_candidate, finalize_packet


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build or finalize the outcome-free Stage 22 campaign packet")
    sub = result.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    for command in (build,):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--repository-root", type=Path, required=True)
        command.add_argument("--implementation-commit", required=True)
        command.add_argument("--stage21-v1-zip", type=Path, required=True)
        command.add_argument("--stage21-v5-zip", type=Path, required=True)
        command.add_argument("--strategy-registry", type=Path, required=True)
        command.add_argument("--control-registry", type=Path, required=True)
        command.add_argument("--v3-contract", type=Path, required=True)
        command.add_argument("--v4-addendum", type=Path, required=True)
        command.add_argument("--v5-closure", type=Path, required=True)
        command.add_argument("--stage22-task", type=Path, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--repository-root", type=Path, required=True)
    finalize.add_argument("--implementation-commit", required=True)
    finalize.add_argument("--review", type=Path, required=True)
    finalize.add_argument("--inherited-manifest", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "build":
        paths = SourcePaths(
            stage21_v1_zip=args.stage21_v1_zip,
            stage21_v5_zip=args.stage21_v5_zip,
            strategy_registry=args.strategy_registry,
            control_registry=args.control_registry,
            v3_contract=args.v3_contract,
            v4_addendum=args.v4_addendum,
            v5_closure=args.v5_closure,
            stage22_task=args.stage22_task,
        )
        result = build_candidate(paths, args.output_root, args.repository_root, args.implementation_commit)
    else:
        result = finalize_packet(args.output_root, args.repository_root, args.implementation_commit, args.review, args.inherited_manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
