#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from tools.core_liquid_campaign.campaign import CampaignContractError, CampaignOrchestrator
from tools.core_liquid_campaign.executor import CacheAuthority, ExecutionAuthorization
from tools.core_liquid_campaign.runtime import ResourceLimits, detached_service_spec


class TelegramTransport:
    """Minimal secret-safe transport; tokens never enter artifacts or output."""

    def __init__(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN"); chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise CampaignContractError("secure Telegram environment is not configured")
        self._token = token; self._chat_id = chat_id

    def _request(self, method: str, fields: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        data = urllib.parse.urlencode({key: str(value) for key, value in fields.items()}).encode("utf-8")
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("ok") is not True:
            raise CampaignContractError(f"Telegram {method} failed without releasing credentials")
        return payload

    def preflight(self) -> bool:
        self._request("getMe", {})
        self._request("sendMessage", {"chat_id": self._chat_id, "text": "Stage 22 secure preflight passed; no economic outcome is included."})
        return True

    def heartbeat(self, payload: Mapping[str, Any]) -> bool:
        allowed = {key: payload[key] for key in ("completed", "failed", "in_flight", "generation") if key in payload}
        self._request("sendMessage", {"chat_id": self._chat_id, "text": "Stage 22 heartbeat " + json.dumps(allowed, sort_keys=True)})
        return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the exact hash-bound Stage 22 campaign under the persistent supervisor")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "service-spec"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--approval-request", type=Path, required=True)
        command.add_argument("--external-approval", type=Path, required=True)
        command.add_argument("--cache-manifest", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--repository-root", type=Path, default=Path.cwd())
        command.add_argument("--workers", type=int, default=4)
    return result


def main() -> int:
    args = parser().parse_args()
    authorization = ExecutionAuthorization(args.manifest, args.approval_request, args.external_approval, args.repository_root)
    manifest = authorization.require()
    limits = ResourceLimits(max_workers=args.workers, max_jobs_in_flight=args.workers)
    if args.command == "service-spec":
        result = detached_service_spec(args.repository_root, args.run_root, __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest(), args.workers, manifest=args.manifest, approval_request=args.approval_request, external_approval=args.external_approval, cache_manifest=args.cache_manifest)
        print(json.dumps(result, sort_keys=True)); return 0
    telegram = TelegramTransport()
    if telegram.preflight() is not True:
        raise CampaignContractError("secure Telegram preflight failed")
    cache = CacheAuthority(args.cache_manifest, args.cache_manifest.parent)
    restarts = 0
    while True:
        orchestrator = CampaignOrchestrator(packet_root=args.manifest.parent, run_root=args.run_root, repository_root=args.repository_root, cache_authority=cache, authorization=authorization, heartbeat=telegram.heartbeat, limits=limits)
        try:
            state = orchestrator.run()
            print(json.dumps({"status": state["status"], "campaign_id": manifest["campaign_id"], "run_root": str(args.run_root), "workers": args.workers}, sort_keys=True))
            return 0
        except CampaignContractError as exc:
            restarts += 1
            if restarts > limits.maximum_supervisor_restarts:
                telegram.heartbeat({"completed": 0, "failed": 1, "in_flight": 0, "generation": "bound_stop"})
                print(json.dumps({"status": "global_resumable_bound_stop", "reason": str(exc), "restarts": restarts}, sort_keys=True))
                return 75


if __name__ == "__main__":
    raise SystemExit(main())
