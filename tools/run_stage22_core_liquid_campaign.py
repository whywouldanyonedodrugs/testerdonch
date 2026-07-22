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
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceLimits, detached_service_spec, launch_detached_supervisor


class TelegramTransport:
    """Minimal secret-safe transport; tokens never enter artifacts or output."""

    def __init__(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("DONCH_TG_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("DONCH_TG_CHAT_ID")
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
        self._request("sendMessage", {"chat_id": self._chat_id, "text": "Stage 24 secure shadow preflight passed; no economic outcome or partial ranking is included."})
        return True

    def heartbeat(self, payload: Mapping[str, Any]) -> bool:
        allowed = {key: payload[key] for key in ("completed", "failed", "in_flight", "generation") if key in payload}
        self._request("sendMessage", {"chat_id": self._chat_id, "text": "Stage 22 heartbeat " + json.dumps(allowed, sort_keys=True)})
        return True

    def launch(self, payload: Mapping[str, Any]) -> bool:
        allowed = {
            key: payload[key]
            for key in ("service_identity", "status", "run_root", "workers")
            if key in payload
        }
        self._request(
            "sendMessage",
            {
                "chat_id": self._chat_id,
                "text": "Stage 24 detached shadow active " + json.dumps(allowed, sort_keys=True),
            },
        )
        return True

    def complete(self, payload: Mapping[str, Any]) -> bool:
        allowed = {
            key: payload[key]
            for key in ("service_identity", "status", "run_root", "health_release")
            if key in payload
        }
        self._request(
            "sendMessage",
            {
                "chat_id": self._chat_id,
                "text": "Stage 24 detached shadow complete " + json.dumps(allowed, sort_keys=True),
            },
        )
        return True

    def bound_stop(self, payload: Mapping[str, Any]) -> bool:
        allowed = {
            key: payload[key]
            for key in ("service_identity", "status", "reason", "resumable")
            if key in payload
        }
        self._request(
            "sendMessage",
            {
                "chat_id": self._chat_id,
                "text": "Stage 24 shadow bound stop " + json.dumps(allowed, sort_keys=True),
            },
        )
        return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the exact hash-bound Stage 22 campaign under the persistent supervisor")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "service-spec", "install-service"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--approval-request", type=Path, required=True)
        command.add_argument("--external-approval", type=Path, required=True)
        command.add_argument("--cache-manifest", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--repository-root", type=Path, default=Path.cwd())
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--telegram-env-file", type=Path, default=Path("/opt/testerdonch/.telegram.env"))
    canary = sub.add_parser("detached-canary")
    canary.add_argument("--run-root", type=Path, required=True)
    shadow = sub.add_parser("shadow-run")
    shadow.add_argument("--spec", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "shadow-run":
        # Keep the installed Stage-24 canary on the same supported launch
        # entrypoint as the eventual campaign.  ShadowAuthorization remains
        # the sole authority object and cannot authorize economic payloads.
        from tools.core_liquid_campaign.shadow_service import run_shadow_service

        print(json.dumps(run_shadow_service(args.spec), sort_keys=True))
        return 0
    if args.command == "detached-canary":
        marker = args.run_root / "DETACHED_CANARY_COMPLETE"
        def work() -> Mapping[str, Any]:
            import time
            time.sleep(1)
            return {"registered_attempt_id": "detached-canary", "aggregate": {}, "status": "complete"}
        state = LazySupervisor(
            args.run_root, ResourceLimits(max_workers=1, max_jobs_in_flight=1, minimum_free_disk_bytes=1, minimum_free_disk_fraction=0),
            real_unit_validator=lambda job, result: result.get("registered_attempt_id") == job,
        ).run(iter([("detached-canary", work)]))
        marker.write_text(json.dumps({"status": state["status"], "supervisor_pid": state["supervisor_pid"]}, sort_keys=True), encoding="utf-8")
        return 0
    authorization = ExecutionAuthorization(args.manifest, args.approval_request, args.external_approval, args.repository_root)
    manifest = authorization.require()
    expected_cache = manifest.get("primary_hashes", {}).get("cache_authority_manifest") or manifest.get("primary_hashes", {}).get("production_cache_manifest")
    if expected_cache != __import__("hashlib").sha256(args.cache_manifest.read_bytes()).hexdigest():
        raise CampaignContractError("launch cache manifest is not the exact human-approved cache authority")
    limits = ResourceLimits(
        max_workers=args.workers, max_jobs_in_flight=args.workers,
        max_rss_bytes=10 * 1024**3, max_output_bytes=24 * 1024**3,
        minimum_free_disk_bytes=8 * 1024**3, heartbeat_seconds=1800,
        graceful_stop_seconds=300, wall_time_seconds=None,
    )
    if args.command in {"service-spec", "install-service"}:
        result = detached_service_spec(args.repository_root, args.run_root, __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest(), args.workers, manifest=args.manifest, approval_request=args.approval_request, external_approval=args.external_approval, cache_manifest=args.cache_manifest, telegram_env_file=args.telegram_env_file)
        if args.command == "install-service":
            installed = launch_detached_supervisor(result, Path.home() / ".config/systemd/user")
            print(json.dumps(installed, sort_keys=True)); return 0
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
