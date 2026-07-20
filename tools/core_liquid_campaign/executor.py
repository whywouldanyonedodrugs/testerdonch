from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .accounting import aggregate_parent_legs, simulate_leg
from .canonical import canonical_hash, sha256_file
from .engine_types import FamilyInput, KRAKEN_PLATFORM, PROTECTED_START, RANKABLE_START
from .family_engines import a1_compression, a2_context, a3_starter_retest, a4_tsmom, kda02b_adjudication
from .family_engines.common import EngineInputError, require_utc
from .schema import CAMPAIGN_ID, economic_address, normalize_config
from .selection import EventObservation, aggregate_streaming


class AuthorizationError(PermissionError):
    pass


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ExecutionAuthorization:
    """Exact file-backed launch authority; no caller-selected expected hashes."""

    manifest_path: Path
    approval_request_path: Path
    external_approval_path: Path
    repository_root: Path

    def require(self) -> dict[str, Any]:
        for path in (self.manifest_path, self.approval_request_path, self.external_approval_path):
            if not path.is_file():
                raise AuthorizationError(f"required authority artifact is absent: {path}")
        manifest_hash = sha256_file(self.manifest_path)
        request_hash = sha256_file(self.approval_request_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        request = json.loads(self.approval_request_path.read_text(encoding="utf-8"))
        approval = json.loads(self.external_approval_path.read_text(encoding="utf-8"))
        if manifest.get("campaign_id") != CAMPAIGN_ID or request.get("campaign_id") != CAMPAIGN_ID or approval.get("campaign_id") != CAMPAIGN_ID:
            raise AuthorizationError("campaign identity mismatch across launch authority")
        if request.get("final_campaign_manifest_sha256") != manifest_hash:
            raise AuthorizationError("approval request does not bind the physical campaign manifest")
        if approval.get("final_campaign_manifest_sha256") != manifest_hash or approval.get("final_human_approval_request_sha256") != request_hash:
            raise AuthorizationError("external approval does not bind exact manifest/request bytes")
        if approval.get("approved") is not True or approval.get("authorization") != "launch_exact_frozen_stage22_campaign":
            raise AuthorizationError("exact external economic approval is absent")
        expected_commit = manifest.get("repository", {}).get("implementation_commit")
        if approval.get("repository_implementation_commit") != expected_commit or not _is_git_commit(expected_commit):
            raise AuthorizationError("external approval repository binding mismatch")
        actual_commit = subprocess.run(
            ["git", "-C", str(self.repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_commit != expected_commit:
            ancestry = subprocess.run(
                ["git", "-C", str(self.repository_root), "merge-base", "--is-ancestor", str(expected_commit), actual_commit],
                capture_output=True,
            )
            if ancestry.returncode != 0:
                raise AuthorizationError("live repository is not a descendant of the approved implementation commit")
        code_inventory = self.manifest_path.parent / "CODE_HASH_INVENTORY.json"
        if sha256_file(code_inventory) != manifest.get("primary_hashes", {}).get("code_inventory"):
            raise AuthorizationError("live code inventory bytes differ from manifest")
        inventory = json.loads(code_inventory.read_text(encoding="utf-8"))
        for record in inventory.get("files", []):
            path = self.repository_root / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise AuthorizationError(f"approved implementation file drift: {record['path']}")
        return manifest


@dataclass(frozen=True)
class CacheAuthority:
    """Physical cache manifest plus its exact source/universe/funding bindings."""

    manifest_path: Path
    cache_root: Path

    def validate(self, campaign_manifest: Mapping[str, Any], frames: Sequence[FamilyInput]) -> dict[str, Any]:
        expected = campaign_manifest.get("execution_input_authority", {})
        if not self.manifest_path.is_file():
            raise AuthorizationError("physical cache manifest is absent")
        cache = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if cache.get("schema") != expected.get("cache_manifest_contract", {}).get("schema"):
            raise AuthorizationError("cache manifest schema differs from the approved contract")
        if cache.get("platform") != KRAKEN_PLATFORM or cache.get("rankable_interval") != "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)":
            raise AuthorizationError("cache platform or protected boundary mismatch")
        required = ("source_manifest_sha256", "pit_universe_sha256", "funding_manifest_sha256", "cache_contract_sha256")
        for key in required:
            value = cache.get(key)
            if not _is_sha256(value) or value != expected.get(key):
                raise AuthorizationError(f"cache dependency mismatch: {key}")
        records = cache.get("artifacts")
        if not isinstance(records, list) or not records:
            raise AuthorizationError("cache manifest has no physical artifacts")
        if cache.get("artifact_inventory_sha256") != canonical_hash(records):
            raise AuthorizationError("cache artifact inventory hash mismatch")
        by_path: dict[str, Mapping[str, Any]] = {}
        for record in records:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise AuthorizationError("unsafe cache artifact path")
            path = self.cache_root / relative
            if str(relative) in by_path or not _is_sha256(record.get("sha256")) or not _is_sha256(record.get("frame_content_sha256")):
                raise AuthorizationError("duplicate or incomplete cache artifact record")
            if not path.is_file() or path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
                raise AuthorizationError(f"cache artifact mismatch: {relative}")
            by_path[str(relative)] = record
        if not frames:
            raise AuthorizationError("no raw family input was supplied")
        for frame in frames:
            frame.validate()
            bindings = frame.metadata.get("cache_bindings")
            if not isinstance(bindings, Mapping) or any(bindings.get(key) != cache[key] for key in required):
                raise AuthorizationError("raw frame is not bound to the approved cache dependencies")
            artifact_path = frame.metadata.get("cache_artifact_path")
            record = by_path.get(str(artifact_path))
            if record is None or frame.content_sha256() != record["frame_content_sha256"]:
                raise AuthorizationError("raw frame content is not bound to a verified cache artifact")
        return cache


def validate_registered_attempt(row: Mapping[str, Any]) -> None:
    if row.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("registered attempt campaign mismatch")
    if row.get("execution_disposition") not in {"execute_once", "execute_if_parent_available"}:
        raise ValueError("duplicate, superseded, or unavailable attempt cannot execute")
    family = str(row["family_id"])
    normalized = normalize_config(family, row["config"])
    if normalized != row["config"]:
        raise ValueError("registered attempt config is not canonical")
    _, address = economic_address(family, normalized)
    if address != row["canonical_economic_address_sha256"]:
        raise ValueError("registered attempt economic-address mismatch")
    if row.get("execution_disposition") == "execute_once" and row.get("duplicate_of_executable_attempt_id") is not None:
        raise ValueError("execute-once row carries a duplicate parent")


def _observation(frame: FamilyInput, event: Mapping[str, Any], base_net: float, stress_net: float, exit_ts: datetime, *, exposure: float, component_metrics: Sequence[tuple[str, float]] = ()) -> EventObservation:
    entry_ts = require_utc(event["entry_ts"])
    decision_ts = require_utc(event["decision_ts"])
    exit_ts = require_utc(exit_ts)
    if not (RANKABLE_START <= decision_ts <= entry_ts < exit_ts < PROTECTED_START):
        raise AuthorizationError("generated event crosses the rankable/protected timestamp firewall")
    day = entry_ts.date().isoformat()
    return EventObservation(
        event_id=str(event["event_id"]),
        symbol=frame.symbol,
        entry_day=day,
        month=day[:7],
        year=entry_ts.year,
        base_net_bps=float(base_net),
        stress_net_bps=float(stress_net),
        market_day=day,
        decision_ts=decision_ts,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        eligible_days=int(frame.metadata.get("eligible_days", 1)),
        threshold_eligible=True,
        component_metrics=tuple(component_metrics),
        holding_seconds_weighted=exposure * (exit_ts - entry_ts).total_seconds(),
        eligible_symbol_seconds=float(frame.metadata.get("eligible_symbol_seconds", max(1.0, (exit_ts - entry_ts).total_seconds()))),
    )


def _simulate_event(frame: FamilyInput, family: str, config: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[EventObservation | None, dict[str, Any]]:
    bars = tuple(bar.trade_bar() for bar in frame.five_minute_bars)
    entry_index = int(event["entry_index"])
    context_multiplier = float(event.get("context_multiplier", 1.0))
    if not 0 <= context_multiplier <= 1:
        raise EngineInputError("context multiplier is outside [0,1]")
    fixed = config.get("fixed_target_R")
    fixed_target = None if fixed in (None, "none") else float(fixed)
    exit_name = str(event.get("exit", config.get("exit", "time_1d")))
    cost = float(event.get("cost_bps", 14.0))
    alignment = str(event.get("funding_alignment", "start_exclusive_end_inclusive"))
    funding = () if event.get("funding_zero") else frame.funding
    gap = float(frame.metadata.get("funding_gap_allowance_bps", 0.0))
    base_exposure = float(event.get("exposure", 1.0)) * context_multiplier
    if family == "A3_STARTER_RETEST_V3":
        starter_fraction = float(event["starter_fraction"])
        starter = simulate_leg(
            bars,
            entry_index=entry_index,
            side=int(event["side"]),
            exit_name=exit_name,
            atr=event.get("atr"),
            fixed_target_r=fixed_target,
            structural_level=event.get("level") if exit_name == "breakout_failure" else None,
            signal_reversal_close_ts=None,
            funding=funding,
            cost_bps=cost,
            funding_alignment=alignment,
            evaluation_start=require_utc(frame.metadata["evaluation_start"]),
            evaluation_end_exclusive=require_utc(frame.metadata["evaluation_end_exclusive"]),
            exposure=context_multiplier,
            gap_allowance_bps=gap,
        )
        if starter.status != "complete" or starter.exit_ts is None:
            return None, {"event_id": event["event_id"], "status": starter.status, "starter": starter.to_dict()}
        add = None
        add_fraction = float(event["add_fraction"])
        retest = None
        if add_fraction > 0:
            forced_lag = event.get("control_add_lag_bars")
            if forced_lag == "NO_ADD":
                retest = a3_starter_retest.RetestResult("unavailable_permuted_no_add", None, None, None)
            elif forced_lag is not None:
                candidate_index = entry_index + int(forced_lag)
                window_seconds = {"6h": 21600, "1d": 86400, "3d": 259200}[str(event["retest_window"])]
                valid = candidate_index < len(bars) and require_utc(bars[candidate_index].open_ts) < starter.exit_ts and (require_utc(bars[candidate_index].open_ts) - starter.entry_ts).total_seconds() < window_seconds
                retest = a3_starter_retest.RetestResult("complete_permuted" if valid else "unavailable_permuted_lag", None, None, candidate_index if valid else None)
            else:
                retest = a3_starter_retest.run_retest_state_machine(
                    frame,
                    direction="long" if int(event["side"]) == 1 else "short",
                    level=float(event["level"]),
                    atr=float(event["atr"]),
                    depth=float(event["retest_depth"]),
                    starter_entry_index=entry_index,
                    starter_exit_ts=starter.exit_ts,
                    window=str(event["retest_window"]),
                )
            if retest.add_entry_index is not None:
                add = simulate_leg(
                    bars,
                    entry_index=retest.add_entry_index,
                    side=int(event["side"]),
                    exit_name=exit_name,
                    atr=event.get("atr"),
                    fixed_target_r=fixed_target,
                    structural_level=event.get("level") if exit_name == "breakout_failure" else None,
                    signal_reversal_close_ts=None,
                    funding=funding,
                    cost_bps=cost,
                    funding_alignment=alignment,
                    evaluation_start=require_utc(frame.metadata["evaluation_start"]),
                    evaluation_end_exclusive=require_utc(frame.metadata["evaluation_end_exclusive"]),
                    exposure=context_multiplier,
                    gap_allowance_bps=gap,
                )
        parent = aggregate_parent_legs(starter, starter_fraction, add, add_fraction)
        stress_starter = simulate_leg(
            bars,
            entry_index=entry_index,
            side=int(event["side"]),
            exit_name=exit_name,
            atr=event.get("atr"),
            fixed_target_r=fixed_target,
            structural_level=event.get("level") if exit_name == "breakout_failure" else None,
            signal_reversal_close_ts=None,
            funding=funding,
            cost_bps=32.0,
            funding_alignment=alignment,
            evaluation_start=require_utc(frame.metadata["evaluation_start"]),
            evaluation_end_exclusive=require_utc(frame.metadata["evaluation_end_exclusive"]),
            exposure=context_multiplier,
            gap_allowance_bps=gap,
        )
        stress_add = None
        if add is not None and retest is not None and retest.add_entry_index is not None:
            stress_add = simulate_leg(
                bars, entry_index=retest.add_entry_index, side=int(event["side"]), exit_name=exit_name, atr=event.get("atr"),
                fixed_target_r=fixed_target, structural_level=event.get("level") if exit_name == "breakout_failure" else None,
                signal_reversal_close_ts=None, funding=funding, cost_bps=32.0, funding_alignment=alignment,
                evaluation_start=require_utc(frame.metadata["evaluation_start"]), evaluation_end_exclusive=require_utc(frame.metadata["evaluation_end_exclusive"]),
                exposure=context_multiplier, gap_allowance_bps=gap,
            )
        stress_parent = aggregate_parent_legs(stress_starter, starter_fraction, stress_add, add_fraction)
        enriched = {**event, "entry_ts": starter.entry_ts}
        observation = _observation(frame, enriched, parent["net_bps"], stress_parent["net_bps"], parent["parent_exit_ts"], exposure=starter_fraction * context_multiplier + (add_fraction * context_multiplier if add is not None else 0.0), component_metrics=(("add_complete", 1.0 if add is not None else 0.0),))
        return observation, {"event_id": event["event_id"], "status": "complete", "starter": starter.to_dict(), "add": add.to_dict() if add else None, "retest": retest.__dict__ if retest else None, "parent": parent, "stress_parent": stress_parent}
    base = simulate_leg(
        bars,
        entry_index=entry_index,
        side=int(event["side"]),
        exit_name=exit_name,
        atr=event.get("atr"),
        fixed_target_r=fixed_target,
        structural_level=event.get("structural_level"),
        signal_reversal_close_ts=set(event.get("signal_reversal_close_ts", ())),
        funding=funding,
        cost_bps=cost,
        funding_alignment=alignment,
        evaluation_start=require_utc(frame.metadata["evaluation_start"]),
        evaluation_end_exclusive=require_utc(frame.metadata["evaluation_end_exclusive"]),
        exposure=base_exposure,
        gap_allowance_bps=gap,
    )
    if base.status != "complete" or base.exit_ts is None or base.net_bps is None:
        return None, {"event_id": event["event_id"], "status": base.status, "base": base.to_dict()}
    stress = simulate_leg(
        bars, entry_index=entry_index, side=int(event["side"]), exit_name=exit_name, atr=event.get("atr"), fixed_target_r=fixed_target,
        structural_level=event.get("structural_level"), signal_reversal_close_ts=set(event.get("signal_reversal_close_ts", ())), funding=funding,
        cost_bps=32.0, funding_alignment=alignment, evaluation_start=require_utc(frame.metadata["evaluation_start"]),
        evaluation_end_exclusive=require_utc(frame.metadata["evaluation_end_exclusive"]), exposure=base_exposure, gap_allowance_bps=gap,
    )
    enriched = {**event, "entry_ts": base.entry_ts}
    observation = _observation(frame, enriched, base.net_bps, float(stress.net_bps), base.exit_ts, exposure=base_exposure, component_metrics=(("favorable_funding_report_only", float(base.favorable_funding_bps)),))
    return observation, {"event_id": event["event_id"], "status": "complete", "base": base.to_dict(), "stress": stress.to_dict()}


def _generate_events(family: str, frame: FamilyInput, config: Mapping[str, Any], control_id: str | None) -> list[dict[str, Any]]:
    if family == "A4_TSMOM_V7":
        return a4_tsmom.evaluate(frame, config, control_id=control_id)
    if family == "A1_COMPRESSION_V2":
        return a1_compression.evaluate(frame, config, control_id=control_id)
    if family == "A3_STARTER_RETEST_V3":
        return a3_starter_retest.evaluate(frame, config, control_id=control_id)
    if family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
        return kda02b_adjudication.evaluate(frame, config, control_id=control_id)
    raise EngineInputError(f"family requires parent-bound dispatcher: {family}")


def dispatch_registered_attempt(
    row: Mapping[str, Any],
    frames: Sequence[FamilyInput],
    *,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    parent_binding: Mapping[str, Any] | None = None,
    parent_frames: Sequence[FamilyInput] | None = None,
    control_id: str | None = None,
) -> dict[str, Any]:
    """Code-owned path from raw/cache frames through engine, accounting and aggregate."""
    validate_registered_attempt(row)
    family = str(row["family_id"])
    config = row["config"]
    observations: list[EventObservation] = []
    ledger: list[dict[str, Any]] = []
    if family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        if parent_binding is None or parent_frames is None:
            return {"status": "unavailable_no_parent", "observations": [], "ledger": [], "aggregate": aggregate_streaming(())}
        expected_template = row.get("parent_binding_template_id")
        if parent_binding.get("parent_binding_template_id") != expected_template:
            raise AuthorizationError("A2 runtime parent binding does not match registered template")
        parent_id = str(parent_binding.get("parent_executable_attempt_id"))
        parent = registry_by_id.get(parent_id)
        if parent is None or parent.get("family_id") != config["parent_family"]:
            raise AuthorizationError("A2 exact parent is absent or from the wrong family")
        if parent_binding.get("parent_only_counterpart_id") != row.get("parent_only_counterpart_id") or parent_binding.get("overlay_counterpart_id") != row.get("overlay_counterpart_id"):
            raise AuthorizationError("A2 atomic counterpart binding mismatch")
        parent_result = dispatch_registered_attempt(parent, parent_frames, registry_by_id=registry_by_id)
        parent_observations = {item.event_id: item for item in parent_result["observations"]}
        parent_ledger = {item["event_id"]: item for item in parent_result["ledger"] if item.get("status") == "complete"}
        frame_by_symbol_and_decision = {(frame.symbol, require_utc(frame.decision_ts)): frame for frame in frames}
        for event_id, parent_observation in sorted(parent_observations.items()):
            frame = frame_by_symbol_and_decision.get((parent_observation.symbol, require_utc(parent_observation.decision_ts)))
            source_ledger = parent_ledger.get(event_id)
            if frame is None or source_ledger is None:
                continue
            base_event = {
                "event_id": event_id,
                "side": source_ledger["base"]["side"] if "base" in source_ledger else source_ledger["starter"]["side"],
                "parent_only_counterpart_id": row["parent_only_counterpart_id"],
                "overlay_counterpart_id": row["overlay_counterpart_id"],
            }
            overlay = a2_context.evaluate_overlay(frame, config, base_event, control_id=control_id)
            multiplier = float(overlay["context_multiplier"])
            observation = EventObservation(
                **{**parent_observation.__dict__, "base_net_bps": parent_observation.base_net_bps * multiplier, "stress_net_bps": parent_observation.stress_net_bps * multiplier, "holding_seconds_weighted": parent_observation.holding_seconds_weighted * multiplier,
                   "component_metrics": tuple(overlay["context_components"])},
            )
            observations.append(observation)
            ledger.append({"event_id": event_id, "status": "complete", "parent_executable_attempt_id": parent_id, "parent_only_counterpart_id": row["parent_only_counterpart_id"], "overlay_counterpart_id": row["overlay_counterpart_id"], "context_multiplier": multiplier, "parent_ledger_sha256": canonical_hash(source_ledger)})
    else:
        generated_work: list[tuple[FamilyInput, dict[str, Any]]] = []
        for frame in frames:
            frame.validate()
            generated_work.extend((frame, event) for event in _generate_events(family, frame, config, control_id))
        if family == "A4_TSMOM_V7" and config.get("exit") == "signal_reversal":
            ordered_work = sorted(generated_work, key=lambda item: (item[0].symbol, require_utc(item[1]["decision_ts"])))
            for index, (frame, event) in enumerate(ordered_work):
                reversal = next((require_utc(later[1]["decision_ts"]) for later in ordered_work[index + 1:] if later[0].symbol == frame.symbol and int(later[1]["side"]) == -int(event["side"])), None)
                event["signal_reversal_close_ts"] = () if reversal is None else (reversal,)
            generated_work = ordered_work
        for frame, event in generated_work:
            observation, materialized = _simulate_event(frame, family, config, event)
            ledger.append(materialized)
            if observation is not None:
                observations.append(observation)
    # Exact actual-exit non-overlap by definition and symbol.
    accepted: list[EventObservation] = []
    last_exit: dict[str, datetime] = {}
    for item in sorted(observations, key=lambda value: (value.symbol, value.entry_ts, value.event_id)):
        if item.symbol not in last_exit or item.entry_ts >= last_exit[item.symbol]:
            accepted.append(item)
            last_exit[item.symbol] = item.exit_ts
    if family == "A4_TSMOM_V7":
        cohort_sizes: dict[datetime, int] = {}
        for item in accepted:
            cohort_sizes[item.entry_ts] = cohort_sizes.get(item.entry_ts, 0) + 1
        accepted = [EventObservation(**{
            **item.__dict__,
            "base_net_bps": item.base_net_bps / cohort_sizes[item.entry_ts],
            "stress_net_bps": item.stress_net_bps / cohort_sizes[item.entry_ts],
            "holding_seconds_weighted": item.holding_seconds_weighted / cohort_sizes[item.entry_ts],
        }) for item in accepted]
    return {
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "executable_attempt_id": row["executable_attempt_id"],
        "canonical_economic_address_sha256": row["canonical_economic_address_sha256"],
        "observations": accepted,
        "ledger": ledger,
        "aggregate": aggregate_streaming(iter(accepted)),
    }


def execute_registered_attempt(
    row: Mapping[str, Any],
    frames: Sequence[FamilyInput],
    *,
    cache_authority: CacheAuthority,
    authorization: ExecutionAuthorization,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    parent_binding: Mapping[str, Any] | None = None,
    parent_frames: Sequence[FamilyInput] | None = None,
    control_id: str | None = None,
) -> dict[str, Any]:
    manifest = authorization.require()
    cache = cache_authority.validate(manifest, [*frames, *(parent_frames or ())])
    result = dispatch_registered_attempt(row, frames, registry_by_id=registry_by_id, parent_binding=parent_binding, parent_frames=parent_frames, control_id=control_id)
    result["cache_manifest_sha256"] = sha256_file(cache_authority.manifest_path)
    result["cache_content_identity_sha256"] = canonical_hash(cache)
    result["external_approval_sha256"] = sha256_file(authorization.external_approval_path)
    return result


def synthetic_probe_attempt(row: Mapping[str, Any], frames: Sequence[FamilyInput], *, registry_by_id: Mapping[str, Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    result = dispatch_registered_attempt(row, frames, registry_by_id=registry_by_id, **kwargs)
    result.update({"synthetic_only": True, "economic_outcomes_opened": False})
    return result


__all__ = [
    "AuthorizationError",
    "CacheAuthority",
    "ExecutionAuthorization",
    "dispatch_registered_attempt",
    "execute_registered_attempt",
    "synthetic_probe_attempt",
    "validate_registered_attempt",
]
