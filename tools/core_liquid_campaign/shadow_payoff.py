from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Mapping

from .canonical import canonical_hash
from .engine_types import FamilyInput, FundingInput, PROTECTED_START, SignalBar
from .executor import AuthorizationError
from .family_engines.common import require_utc
from .selection import EventObservation


SHADOW_PAYOFF_PROVIDER_VERSION = "stage24_deterministic_synthetic_post_entry_v1"
SHADOW_SCENARIOS = frozenset({
    "stable_pass", "negative_fail", "unstable_fail", "sparse_fail",
    "concentration_fail",
})


@dataclass
class ShadowPayoffProvider:
    """Outcome-firewalled materializer for production-path shadow execution."""

    campaign_identity: str
    seed: int = 240021
    scenario_by_attempt: Mapping[str, str] | None = None
    concentration_symbol_by_attempt: Mapping[str, str] | None = None
    calls: int = 0
    real_post_entry_rows_opened: int = 0
    real_funding_rows_opened: int = 0

    def __call__(
        self,
        frame: FamilyInput,
        attempt_id: str,
        family: str,
        config: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> tuple[EventObservation | None, dict[str, Any]]:
        decision = require_utc(event["decision_ts"])
        evaluation_end = require_utc(frame.metadata["evaluation_end_exclusive"])
        entry = decision + timedelta(minutes=5)
        if entry + timedelta(days=10, minutes=10) >= min(evaluation_end, PROTECTED_START):
            raise AuthorizationError("shadow event lacks a protected-safe synthetic payoff interval")
        identity = {
            "provider": SHADOW_PAYOFF_PROVIDER_VERSION,
            "campaign_identity": self.campaign_identity,
            "seed": self.seed,
            "attempt_id": attempt_id,
            "family": family,
            "attempt_config_sha256": canonical_hash(config),
            "event_id": str(event["event_id"]),
            "symbol": frame.symbol,
            "fold_id": frame.fold_id,
            "decision_ts": decision.isoformat(),
            "side": int(event["side"]),
        }
        digest = canonical_hash(identity)
        scenario = None if self.scenario_by_attempt is None else self.scenario_by_attempt.get(attempt_id)
        if scenario is not None and scenario not in SHADOW_SCENARIOS:
            raise AuthorizationError("shadow payoff scenario is not registered")
        if scenario == "sparse_fail" and int(digest[:8], 16) % 11:
            self.calls += 1
            return None, {
                "event_id": str(event["event_id"]), "status": "synthetic_scenario_suppressed",
                "shadow_only": True, "economic_outcome_opened": False,
                "real_post_entry_rows_opened": 0, "real_funding_rows_opened": 0,
                "provider_version": SHADOW_PAYOFF_PROVIDER_VERSION,
                "provider_identity_sha256": canonical_hash(identity),
                "synthetic_scenario": scenario, "actual_accounting_path_executed": False,
            }
        if scenario == "concentration_fail":
            permitted = None if self.concentration_symbol_by_attempt is None else self.concentration_symbol_by_attempt.get(attempt_id)
            if permitted is not None and frame.symbol != permitted:
                self.calls += 1
                return None, {
                    "event_id": str(event["event_id"]), "status": "synthetic_scenario_suppressed",
                    "shadow_only": True, "economic_outcome_opened": False,
                    "real_post_entry_rows_opened": 0, "real_funding_rows_opened": 0,
                    "provider_version": SHADOW_PAYOFF_PROVIDER_VERSION,
                    "provider_identity_sha256": canonical_hash(identity),
                    "synthetic_scenario": scenario, "actual_accounting_path_executed": False,
                }
        phase = int(digest[:8], 16) / float(0xFFFFFFFF) * math.tau
        drift = (int(digest[8:16], 16) / float(0xFFFFFFFF) - 0.5) * 0.00008
        prices = [100.0 + drift * index + 0.015 * math.sin(index / 17.0 + phase) for index in range(2886)]
        scenario_positive = scenario in {"stable_pass", "concentration_fail"}
        if scenario == "unstable_fail":
            scenario_positive = int(canonical_hash({"attempt": attempt_id, "fold": frame.fold_id})[:8], 16) % 2 == 0
        if scenario in SHADOW_SCENARIOS:
            side = int(event["side"])
            signed_peak = 6.0 if scenario_positive else -6.0
            prices = []
            for index in range(2886):
                if index <= 200:
                    signed_move = signed_peak * index / 200.0
                elif scenario_positive and index <= 240:
                    signed_move = signed_peak - (index - 200) * 0.025
                else:
                    signed_move = signed_peak - (1.0 if scenario_positive else 0.0)
                prices.append(100.0 + side * signed_move)
        if family == "A3_STARTER_RETEST_V3" and event.get("retest_depth") is not None and event.get("atr") is not None:
            side = int(event["side"]); level = float(event["level"]); atr = float(event["atr"]); depth = float(event["retest_depth"])
            outside = level + side * (depth + 0.5) * atr
            activation = level + side * depth * 0.5 * atr
            reclaim = level + side * depth * 0.75 * atr
            prices = [outside] * len(prices)
            prices[2] = activation
            reclaim_close_index = 4 + int(digest[16:18], 16) % 6
            for index in range(3, reclaim_close_index):
                prices[index] = level + side * depth * 0.4 * atr
            prices[reclaim_close_index] = reclaim
            for index in range(reclaim_close_index + 1, len(prices)):
                if scenario in SHADOW_SCENARIOS:
                    elapsed = index - reclaim_close_index
                    signed_peak = 6.0 if scenario_positive else -6.0
                    signed_move = signed_peak * min(elapsed, 200) / 200.0
                    if scenario_positive and elapsed > 200:
                        signed_move = signed_peak - min(elapsed - 200, 40) * 0.025
                    prices[index] = reclaim + side * signed_move
                else:
                    prices[index] = reclaim + side * 0.0001 * (index - reclaim_close_index)
        bars = tuple(
            SignalBar(
                entry + timedelta(minutes=5 * index),
                entry + timedelta(minutes=5 * (index + 1)),
                prices[index],
                max(prices[index], prices[index + 1]) + 0.005,
                min(prices[index], prices[index + 1]) - 0.005,
                prices[index + 1],
                entry + timedelta(minutes=5 * (index + 1)),
                entry + timedelta(minutes=5 * (index + 1)),
            )
            for index in range(len(prices) - 1)
        )
        funding_start = entry.replace(minute=0, second=0, microsecond=0)
        rate = 0.0 if scenario in SHADOW_SCENARIOS else (int(digest[18:26], 16) / float(0xFFFFFFFF) - 0.5) * 0.02
        synthetic_funding = tuple(
            FundingInput(
                funding_start + timedelta(hours=index),
                funding_start + timedelta(hours=index),
                format(rate, ".12f"),
                "exact",
            )
            for index in range(242)
        )
        synthetic_frame = replace(
            frame,
            five_minute_bars=bars,
            funding=synthetic_funding,
            metadata={
                **frame.metadata,
                "evaluation_start": entry,
                "shadow_post_entry_source": "deterministic_synthetic_path",
                "shadow_post_entry_path_sha256": canonical_hash(prices),
            },
        )
        synthetic_event = {**event, "entry_index": 0}
        from .executor import _simulate_event
        observation, materialized = _simulate_event(synthetic_frame, family, config, synthetic_event)
        self.calls += 1
        return observation, {
            **materialized,
            "shadow_only": True,
            "economic_outcome_opened": False,
            "real_post_entry_rows_opened": 0,
            "real_funding_rows_opened": 0,
            "provider_version": SHADOW_PAYOFF_PROVIDER_VERSION,
            "provider_identity_sha256": canonical_hash(identity),
            "synthetic_path_sha256": canonical_hash(prices),
            "synthetic_path_rows": len(bars),
            "synthetic_funding_rows": len(synthetic_funding),
            "synthetic_scenario": scenario or "legacy_deterministic_path",
            "synthetic_funding_schedule_sha256": canonical_hash([
                {
                    "row_timestamp": row.row_timestamp.isoformat(),
                    "publication_ts": row.publication_ts.isoformat(),
                    "absolute_rate_usd_per_contract_unit": row.absolute_rate_usd_per_contract_unit,
                    "source_partition": row.source_partition,
                }
                for row in synthetic_funding
            ]),
            "actual_accounting_path_executed": True,
        }

    def attestation(self) -> dict[str, Any]:
        if self.real_post_entry_rows_opened or self.real_funding_rows_opened:
            raise AuthorizationError("shadow payoff provider crossed the outcome firewall")
        return {
            "provider_version": SHADOW_PAYOFF_PROVIDER_VERSION,
            "campaign_identity": self.campaign_identity,
            "seed": self.seed,
            "registered_scenario_count": len(self.scenario_by_attempt or {}),
            "calls": self.calls,
            "real_post_entry_rows_opened": self.real_post_entry_rows_opened,
            "real_funding_rows_opened": self.real_funding_rows_opened,
            "economic_outcomes_opened": False,
        }


__all__ = ["SHADOW_PAYOFF_PROVIDER_VERSION", "ShadowPayoffProvider"]
