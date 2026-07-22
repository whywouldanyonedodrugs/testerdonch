from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, atomic_write_jsonl, canonical_hash, sha256_file
from .kda02b_population_index import SELECTED_MODELS
from .schema import CAMPAIGN_ID, gower_distance
from .selection import selection_role_eligible


SHADOW_PACKET_SCHEMA = "stage24_bounded_shadow_campaign_packet_v1"
SHADOW_MANIFEST_SCHEMA = "stage24_bounded_shadow_campaign_manifest_v1"


class ShadowCampaignPacketError(RuntimeError):
    pass


class BoundedShadowPopulationSchedule:
    """Outcome-free canary slice over an already validated full schedule."""

    def __init__(self, complete_schedule: Any, policy: Mapping[str, Any], *, population_adapter: Any | None = None) -> None:
        self.complete_schedule = complete_schedule
        self.population_adapter = population_adapter
        self.partitions = getattr(complete_schedule, "partitions", {})
        self._last_frame_key: tuple[Any, ...] | None = None
        self._last_frame: Any | None = None
        self._a1_raw_arrays: dict[str, tuple[Any, Any]] = {}
        if policy.get("schema") == "stage24_shadow_event_locator_policy_v2":
            expected_v2 = {
                "schema": "stage24_shadow_event_locator_policy_v2",
                "selection": "actual_production_enumerator_pre_entry_event_locators",
                "target_eligible_event_days_per_attempt_partition": int(policy.get("target_eligible_event_days_per_attempt_partition", 0)),
                "maximum_candidate_locators_per_attempt_partition": int(policy.get("maximum_candidate_locators_per_attempt_partition", 0)),
                "empty_attempt_partitions_preserved": True,
                "real_post_entry_values_used": False,
                "economic_values_used_for_selection": False,
                "synthetic_payoff_generated_after_locator_freeze": True,
                "full_launch_population_authority_preserved": True,
            }
            if (
                dict(policy) != expected_v2
                or not 1 <= expected_v2["target_eligible_event_days_per_attempt_partition"] <= 30
                or not 100 <= expected_v2["maximum_candidate_locators_per_attempt_partition"] <= 10_000
            ):
                raise ShadowCampaignPacketError("shadow event-locator policy is invalid or broadened")
            self.policy = expected_v2
            self.mode = "production_event_locator"
            return
        expected = {
            "schema": "stage24_shadow_population_slice_policy_v1",
            "selection": "first_authority_order_locator_on_each_distinct_UTC_day",
            "maximum_distinct_days_per_attempt_partition": int(policy.get("maximum_distinct_days_per_attempt_partition", 0)),
            "economic_values_used_for_selection": False,
            "benchmark_frame_values_used": False,
            "full_launch_population_authority_preserved": True,
        }
        if dict(policy) != expected or not 1 <= expected["maximum_distinct_days_per_attempt_partition"] <= 30:
            raise ShadowCampaignPacketError("shadow population slice policy is invalid or broadened")
        self.policy = expected
        self.mode = "legacy_first_daily"

    def iter_locators(self, attempt: Mapping[str, Any], **kwargs: Any) -> Iterable[Any]:
        if self.mode == "production_event_locator":
            if attempt.get("family_id") == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                parent = kwargs.pop("parent_attempt", None)
                if not isinstance(parent, Mapping):
                    raise ShadowCampaignPacketError("A2 event-locator sampling lacks its exact parent")
                for locator in self.iter_batch_locators((parent,), **kwargs):
                    overlay_locator = replace(
                        locator,
                        family_id="A2_PRIOR_HIGH_RS_CONTEXT_V1",
                        executable_attempt_id=str(attempt["executable_attempt_id"]),
                        canonical_economic_address_sha256=str(attempt["canonical_economic_address_sha256"]),
                    )
                    # The real adapter intentionally rejects generic A2
                    # locators.  Preserve the exact parent-derived pre-entry
                    # frame under the registered overlay locator; the owning
                    # executor constructs the parent frame separately.
                    self._last_frame_key = (
                        overlay_locator.family_id, overlay_locator.symbol,
                        overlay_locator.decision_ts, overlay_locator.outer_fold_id,
                        overlay_locator.inner_fold_id,
                    )
                    yield overlay_locator
                return
            yield from self.iter_batch_locators((attempt,), **kwargs)
            return
        selected_days: set[object] = set()
        maximum = int(self.policy["maximum_distinct_days_per_attempt_partition"])
        for locator in self.complete_schedule.iter_locators(attempt, **kwargs):
            day = locator.decision_ts.date()
            if day in selected_days:
                continue
            selected_days.add(day)
            yield replace(
                locator,
                executable_attempt_id=str(attempt["executable_attempt_id"]),
                canonical_economic_address_sha256=str(attempt["canonical_economic_address_sha256"]),
            )
            if len(selected_days) >= maximum:
                return

    def count(self, attempt: Mapping[str, Any], **kwargs: Any) -> int:
        return sum(1 for _ in self.iter_locators(attempt, **kwargs))

    @staticmethod
    def _confirmation_offset(config: Mapping[str, Any]) -> timedelta:
        base = str(config["base_duration"]); units = {"h": 12, "d": 288}
        base_bars = int(base[:-1]) * units[base[-1]]
        confirmation = {"one_close": 1, "two_closes": 2, "close_plus_bounded_15m_delay": 4}[str(config["confirmation"])]
        return timedelta(minutes=5 * (base_bars + confirmation))

    def _pit_eligible(self, attempt: Mapping[str, Any], symbol: str, decision: datetime) -> bool:
        adapter = self.population_adapter
        if adapter is None:
            raise ShadowCampaignPacketError("production event-locator sampling lacks its FamilyInput adapter")
        day_ms = int(decision.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        row = next((item for item in adapter._pit_by_day.get(day_ms, ()) if str(item["symbol"]) == symbol), None)
        if row is None or not bool(row[f"top_{int(attempt['config']['PIT_liquidity_top_n'])}"]):
            return False
        return int(row["decision_count_5m"]) == 288 or decision == decision.replace(hour=0, minute=0)

    def _a3_candidates(
        self,
        rows: Sequence[Mapping[str, Any]],
        partition: Any,
        partition_key: tuple[str, str, str | None],
    ) -> Iterable[tuple[str, datetime]]:
        import numpy as np

        adapter = self.population_adapter
        if adapter is None:
            raise ShadowCampaignPacketError("A3 event sampling lacks its FamilyInput adapter")
        root = adapter._a3.cache_root; manifest = adapter._a3.manifest
        candidates: set[tuple[str, datetime]] = set()
        end_ms = partition.evaluation_end_exclusive_ms - 11 * 86_400_000
        for row in rows:
            config = row["config"]; side = 1 if config["direction"] == "long" else -1
            scope = str(config["breakout_rank_scope"])
            confirmation_offset = {
                "one_close": timedelta(0), "two_closes": timedelta(minutes=5),
                "close_plus_15m_delay": timedelta(minutes=15),
            }.get(str(config["confirmation"]))
            if scope not in {"symbol_side", "global_side"} or confirmation_offset is None:
                raise ShadowCampaignPacketError("bounded A3 event cohort has an unsupported scope or confirmation")
            name = f"A3_breakout:lookback={config['breakout_lookback_days']}:atr={config['ATR_window_days']}:side={side}"
            record = manifest["features"].get(name)
            if record is None:
                raise ShadowCampaignPacketError(f"A3 event-locator authority omits {name}")
            timestamps = np.load(root / record["timestamps_path"], mmap_mode="r", allow_pickle=False)
            symbols = np.load(root / record["symbols_path"], mmap_mode="r", allow_pickle=False)
            values = np.load(root / record["values_path"], mmap_mode="r", allow_pickle=False)
            offset_ms = int(confirmation_offset.total_seconds() * 1000)
            begin = int(np.searchsorted(timestamps, partition.evaluation_start_ms - offset_ms, side="left"))
            end = int(np.searchsorted(timestamps, end_ms - offset_ms, side="left"))
            training = adapter.partitions[partition_key]
            training_start_ms = int(training["training_start"].timestamp() * 1000)
            training_end_ms = int(training["training_end_exclusive"].timestamp() * 1000)
            probability = int(str(config["breakout_rank_min"])[1:]) / 100.0
            thresholds: dict[int, float] = {}
            for timestamp, code, value in zip(timestamps[begin:end], symbols[begin:end], values[begin:end]):
                code_value = int(code); threshold_key = code_value if scope == "symbol_side" else 0
                if threshold_key not in thresholds:
                    symbol_mask = (timestamps >= training_start_ms) & (timestamps < training_end_ms) & np.isfinite(values)
                    if scope == "symbol_side":
                        symbol_mask &= symbols == code_value
                    population = np.asarray(values[symbol_mask], dtype="<f8")
                    thresholds[threshold_key] = math.inf if len(population) < 30 else float(np.quantile(population, probability, method="linear"))
                if float(value) < thresholds[threshold_key]:
                    continue
                decision = datetime.fromtimestamp(int(timestamp) / 1000, tz=timezone.utc) + confirmation_offset
                symbol = adapter._a3.by_code[int(code)]
                if self._pit_eligible(row, symbol, decision):
                    candidates.add((symbol, decision))
        yield from sorted(candidates, key=lambda item: (item[1], item[0]))

    def _a1_candidates(
        self,
        rows: Sequence[Mapping[str, Any]],
        partition: Any,
        partition_key: tuple[str, str, str | None],
    ) -> Iterable[tuple[str, datetime]]:
        import numpy as np
        from .production_population_tables import _load_trade_arrays

        adapter = self.population_adapter
        if adapter is None:
            raise ShadowCampaignPacketError("A1 event sampling lacks its FamilyInput adapter")
        authority = adapter._a1; root = authority.cache_root; manifest = authority.manifest
        timestamps = authority._timestamps; symbols = authority._symbols
        candidates: set[tuple[str, datetime]] = set()
        for row in rows:
            config = row["config"]
            if config["direction"] not in {"long", "short"} or config["impulse_rank_scope"] != "symbol_side":
                raise ShadowCampaignPacketError("bounded A1 event cohort requires a directional symbol-side impulse address")
            side = 1 if config["direction"] == "long" else -1
            name = f"A1_impulse:window={config['impulse_window']}"
            values = np.load(root / manifest["features"][name]["path"], mmap_mode="r", allow_pickle=False)
            threshold_probability = int(str(config["impulse_rank_min"])[1:]) / 100.0
            offset = self._confirmation_offset(config)
            base = str(config["base_duration"]); base_bars = int(base[:-1]) * {"h": 12, "d": 288}[base[-1]]
            if config["shape_rank_scope"] != "symbol":
                raise ShadowCampaignPacketError("bounded A1 event cohort requires symbol-local shape ranks")
            contraction_name = f"A1_contraction:base={base}:baseline={config['contraction_baseline']}"
            contraction_values = np.load(root / manifest["features"][contraction_name]["path"], mmap_mode="r", allow_pickle=False)
            smoothness_values = None
            if config["smoothness_rank_min"] != "none":
                smoothness_name = f"A1_smoothness:base={base}"
                smoothness_values = np.load(root / manifest["features"][smoothness_name]["path"], mmap_mode="r", allow_pickle=False)
            training_start_ms = int(adapter.partitions[partition_key]["training_start"].timestamp() * 1000)
            training_end_ms = int(adapter.partitions[partition_key]["training_end_exclusive"].timestamp() * 1000)
            for symbol, code in sorted(authority.symbol_codes.items()):
                left = int(np.searchsorted(symbols, code, side="left")); right = int(np.searchsorted(symbols, code, side="right"))
                symbol_times = timestamps[left:right]; symbol_values = np.asarray(values[left:right], dtype="<f8") * side
                training = symbol_values[(symbol_times >= training_start_ms) & (symbol_times < training_end_ms) & np.isfinite(symbol_values)]
                if len(training) < 30:
                    continue
                ordered_impulse = np.sort(training, kind="mergesort")
                impulse_start_ms = partition.evaluation_start_ms - int(offset.total_seconds() * 1000)
                impulse_end_ms = partition.evaluation_end_exclusive_ms - 11 * 86_400_000 - int(offset.total_seconds() * 1000)
                start = int(np.searchsorted(symbol_times, impulse_start_ms, side="left")); stop = int(np.searchsorted(symbol_times, impulse_end_ms, side="left"))
                local = symbol_values[start:stop]
                if not len(local):
                    continue
                previous = symbol_values[max(0, start - 1):max(0, stop - 1)]
                if start == 0:
                    previous = np.concatenate((np.asarray([np.nan]), local[:-1]))
                current_percentiles = np.searchsorted(ordered_impulse, local, side="right") / len(ordered_impulse)
                previous_percentiles = np.searchsorted(ordered_impulse, previous, side="right") / len(ordered_impulse)
                crossed = np.flatnonzero(np.isfinite(local) & (current_percentiles >= threshold_probability) & np.isfinite(previous) & (previous_percentiles < threshold_probability))
                shape_training_mask = (symbol_times >= training_start_ms) & (symbol_times < training_end_ms)
                contraction_training = np.sort(np.asarray(contraction_values[left:right][shape_training_mask & np.isfinite(contraction_values[left:right])], dtype="<f8"), kind="mergesort")
                smoothness_training = None if smoothness_values is None else np.sort(np.asarray(smoothness_values[left:right][shape_training_mask & np.isfinite(smoothness_values[left:right])], dtype="<f8"), kind="mergesort")
                raw_times = raw_closes = None
                for local_index in crossed:
                    physical_index = left + start + int(local_index)
                    base_end_index = physical_index + base_bars
                    if base_end_index >= right or int(timestamps[base_end_index]) - int(timestamps[physical_index]) != base_bars * 300_000:
                        continue
                    if config["contraction_rank_max"] != "none":
                        contraction = float(contraction_values[base_end_index])
                        contraction_max = int(str(config["contraction_rank_max"])[1:]) / 100.0
                        if not len(contraction_training) or not math.isfinite(contraction) or float(np.searchsorted(contraction_training, contraction, side="right")) / len(contraction_training) > contraction_max:
                            continue
                    if smoothness_training is not None:
                        smoothness = float(smoothness_values[base_end_index])
                        smoothness_min = int(str(config["smoothness_rank_min"])[1:]) / 100.0
                        if not len(smoothness_training) or not math.isfinite(smoothness) or float(np.searchsorted(smoothness_training, smoothness, side="right")) / len(smoothness_training) < smoothness_min:
                            continue
                    if raw_times is None or raw_closes is None:
                        cached_raw = self._a1_raw_arrays.get(symbol)
                        if cached_raw is None:
                            cached_raw = _load_trade_arrays(adapter._parts(symbol))
                            self._a1_raw_arrays[symbol] = cached_raw
                        raw_times, raw_closes = cached_raw
                    impulse_close_ms = int(symbol_times[start + int(local_index)])
                    raw_index = int(np.searchsorted(raw_times, impulse_close_ms - 300_000, side="left"))
                    impulse_bars = {"6h": 72, "12h": 144, "1d": 288, "3d": 864, "7d": 2016}[str(config["impulse_window"])]
                    confirmation_offsets = {
                        "one_close": (base_bars + 1,),
                        "two_closes": (base_bars + 1, base_bars + 2),
                        "close_plus_bounded_15m_delay": (base_bars + 1, base_bars + 4),
                    }[str(config["confirmation"])]
                    required_end = raw_index + max(confirmation_offsets)
                    if raw_index >= len(raw_times) or int(raw_times[raw_index]) != impulse_close_ms - 300_000 or raw_index < impulse_bars or required_end >= len(raw_times):
                        continue
                    if int(raw_times[required_end]) - int(raw_times[raw_index]) != max(confirmation_offsets) * 300_000:
                        continue
                    extreme = (
                        float(np.max(raw_closes[raw_index - impulse_bars:raw_index + 1]))
                        if side == 1 else float(np.min(raw_closes[raw_index - impulse_bars:raw_index + 1]))
                    )
                    base_closes = raw_closes[raw_index + 1:raw_index + base_bars + 1]
                    confirmation_closes = raw_closes[[raw_index + item for item in confirmation_offsets]]
                    if np.any(side * (base_closes - extreme) > 0) or np.any(side * (confirmation_closes - extreme) <= 0):
                        continue
                    decision = datetime.fromtimestamp(int(symbol_times[start + int(local_index)]) / 1000, tz=timezone.utc) + offset
                    if self._pit_eligible(row, symbol, decision):
                        candidates.add((symbol, decision))
        yield from sorted(candidates, key=lambda item: (item[1], item[0]))

    def _candidate_pairs(self, rows: Sequence[Mapping[str, Any]], *, phase: str, outer_fold_id: str, inner_fold_id: str | None) -> Iterable[tuple[str, datetime]]:
        family = str(rows[0]["family_id"])
        partition_key = (phase, outer_fold_id, inner_fold_id)
        partition = self.complete_schedule.partition(phase=phase, outer_fold_id=outer_fold_id, inner_fold_id=inner_fold_id)
        if family == "A1_COMPRESSION_V2":
            yield from self._a1_candidates(rows, partition, partition_key)
        elif family == "A3_STARTER_RETEST_V3":
            yield from self._a3_candidates(rows, partition, partition_key)
        elif family == "A4_TSMOM_V7":
            seen: set[tuple[str, datetime]] = set()
            for row in rows:
                for locator in self.complete_schedule.iter_locators(row, phase=phase, outer_fold_id=outer_fold_id, inner_fold_id=inner_fold_id):
                    if locator.decision_ts.timestamp() * 1000 >= partition.evaluation_end_exclusive_ms - 11 * 86_400_000:
                        continue
                    key = locator.symbol, locator.decision_ts
                    if key not in seen:
                        seen.add(key); yield key
        else:
            raise ShadowCampaignPacketError("event-locator sampler received a non-parent family")

    def iter_batch_locators(self, rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> Iterable[Any]:
        from .a1_state import initial_state
        from .family_engines import a1_compression
        from .executor import _generate_events
        from .family_engines.common import EngineInputError, require_utc
        from .lazy_production_inputs import FamilyDecisionLocator
        from .engine_types import _mark_validated_frame

        if self.mode != "production_event_locator":
            yield from self.iter_locators(rows[0], **kwargs)
            return
        if not rows or self.population_adapter is None:
            raise ShadowCampaignPacketError("event-locator batch is empty or lacks its production adapter")
        family = str(rows[0]["family_id"])
        prototype = rows[0]
        target = int(self.policy["target_eligible_event_days_per_attempt_partition"])
        maximum = int(self.policy["maximum_candidate_locators_per_attempt_partition"]) * len(rows)
        eligible_days: dict[str, set[object]] = {str(row["executable_attempt_id"]): set() for row in rows}
        scanned = 0
        for symbol, decision in self._candidate_pairs(rows, **kwargs):
            scanned += 1
            if scanned > maximum:
                break
            locator = FamilyDecisionLocator(
                family, str(kwargs["phase"]), str(kwargs["outer_fold_id"]), kwargs.get("inner_fold_id"),
                symbol, decision, str(prototype["executable_attempt_id"]),
                str(prototype["canonical_economic_address_sha256"]),
            )
            try:
                frame = self.population_adapter.frame(locator)
            except EngineInputError:
                continue
            matched = False
            for row in rows:
                try:
                    if family == "A1_COMPRESSION_V2":
                        bound = replace(frame, metadata={**frame.metadata, "a1_persistent_state": initial_state().payload()})
                        _mark_validated_frame(bound)
                        events = _generate_events(family, bound, row["config"], None, None)
                        _, events = a1_compression.advance_persistent_state(bound, row["config"], events)
                        exact = [event for event in events if require_utc(event["decision_ts"]) == decision]
                    else:
                        events = _generate_events(family, frame, row["config"], None, None)
                        exact = [event for event in events if require_utc(event["decision_ts"]) == decision]
                except EngineInputError:
                    exact = []
                if exact:
                    eligible_days[str(row["executable_attempt_id"])].add(decision.date()); matched = True
            if not matched:
                continue
            self._last_frame_key = (family, symbol, decision, str(kwargs["outer_fold_id"]), kwargs.get("inner_fold_id"))
            self._last_frame = frame
            yield locator
            if all(len(days) >= target for days in eligible_days.values()):
                self.last_reconciliation = {
                    "family_id": family,
                    "partition": {key: value for key, value in kwargs.items() if key in {"phase", "outer_fold_id", "inner_fold_id"}},
                    "eligible_event_days_by_attempt": {key: len(value) for key, value in sorted(eligible_days.items())},
                    "candidate_locators_scanned": scanned,
                    "real_post_entry_rows_opened": 0,
                    "economic_outcomes_opened": False,
                    "status": "pass",
                }
                return
        missing = {identity: len(days) for identity, days in eligible_days.items() if len(days) < target}
        self.last_reconciliation = {
            "family_id": family,
            "partition": {key: value for key, value in kwargs.items() if key in {"phase", "outer_fold_id", "inner_fold_id"}},
            "eligible_event_days_by_attempt": {key: len(value) for key, value in sorted(eligible_days.items())},
            "candidate_locators_scanned": scanned,
            "real_post_entry_rows_opened": 0,
            "economic_outcomes_opened": False,
            "status": "pass" if not missing else "fail_insufficient_event_days",
        }
        return

    def frame(self, locator: Any) -> Any:
        key = (locator.family_id, locator.symbol, locator.decision_ts, locator.outer_fold_id, locator.inner_fold_id)
        if self._last_frame_key == key and self._last_frame is not None:
            if locator.family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                from .engine_types import _mark_validated_frame

                rebound = replace(self._last_frame, metadata={
                    **self._last_frame.metadata,
                    "requested_family_id": locator.family_id,
                    "decision_locator": locator.identity_payload(),
                    "decision_locator_sha256": canonical_hash(locator.identity_payload()),
                })
                _mark_validated_frame(rebound)
                return rebound
            return self._last_frame
        if self.population_adapter is None:
            raise ShadowCampaignPacketError("event-locator frame cache lacks its production adapter")
        return self.population_adapter.frame(locator)

class BoundedShadowKDA02BAdapter:
    """Cap eligible KDA frames by fold while preserving unavailable rows seen."""

    def __init__(self, complete_adapter: Any, policy: Mapping[str, Any]) -> None:
        expected = {
            "schema": "stage24_shadow_kda02b_slice_policy_v1",
            "selection": "first_authority_order_eligible_record_per_outer_fold",
            "maximum_eligible_records_per_cell_fold": int(policy.get("maximum_eligible_records_per_cell_fold", 0)),
            "typed_unavailable_rows": "preserve_every_row_encountered_before_slice_completion",
            "economic_values_used_for_selection": False,
            "full_kda02b_population_authority_preserved": True,
        }
        if dict(policy) != expected or not 1 <= expected["maximum_eligible_records_per_cell_fold"] <= 3:
            raise ShadowCampaignPacketError("shadow KDA02B population slice policy is invalid or broadened")
        self.complete_adapter = complete_adapter
        self.policy = expected
        self.last_reconciliation: dict[str, Any] | None = None

    def stream(self, *, cell_id: str | None = None, outer_fold_id: str | None = None) -> Iterable[Any]:
        maximum = int(self.policy["maximum_eligible_records_per_cell_fold"])
        eligible: Counter[str] = Counter()
        unavailable = 0
        expected_folds = {outer_fold_id} if outer_fold_id is not None else {model.removeprefix("Q_") for model in SELECTED_MODELS}
        for record in self.complete_adapter.stream(cell_id=cell_id, outer_fold_id=outer_fold_id):
            fold = str(record.outer_fold_id)
            if record.status == "typed_unavailable":
                unavailable += 1
                yield record
            elif eligible[fold] < maximum:
                eligible[fold] += 1
                yield record
            if expected_folds and all(eligible[fold] >= maximum for fold in expected_folds):
                break
        if not expected_folds or any(eligible[fold] < maximum for fold in expected_folds):
            raise ShadowCampaignPacketError("KDA02B shadow slice lacks an eligible record in a required fold")
        self.last_reconciliation = {
            "cell_id": cell_id,
            "eligible_records_by_outer_fold": dict(sorted(eligible.items())),
            "typed_unavailable_rows_preserved": unavailable,
            "slice_policy_sha256": canonical_hash(self.policy),
            "economic_outcomes_opened": False,
            "status": "pass",
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ShadowCampaignPacketError(f"required frozen registry is absent: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _file_record(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise ShadowCampaignPacketError(f"required shadow authority file is absent: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _exact_subset(
    rows: Sequence[Mapping[str, Any]],
    selected: set[str],
    *,
    identity_field: str,
    label: str,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows if str(row.get(identity_field)) in selected]
    observed = {str(row[identity_field]) for row in output}
    if observed != selected:
        raise ShadowCampaignPacketError(f"{label} subset contains an unknown or missing frozen identity")
    return output


def _shadow_scenario_matrix(execution: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze test-only payoff scenarios from identities, never outcome values."""

    parent_ids = {
        str(row.get("resolved_parent_executable_attempt_id"))
        for row in execution
        if row.get("family_id") == "A2_PRIOR_HIGH_RS_CONTEXT_V1"
        and row.get("config", {}).get("parent_binding_mode") == "source_attempt"
    }
    assignments: list[dict[str, Any]] = []
    failures = iter(("negative_fail", "unstable_fail", "sparse_fail", "concentration_fail") * 3)
    for family in ("A4_TSMOM_V7", "A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"):
        rows = sorted(
            (row for row in execution if row.get("family_id") == family),
            key=lambda row: (
                sum(
                    row.get("config", {}).get(field) != "none"
                    for field in ("contraction_rank_max", "smoothness_rank_min")
                ) if family == "A1_COMPRESSION_V2" else 0,
                str(row["canonical_economic_address_sha256"]),
            ),
        )
        if len(rows) < 5:
            raise ShadowCampaignPacketError(f"shadow scenario matrix lacks a five-address {family} neighborhood")
        stable = {str(row["executable_attempt_id"]) for row in rows[:3]} | (parent_ids & {str(row["executable_attempt_id"]) for row in rows})
        while len(stable) < 3:
            stable.add(str(rows[len(stable)]["executable_attempt_id"]))
        for row in rows:
            identity = str(row["executable_attempt_id"])
            scenario = "stable_pass" if identity in stable else next(failures)
            assignments.append({
                "executable_attempt_id": identity,
                "canonical_economic_address_sha256": str(row["canonical_economic_address_sha256"]),
                "family_id": family,
                "scenario": scenario,
                "concentration_symbol": "PF_XBTUSD" if scenario == "concentration_fail" else None,
            })
    observed = {str(row["scenario"]) for row in assignments}
    expected = {"stable_pass", "negative_fail", "unstable_fail", "sparse_fail", "concentration_fail"}
    if observed != expected or not parent_ids <= {str(row["executable_attempt_id"]) for row in assignments if row["scenario"] == "stable_pass"}:
        raise ShadowCampaignPacketError("shadow scenario matrix does not cover every fixed scenario or A2 source parent")
    payload = {
        "schema": "stage24_fixed_synthetic_scenario_matrix_v1",
        "selection_rule": "configuration_only_filter_count_then_canonical_address_first_three_stable_plus_every_A2_source_parent_then_fixed_failure_cycle",
        "assignments": assignments,
        "assignment_inventory_sha256": canonical_hash(assignments),
        "fixed_before_synthetic_payoff": True,
        "real_outcomes_used": False,
        "economic_outcomes_opened": False,
    }
    return payload


def _topology_neighborhood(
    family: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    required_id: str | None = None,
    required_config: Mapping[str, Any] | None = None,
    preferred_center_id: str | None = None,
) -> list[Mapping[str, Any]]:
    eligible = [
        row for row in rows
        if row.get("family_id") == family
        and selection_role_eligible(row)
        and all(row.get("config", {}).get(key) == value for key, value in (required_config or {}).items())
    ]
    if required_id is not None:
        required = next((row for row in rows if str(row.get("executable_attempt_id")) == required_id), None)
        if required is None or required.get("family_id") != family:
            raise ShadowCampaignPacketError("required A2 source parent is absent or has the wrong family")
    else:
        required = None
    centers = sorted(eligible, key=lambda row: str(row["canonical_economic_address_sha256"]))
    if preferred_center_id is not None:
        preferred = next((row for row in centers if str(row["executable_attempt_id"]) == preferred_center_id), None)
        if preferred is None:
            raise ShadowCampaignPacketError("preferred topology center is absent")
        centers = [preferred]
    for center in centers:
        neighbors = sorted(
            (row for row in eligible if gower_distance(family, center["config"], row["config"]) <= 0.15),
            key=lambda row: (gower_distance(family, center["config"], row["config"]), str(row["canonical_economic_address_sha256"])),
        )
        selected = neighbors[:5]
        varied = sum(len({str(row["config"].get(field)) for row in selected}) > 1 for field in center["config"])
        if len(selected) == 5 and varied >= 2:
            if required is not None and str(required["executable_attempt_id"]) not in {str(row["executable_attempt_id"]) for row in selected}:
                selected.append(required)
            return selected
    raise ShadowCampaignPacketError(f"no exact five-address topology neighborhood exists for {family}")


def select_bounded_shadow_identities(source_packet_root: Path) -> dict[str, list[str]]:
    """Select a deterministic, configuration-only canary view before payoffs."""

    execution = _read_jsonl(source_packet_root / "FINAL_EXECUTION_REGISTRY.jsonl")
    controls = _read_jsonl(source_packet_root / "FINAL_CONTROL_REGISTRY.jsonl")
    a2_rows = [
        row for row in execution
        if row.get("family_id") == "A2_PRIOR_HIGH_RS_CONTEXT_V1"
        and row.get("config", {}).get("parent_binding_mode") == "source_attempt"
        and selection_role_eligible(row)
    ]
    a2_selected: list[Mapping[str, Any]] | None = None
    parent_id: str | None = None
    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for row in a2_rows:
        by_parent.setdefault(str(row.get("resolved_parent_executable_attempt_id")), []).append(row)
    for candidate_parent, rows in sorted(by_parent.items()):
        try:
            selected = _topology_neighborhood("A2_PRIOR_HIGH_RS_CONTEXT_V1", rows)
        except ShadowCampaignPacketError:
            continue
        a2_selected = selected; parent_id = candidate_parent; break
    if a2_selected is None or parent_id is None:
        raise ShadowCampaignPacketError("no exact A2 source-parent topology neighborhood exists")

    parent = next((row for row in execution if str(row.get("executable_attempt_id")) == parent_id), None)
    if parent is None or parent.get("family_id") not in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"}:
        raise ShadowCampaignPacketError("A2 topology parent is absent or unsupported")
    selected_rows: list[Mapping[str, Any]] = []
    for family in ("A4_TSMOM_V7", "A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"):
        selected_rows.extend(_topology_neighborhood(
            family, execution,
            required_id=parent_id if parent.get("family_id") == family else None,
            required_config=(
                {"breakout_rank_scope": "global_side", "confirmation": "one_close"}
                if family == "A3_STARTER_RETEST_V3" else
                None
            ),
            preferred_center_id="A1_COMPRESSION_V2:S22:L:1668:1"
            if family == "A1_COMPRESSION_V2" else None,
        ))
    selected_rows.extend(a2_selected)
    kda = [row for row in execution if row.get("family_id") == "KDA02B_SURVIVOR_ADJUDICATION_V1"]
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for row in kda:
        cells.setdefault(str(row["config"]["stage20_cell_id"]), []).append(row)
    selected_cell = next((rows for _, rows in sorted(cells.items()) if len(rows) == 11), None)
    if selected_cell is None:
        raise ShadowCampaignPacketError("no complete 11-variant KDA02B cell exists")
    selected_rows.extend(selected_cell)

    controls_by_class: dict[str, Mapping[str, Any]] = {}
    for row in sorted(controls, key=lambda item: (str(item["control_id"]), str(item["control_attempt_id"]))):
        if str(row.get("parent_slot", "")).endswith("beam:01"):
            controls_by_class.setdefault(str(row["control_id"]), row)
    if len(controls_by_class) != 20:
        raise ShadowCampaignPacketError("one exact beam-01 control is not available for every control class")
    return {
        "executable_attempt_ids": sorted({str(row["executable_attempt_id"]) for row in selected_rows}),
        "control_attempt_ids": sorted(str(row["control_attempt_id"]) for row in controls_by_class.values()),
    }


def build_bounded_shadow_packet(
    *,
    source_packet_root: Path,
    output_root: Path,
    execution_input_authority_path: Path,
    cache_manifest_path: Path,
    launch_population_authority_path: Path,
    kda02b_population_manifest_path: Path,
    executable_attempt_ids: Sequence[str],
    control_attempt_ids: Sequence[str],
    include_synthetic_scenario_matrix: bool = False,
) -> dict[str, Any]:
    """Freeze an exact no-outcome subset consumed by CampaignOrchestrator.

    The function copies registry rows, never edits their economic fields.  The
    full launch-population authorities remain separately bound; the 567-frame
    semantic cache is explicitly a canary/benchmark probe and cannot become the
    eventual launch-input authority through this packet.
    """

    source_packet_root = source_packet_root.resolve()
    output_root = output_root.resolve()
    source_paths = {
        "strategy": source_packet_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl",
        "execution": source_packet_root / "FINAL_EXECUTION_REGISTRY.jsonl",
        "controls": source_packet_root / "FINAL_CONTROL_REGISTRY.jsonl",
        "counterparts": source_packet_root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl",
    }
    source_rows = {name: _read_jsonl(path) for name, path in source_paths.items()}
    attempt_ids = {str(value) for value in executable_attempt_ids}
    control_ids = {str(value) for value in control_attempt_ids}
    if not attempt_ids or not control_ids:
        raise ShadowCampaignPacketError("bounded shadow subset must include attempts and controls")

    execution = _exact_subset(
        source_rows["execution"], attempt_ids,
        identity_field="executable_attempt_id", label="execution",
    )
    strategy = [dict(row) for row in source_rows["strategy"] if str(row.get("executable_attempt_id")) in attempt_ids]
    if not strategy or {str(row["executable_attempt_id"]) for row in strategy} != attempt_ids:
        raise ShadowCampaignPacketError("strategy multiplicity subset does not cover every selected execution")
    controls = _exact_subset(
        source_rows["controls"], control_ids,
        identity_field="control_attempt_id", label="control",
    )
    a2_ids = {str(row["executable_attempt_id"]) for row in execution if row.get("family_id") == "A2_PRIOR_HIGH_RS_CONTEXT_V1"}
    counterparts = [dict(row) for row in source_rows["counterparts"] if str(row.get("a2_executable_attempt_id")) in a2_ids]
    if {str(row["a2_executable_attempt_id"]) for row in counterparts} != a2_ids:
        raise ShadowCampaignPacketError("A2 subset lacks an exact frozen counterpart binding")

    families = {str(row["family_id"]) for row in execution}
    expected_families = {
        "A4_TSMOM_V7", "A1_COMPRESSION_V2", "A2_PRIOR_HIGH_RS_CONTEXT_V1",
        "A3_STARTER_RETEST_V3", "KDA02B_SURVIVOR_ADJUDICATION_V1",
    }
    if families != expected_families:
        raise ShadowCampaignPacketError("bounded shadow subset does not cover all five frozen families")
    for row in execution:
        if row.get("family_id") == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and row.get("config", {}).get("parent_binding_mode") == "source_attempt":
            if str(row.get("resolved_parent_executable_attempt_id")) not in attempt_ids:
                raise ShadowCampaignPacketError("A2 source-attempt parent is absent from the bounded subset")
    kda_cells: dict[str, set[str]] = {}
    for row in execution:
        if row.get("family_id") == "KDA02B_SURVIVOR_ADJUDICATION_V1":
            config = row["config"]
            kda_cells.setdefault(str(config["stage20_cell_id"]), set()).add(str(config["adjudication_variant"]))
    if len(kda_cells) != 1 or len(next(iter(kda_cells.values()), ())) != 11:
        raise ShadowCampaignPacketError("bounded shadow subset must preserve all 11 variants of one exact KDA02B cell")
    control_classes = {str(row["control_id"]) for row in controls}
    if len(control_classes) != 20:
        raise ShadowCampaignPacketError("bounded shadow subset must contain every one of the 20 frozen control classes")

    execution_authority = json.loads(execution_input_authority_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if len(cache_manifest.get("artifacts", ())) != 567:
        raise ShadowCampaignPacketError("shadow cache is not the reviewed 567-frame benchmark probe")

    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise ShadowCampaignPacketError("shadow packet output root is not empty")
    output_paths = {
        "strategy": output_root / "FINAL_REGISTERED_CONFIGURATION_REGISTRY.jsonl",
        "execution": output_root / "FINAL_EXECUTION_REGISTRY.jsonl",
        "controls": output_root / "FINAL_CONTROL_REGISTRY.jsonl",
        "counterparts": output_root / "A2_PARENT_COUNTERPART_REGISTRY.jsonl",
    }
    for name, rows in (("strategy", strategy), ("execution", execution), ("controls", controls), ("counterparts", counterparts)):
        atomic_write_jsonl(output_paths[name], rows)
    scenario_path = output_root / "SYNTHETIC_SCENARIO_MATRIX.json"
    if include_synthetic_scenario_matrix:
        atomic_write_json(scenario_path, _shadow_scenario_matrix(execution))

    source_records = {name: _file_record(path, f"full_frozen_{name}_registry") for name, path in source_paths.items()}
    subset_records = {name: _file_record(path, f"bounded_shadow_{name}_registry") for name, path in output_paths.items()}
    subset_authority = {
        "schema": SHADOW_PACKET_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "mode": "shadow_no_outcome",
        "source_registry_records": source_records,
        "source_registry_inventory_sha256": canonical_hash(source_records),
        "subset_registry_records": subset_records,
        "subset_registry_inventory_sha256": canonical_hash(subset_records),
        "selected_executable_attempt_ids": sorted(attempt_ids),
        "selected_control_attempt_ids": sorted(control_ids),
        "execution_rows": len(execution),
        "strategy_rows_preserving_selected_multiplicity": len(strategy),
        "control_rows": len(controls),
        "control_classes": sorted(control_classes),
        "families": sorted(families),
        "kda02b_cell": next(iter(kda_cells)),
        "kda02b_variants": sorted(next(iter(kda_cells.values()))),
        "cache_classification": "benchmark_probe_only",
        "cache_is_launch_input_authority": False,
        "economic_outcomes_authorized": False,
        "protected_outcomes_authorized": False,
        "capitalcom_payload_access": False,
    }
    subset_authority["authority_sha256"] = canonical_hash(subset_authority)
    subset_path = output_root / "SHADOW_SUBSET_AUTHORITY.json"
    atomic_write_json(subset_path, subset_authority)

    launch_record = _file_record(launch_population_authority_path, "complete_A1_A4_launch_population_authority")
    kda_record = _file_record(kda02b_population_manifest_path, "complete_KDA02B_launch_population_authority")
    manifest = {
        "schema": SHADOW_MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "mode": "shadow_no_outcome",
        "execution_input_authority": execution_authority,
        "primary_hashes": {
            "strategy_registry": subset_records["strategy"]["sha256"],
            "execution_registry": subset_records["execution"]["sha256"],
            "control_registry": subset_records["controls"]["sha256"],
            "a2_counterpart_registry": subset_records["counterparts"]["sha256"],
            "cache_authority_manifest": sha256_file(cache_manifest_path),
        },
        "launch_population_authority": launch_record,
        "kda02b_lazy_population_authority": kda_record,
        "shadow_subset_authority": _file_record(subset_path, "shadow_subset_authority"),
        "cache_role": "benchmark_probe_only",
        "cache_artifacts": 567,
        "cache_is_launch_input_authority": False,
        "synthetic_payoff_provider_required": True,
        "economic_outcomes_authorized": False,
        "protected_outcomes_authorized": False,
        "capitalcom_payload_access": False,
    }
    if include_synthetic_scenario_matrix:
        manifest["synthetic_scenario_matrix"] = _file_record(scenario_path, "fixed_synthetic_scenario_matrix")
    manifest_path = output_root / "SHADOW_CAMPAIGN_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    request = {
        "schema": "stage24_bounded_shadow_authorization_request_v1",
        "campaign_id": CAMPAIGN_ID,
        "authorization_requested": "execute_exact_bounded_stage24_shadow_no_outcome",
        "shadow_campaign_manifest_sha256": sha256_file(manifest_path),
        "economic_outcomes_authorized": False,
    }
    request_path = output_root / "SHADOW_AUTHORIZATION_REQUEST.json"
    atomic_write_json(request_path, request)
    approval = {
        "schema": "stage24_bounded_shadow_authorization_v1",
        "campaign_id": CAMPAIGN_ID,
        "approved": True,
        "authorization": "execute_exact_bounded_stage24_shadow_no_outcome",
        "shadow_campaign_manifest_sha256": sha256_file(manifest_path),
        "shadow_authorization_request_sha256": sha256_file(request_path),
        "economic_outcomes_authorized": False,
        "protected_outcomes_authorized": False,
        "capitalcom_payload_access": False,
    }
    approval_path = output_root / "SHADOW_EXTERNAL_AUTHORIZATION.json"
    atomic_write_json(approval_path, approval)
    packet = {
        "packet_root": str(output_root),
        "manifest": _file_record(manifest_path, "shadow_campaign_manifest"),
        "approval_request": _file_record(request_path, "shadow_authorization_request"),
        "external_authorization": _file_record(approval_path, "shadow_external_authorization"),
        "cache_manifest": _file_record(cache_manifest_path, "benchmark_probe_cache_manifest"),
        "execution_input_authority": _file_record(execution_input_authority_path, "execution_input_authority"),
        "launch_population_authority": launch_record,
        "kda02b_population_authority": kda_record,
        "subset_authority": _file_record(subset_path, "shadow_subset_authority"),
    }
    if include_synthetic_scenario_matrix:
        packet["synthetic_scenario_matrix"] = _file_record(scenario_path, "fixed_synthetic_scenario_matrix")
    return packet


def build_shadow_service_authority(
    *,
    spec_path: Path,
    packet: Mapping[str, Any],
    repository_root: Path,
    stage24_task_path: Path,
    reviewed_commit: str,
    run_root: Path,
    service_identity: str,
    workers: int = 1,
    heartbeat_seconds: int = 1,
    hold_after_health_seconds: float = 0.0,
    maximum_distinct_days_per_attempt_partition: int = 2,
    production_event_locator_sampling: bool = False,
    target_eligible_event_days_per_attempt_partition: int = 6,
    maximum_candidate_locators_per_attempt_partition: int = 1000,
    maximum_kda02b_eligible_records_per_cell_fold: int = 1,
    reused_evidence_authority_path: Path | None = None,
    continuation_task_path: Path | None = None,
) -> dict[str, Any]:
    if workers < 1 or workers > 4:
        raise ShadowCampaignPacketError("shadow worker count is outside the frozen one-to-four bound")
    if heartbeat_seconds < 1:
        raise ShadowCampaignPacketError("shadow heartbeat interval must be positive")
    repository_root = repository_root.resolve()
    task = _file_record(stage24_task_path, "stage24_task")
    if task["sha256"] != "9e546e9376408f97bc3bc2ef2862c06e746864eacbd9a5a70f0e71680eeeccdf":
        raise ShadowCampaignPacketError("Stage 24 task hash differs")
    current = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if current != reviewed_commit:
        raise ShadowCampaignPacketError("reviewed shadow commit is not the exact live worktree commit")
    bound_files = [dict(value) for key, value in packet.items() if key != "packet_root" and isinstance(value, Mapping)]
    identity_bindings = {
        "stage24_task_sha256": task["sha256"],
        "reviewed_commit": reviewed_commit,
        "shadow_manifest_sha256": packet["manifest"]["sha256"],
        "shadow_subset_authority_sha256": packet["subset_authority"]["sha256"],
        "benchmark_probe_cache_manifest_sha256": packet["cache_manifest"]["sha256"],
        "complete_launch_population_authority_sha256": packet["launch_population_authority"]["sha256"],
        "complete_kda02b_population_authority_sha256": packet["kda02b_population_authority"]["sha256"],
        "synthetic_provider_version": "stage24-real-orchestrator-shadow-v1",
    }
    continuation_task = None
    if continuation_task_path is not None:
        continuation_task = _file_record(continuation_task_path, "stage24_event_locator_continuation_task")
        if continuation_task["sha256"] != "6da86984b890314ea4422c8787d5c6de282342385c6505005358bac31e3493d3":
            raise ShadowCampaignPacketError("Stage 24 event-locator continuation task hash differs")
        identity_bindings["continuation_task_sha256"] = continuation_task["sha256"]
    reused_evidence = None
    if reused_evidence_authority_path is not None:
        reused_evidence = _file_record(reused_evidence_authority_path, "stage24_reused_shadow_evidence_authority")
        identity_bindings["reused_evidence_authority_sha256"] = reused_evidence["sha256"]
    population_slice_policy = ({
        "schema": "stage24_shadow_event_locator_policy_v2",
        "selection": "actual_production_enumerator_pre_entry_event_locators",
        "target_eligible_event_days_per_attempt_partition": target_eligible_event_days_per_attempt_partition,
        "maximum_candidate_locators_per_attempt_partition": maximum_candidate_locators_per_attempt_partition,
        "empty_attempt_partitions_preserved": True,
        "real_post_entry_values_used": False,
        "economic_values_used_for_selection": False,
        "synthetic_payoff_generated_after_locator_freeze": True,
        "full_launch_population_authority_preserved": True,
    } if production_event_locator_sampling else {
        "schema": "stage24_shadow_population_slice_policy_v1",
        "selection": "first_authority_order_locator_on_each_distinct_UTC_day",
        "maximum_distinct_days_per_attempt_partition": maximum_distinct_days_per_attempt_partition,
        "economic_values_used_for_selection": False,
        "benchmark_frame_values_used": False,
        "full_launch_population_authority_preserved": True,
    })
    BoundedShadowPopulationSchedule(object(), population_slice_policy)
    identity_bindings["population_slice_policy_sha256"] = canonical_hash(population_slice_policy)
    kda02b_slice_policy = {
        "schema": "stage24_shadow_kda02b_slice_policy_v1",
        "selection": "first_authority_order_eligible_record_per_outer_fold",
        "maximum_eligible_records_per_cell_fold": maximum_kda02b_eligible_records_per_cell_fold,
        "typed_unavailable_rows": "preserve_every_row_encountered_before_slice_completion",
        "economic_values_used_for_selection": False,
        "full_kda02b_population_authority_preserved": True,
    }
    BoundedShadowKDA02BAdapter(object(), kda02b_slice_policy)
    identity_bindings["kda02b_slice_policy_sha256"] = canonical_hash(kda02b_slice_policy)
    spec = {
        "schema": "stage24_shadow_service_spec_v2",
        "mode": "shadow_no_outcome",
        "repository_root": str(repository_root),
        "run_root": str(run_root.resolve()),
        "service_identity": service_identity,
        "workers": workers,
        "heartbeat_seconds": heartbeat_seconds,
        "hold_after_health_seconds": hold_after_health_seconds,
        "reviewed_commit": reviewed_commit,
        "stage24_task": task,
        "bound_files": bound_files,
        "shadow_campaign_packet": dict(packet),
        "identity_bindings": identity_bindings,
        "population_slice_policy": population_slice_policy,
        "kda02b_slice_policy": kda02b_slice_policy,
        "synthetic_provider_version": "stage24-real-orchestrator-shadow-v1",
        "payoff_provider": "ShadowPayoffProvider",
        "cache_classification": "benchmark_probe_only",
        "cache_is_launch_input_authority": False,
        "economic_outcomes_authorized": False,
        "protected_outcomes_authorized": False,
        "capitalcom_payload_access": False,
    }
    if continuation_task is not None:
        spec["continuation_task"] = continuation_task
    if reused_evidence is not None:
        spec["reused_evidence_authority"] = reused_evidence
    atomic_write_json(spec_path, spec)
    return spec


class ShadowCampaignAuthorization:
    """File-backed no-outcome authority with the interface used by the orchestrator."""

    def __init__(self, spec: Mapping[str, Any], repository_root: Path) -> None:
        packet = spec.get("shadow_campaign_packet")
        if not isinstance(packet, Mapping):
            raise ShadowCampaignPacketError("shadow campaign packet binding is absent")
        self.repository_root = repository_root
        self.manifest_path = Path(str(packet["manifest"]["path"]))
        self.approval_request_path = Path(str(packet["approval_request"]["path"]))
        self.external_approval_path = Path(str(packet["external_authorization"]["path"]))
        self.packet = dict(packet)

    @staticmethod
    def _require_record(record: Mapping[str, Any]) -> Path:
        path = Path(str(record.get("path", "")))
        if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise ShadowCampaignPacketError(f"shadow campaign authority bytes differ: {record.get('role')}")
        return path

    def require(self) -> dict[str, Any]:
        paths = {name: self._require_record(record) for name, record in self.packet.items() if name != "packet_root"}
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        request = json.loads(paths["approval_request"].read_text(encoding="utf-8"))
        approval = json.loads(paths["external_authorization"].read_text(encoding="utf-8"))
        if manifest.get("schema") != SHADOW_MANIFEST_SCHEMA or manifest.get("mode") != "shadow_no_outcome":
            raise ShadowCampaignPacketError("shadow campaign manifest schema/mode differs")
        manifest_hash = sha256_file(paths["manifest"]); request_hash = sha256_file(paths["approval_request"])
        if request.get("shadow_campaign_manifest_sha256") != manifest_hash:
            raise ShadowCampaignPacketError("shadow request does not bind the manifest")
        if (
            approval.get("authorization") != "execute_exact_bounded_stage24_shadow_no_outcome"
            or approval.get("shadow_campaign_manifest_sha256") != manifest_hash
            or approval.get("shadow_authorization_request_sha256") != request_hash
        ):
            raise ShadowCampaignPacketError("exact shadow authorization binding differs")
        for payload in (manifest, request, approval):
            if payload.get("economic_outcomes_authorized") is not False:
                raise ShadowCampaignPacketError("shadow packet broadens economic outcome authority")
        if (
            manifest.get("protected_outcomes_authorized") is not False
            or manifest.get("capitalcom_payload_access") is not False
            or manifest.get("cache_role") != "benchmark_probe_only"
            or manifest.get("cache_is_launch_input_authority") is not False
            or manifest.get("synthetic_payoff_provider_required") is not True
            or int(manifest.get("cache_artifacts", -1)) != 567
        ):
            raise ShadowCampaignPacketError("shadow manifest outcome firewall/cache classification differs")
        subset = json.loads(paths["subset_authority"].read_text(encoding="utf-8"))
        recorded_hash = subset.get("authority_sha256")
        if recorded_hash != canonical_hash({key: value for key, value in subset.items() if key != "authority_sha256"}):
            raise ShadowCampaignPacketError("shadow subset authority hash differs")
        if subset.get("cache_classification") != "benchmark_probe_only" or subset.get("cache_is_launch_input_authority") is not False:
            raise ShadowCampaignPacketError("shadow subset cache role differs")
        source_records = subset.get("source_registry_records", {})
        subset_records = subset.get("subset_registry_records", {})
        if (
            subset.get("source_registry_inventory_sha256") != canonical_hash(source_records)
            or subset.get("subset_registry_inventory_sha256") != canonical_hash(subset_records)
        ):
            raise ShadowCampaignPacketError("shadow source/subset registry inventory hash differs")
        for record in source_records.values():
            self._require_record(record)
        for name, key in (("strategy", "strategy_registry"), ("execution", "execution_registry"), ("controls", "control_registry"), ("counterparts", "a2_counterpart_registry")):
            record = subset_records[name]
            self._require_record(record)
            if record["sha256"] != manifest["primary_hashes"][key]:
                raise ShadowCampaignPacketError(f"shadow manifest registry binding differs: {name}")
            source_rows = _read_jsonl(Path(str(source_records[name]["path"])))
            selected_rows = _read_jsonl(Path(str(record["path"])))
            source_counts = Counter(canonical_hash(row) for row in source_rows)
            selected_counts = Counter(canonical_hash(row) for row in selected_rows)
            if any(count > source_counts[row_hash] for row_hash, count in selected_counts.items()):
                raise ShadowCampaignPacketError(f"shadow {name} registry contains a row absent from the exact frozen source")
        if sha256_file(paths["cache_manifest"]) != manifest["primary_hashes"]["cache_authority_manifest"]:
            raise ShadowCampaignPacketError("shadow cache binding differs")
        cache = json.loads(paths["cache_manifest"].read_text(encoding="utf-8"))
        if len(cache.get("artifacts", ())) != 567:
            raise ShadowCampaignPacketError("shadow cache no longer has the reviewed 567 benchmark-probe frames")
        actual_commit = subprocess.run(
            ["git", "-C", str(self.repository_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if not actual_commit:
            raise ShadowCampaignPacketError("live shadow repository commit is absent")
        return manifest


__all__ = [
    "SHADOW_MANIFEST_SCHEMA", "SHADOW_PACKET_SCHEMA", "ShadowCampaignAuthorization",
    "ShadowCampaignPacketError", "build_bounded_shadow_packet", "build_shadow_service_authority",
    "BoundedShadowKDA02BAdapter", "BoundedShadowPopulationSchedule", "select_bounded_shadow_identities",
]
