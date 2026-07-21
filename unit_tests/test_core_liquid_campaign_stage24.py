from __future__ import annotations

import json
import tempfile
import unittest
import weakref
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.core_liquid_campaign.cache import SemanticCacheWriter, _restore_metadata
from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash, pretty_json_bytes, sha256_file
from tools.core_liquid_campaign.a1_state import initial_state, transition
from tools.core_liquid_campaign.family_engines.common import EngineInputError, weak_percentile, weak_percentile_prevalidated_sorted
from tools.core_liquid_campaign.campaign import CampaignOrchestrator
from tools.core_liquid_campaign.controls import CONTROL_IDS, derive_control_inputs, execute_control
from tools.core_liquid_campaign.executor import CacheAuthority, dispatch_registered_attempt
from tools.core_liquid_campaign.engine_types import DailyBar, ExactPopulationTableView, ExactPopulationView
from tools.core_liquid_campaign.schema import CAMPAIGN_ID, baseline_config, economic_address, normalize_config
from tools.core_liquid_campaign.shadow_payoff import ShadowPayoffProvider
from tools.core_liquid_campaign.shadow_service import _write_shadow_bound_stop
from tools.core_liquid_campaign.synthetic import a1_frame, a3_frame, a4_frame, frame_for_family, with_source_authority
from tools.core_liquid_campaign.terminal import TerminalContractError, independent_terminal_recomputation, terminal_package, verify_terminal_inventory
from tools.core_liquid_campaign.runtime import LazySupervisor, ResourceLimits
from tools.core_liquid_campaign.runtime import detached_shadow_service_spec
from tools.core_liquid_campaign.production_readiness_gate import _a1_state_gate, _bounded_cold_warm_replay, _outer_benchmark_frames
from tools.core_liquid_campaign.production_inputs import _a2_proximity_feature_arrays, _thresholds
from tools.core_liquid_campaign.family_engines import a1_compression
from tools.core_liquid_campaign.production_population_tables import A1PopulationTableAuthority, _a3_symbol_events, _feature_arrays


class Stage24KnownDefectTests(unittest.TestCase):
    @staticmethod
    def _attempt(family: str, attempt_id: str) -> dict[str, object]:
        config = normalize_config(family, baseline_config(family))
        return {
            "campaign_id": CAMPAIGN_ID,
            "family_id": family,
            "config": config,
            "execution_disposition": "execute_once",
            "executable_attempt_id": attempt_id,
            "canonical_economic_address_sha256": economic_address(family, config)[1],
            "duplicate_of_executable_attempt_id": None,
        }

    @staticmethod
    def _control_result_signature(result: dict[str, object]) -> str:
        def serializable(value: object) -> object:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {str(key): serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            return value

        observations = result.get("observations", ())
        return canonical_hash(serializable({
            "status": result.get("status"),
            "event_ids": sorted(item.event_id for item in observations),
            "ledger": result.get("ledger", ()),
            "aggregate": result.get("aggregate", {}),
            "allocation_unavailable": result.get("allocation_unavailable", ()),
        }))

    def test_prevalidated_percentile_is_exactly_equivalent(self) -> None:
        population = tuple(float(index % 37) for index in range(100))
        ordered = tuple(sorted(population))
        for value in (-1.0, 0.0, 12.5, 36.0, 99.0):
            self.assertEqual(weak_percentile(value, population), weak_percentile_prevalidated_sorted(value, ordered))

    def test_hash_bound_external_population_preserves_exact_percentiles_and_type7(self) -> None:
        import numpy as np
        from tools.core_liquid_campaign.family_engines.common import type7_quantile

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "populations/exact.npy"
            path.parent.mkdir(parents=True)
            values = np.asarray(sorted(float((index * 17) % 101) + index / 1000 for index in range(200)), dtype="<f8")
            np.save(path, values, allow_pickle=False)
            view = ExactPopulationView(
                "populations/exact.npy", sha256_file(path), len(values), 11, 177,
                len(set(float(value) for value in values[11:177])), str(root),
            )
            view.validate_physical()
            selected = tuple(float(value) for value in values[11:177])
            for value in (-1.0, 12.5, 1000.0):
                self.assertEqual(weak_percentile(value, selected), weak_percentile(value, view))
            for probability in (0.0, 0.2, 0.5, 0.95, 1.0):
                self.assertEqual(type7_quantile(selected, probability), type7_quantile(view, probability))
            path.write_bytes(path.read_bytes()[:-1] + b"x")
            with self.assertRaises(EngineInputError):
                view.validate_physical()

    def test_external_population_round_trips_through_cache_authority(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); cache_root = root / "cache"
            source = root / "source.json"; atomic_write_json(source, {"fixture": True})
            source_record = {"role": "fixture", "path": "source.json", "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            authority = {
                "platform": "kraken_native_linear_pf", "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
                "source_manifest_sha256": source_record["sha256"], "pit_universe_sha256": "b" * 64,
                "funding_manifest_sha256": "c" * 64, "cache_contract_sha256": "d" * 64,
                "fold_graph_sha256": "e" * 64, "rankable_funding_package_sha256": "f" * 64,
                "source_records": [source_record], "cache_manifest_contract": {"schema": "stage22_semantic_cache_manifest_v1"},
            }
            external_path = cache_root / "populations/exact.npy"; external_path.parent.mkdir(parents=True)
            values = np.asarray([float(index) for index in range(40)], dtype="<f8")
            np.save(external_path, values, allow_pickle=False)
            frame = a4_frame()
            name, population = next(iter(frame.threshold_populations.items()))
            external = ExactPopulationView(
                "populations/exact.npy", sha256_file(external_path), 40, 0, 40, 40, str(cache_root),
            )
            populations = {**frame.threshold_populations, name: replace(population, values=external)}
            partition = {
                "phase": "outer_evaluation", "outer_fold_id": "2025Q2", "inner_fold_id": None,
                "training_start": population.training_start, "training_end_exclusive": population.training_end_exclusive,
                "evaluation_start": frame.metadata["evaluation_start"], "evaluation_end_exclusive": frame.metadata["evaluation_end_exclusive"],
            }
            frame = replace(frame, threshold_populations=populations, metadata={**frame.metadata, "campaign_partition": partition})
            frame = with_source_authority(frame, authority)
            writer = SemanticCacheWriter(cache_root, authority, authority_root=root)
            record = writer.add(frame); manifest_path = writer.finalize()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("npy_float64_le_sorted_v1", {row["encoding"] for row in manifest["components"]})
            cache = CacheAuthority(manifest_path, cache_root)
            _, preloaded = cache.preload_frames({"execution_input_authority": authority})
            self.assertEqual((), preloaded)
            self.assertEqual(0, len(cache._decoded_frames))
            _, decoded = cache.load_frames({"execution_input_authority": authority}, [record["path"]])
            restored = decoded[0].threshold_populations[name].values
            self.assertIsInstance(restored, ExactPopulationView)
            self.assertEqual(weak_percentile(19.0, external), weak_percentile(19.0, restored))

    def test_shared_population_table_applies_exact_fold_symbol_and_decile_selectors(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); component_root = root / "populations"; component_root.mkdir()
            arrays = {
                "values.npy": np.asarray([float(index) for index in range(120)], dtype="<f8"),
                "timestamps.npy": np.asarray([100] * 40 + [200] * 40 + [300] * 40, dtype="<i8"),
                "symbols.npy": np.asarray([1 + index % 2 for index in range(120)], dtype="<u2"),
                "deciles.npy": np.asarray([4 if index % 5 == 0 else 3 for index in range(120)], dtype="u1"),
            }
            for name, array in arrays.items():
                np.save(component_root / name, array, allow_pickle=False)
            selected = [
                float(value) for value, timestamp, symbol, decile in zip(*arrays.values())
                if 100 <= timestamp < 300 and symbol == 2 and decile == 3
            ]
            view = ExactPopulationTableView(
                values_path="populations/values.npy", values_sha256=sha256_file(component_root / "values.npy"),
                timestamps_path="populations/timestamps.npy", timestamps_sha256=sha256_file(component_root / "timestamps.npy"),
                symbols_path="populations/symbols.npy", symbols_sha256=sha256_file(component_root / "symbols.npy"),
                deciles_path="populations/deciles.npy", deciles_sha256=sha256_file(component_root / "deciles.npy"),
                physical_count=120, training_start_ms=100, training_end_ms=300,
                selected_count=len(selected), unique_count=len(set(selected)), minimum_unique_count_verified=20,
                symbol_code=2, liquidity_decile=3, root=str(root),
            )
            view.validate_physical()
            self.assertEqual(sorted(selected), list(view))
            self.assertEqual(weak_percentile(2.5, selected), weak_percentile(2.5, view))

    def test_a1_persistent_state_covers_owner_gap_cooldown_and_strict_rearm(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state = initial_state()
        state = transition(state, timestamp=start, action="history_complete")
        with self.assertRaises(EngineInputError):
            transition(state, timestamp=start, action="rearm", percentiles={1: 0.49, -1: 0.49})
        state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.49, -1: 0.49})
        state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=-1)
        state = transition(state, timestamp=start + timedelta(minutes=15), action="base")
        state = transition(state, timestamp=start + timedelta(minutes=20), action="confirmation")
        state = transition(state, timestamp=start + timedelta(minutes=25), action="gap")
        self.assertEqual((state.state, state.owner, state.terminal_episode_reason), ("history_rebuild", -1, "temporal_gap"))
        state = transition(state, timestamp=start + timedelta(days=1), action="history_complete")
        state = transition(state, timestamp=start + timedelta(days=1, minutes=5), action="rearm", percentiles={-1: 0.50, 1: 0.1})
        self.assertEqual("disarmed", state.state)
        state = transition(state, timestamp=start + timedelta(days=1, minutes=10), action="rearm", percentiles={-1: 0.49, 1: 0.9})
        self.assertEqual("armed", state.state)
        self.assertIsNone(state.owner)

    def test_a1_persistent_state_round_trips_cooldown(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        state = transition(initial_state(), timestamp=start, action="history_complete")
        state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.1, -1: 0.1})
        state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=1)
        state = transition(
            state,
            timestamp=start + timedelta(minutes=15),
            action="episode_terminal",
            terminal_reason="actual_exit",
            cooldown_until=start + timedelta(hours=1),
        )
        state = transition(state, timestamp=start + timedelta(minutes=20), action="rearm", percentiles={1: 0.1})
        self.assertEqual("cooldown", state.state)
        state = transition(state, timestamp=start + timedelta(hours=1), action="cooldown_expired")
        self.assertEqual("disarmed", state.state)
        self.assertEqual(6, state.state_generation)

    def test_a1_gap_at_each_active_state_preserves_owner_and_reason(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for side in (-1, 1):
            state = transition(initial_state(), timestamp=start, action="history_complete")
            state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.1, -1: 0.1})
            state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=side)
            active = [state]
            active.append(transition(active[-1], timestamp=start + timedelta(minutes=15), action="base"))
            active.append(transition(active[-1], timestamp=start + timedelta(minutes=20), action="confirmation"))
            for offset, candidate in enumerate(active, start=3):
                gapped = transition(candidate, timestamp=start + timedelta(hours=1, minutes=offset * 5), action="gap")
                self.assertEqual(("history_rebuild", side, "temporal_gap"), (gapped.state, gapped.owner, gapped.terminal_episode_reason))

    def test_production_a1_engine_consumes_persisted_start_state_and_fails_closed_mid_episode(self) -> None:
        frame = a1_frame()
        metadata = {**frame.metadata, "production_input": True, "a1_persistent_state": initial_state().payload()}
        a1_frame_from_rebuild = replace(frame, metadata=metadata)
        dispatch_registered_attempt(
            self._attempt("A1_COMPRESSION_V2", "persisted-a1"),
            (a1_frame_from_rebuild,),
            registry_by_id={"persisted-a1": self._attempt("A1_COMPRESSION_V2", "persisted-a1")},
            payoff_provider=ShadowPayoffProvider("stage24-a1-persisted-state"),
        )
        start = frame.five_minute_bars[0].open_ts - timedelta(minutes=15)
        state = transition(initial_state(), timestamp=start, action="history_complete")
        state = transition(state, timestamp=start + timedelta(minutes=5), action="rearm", percentiles={1: 0.1, -1: 0.1})
        state = transition(state, timestamp=start + timedelta(minutes=10), action="trigger", side=1)
        broken = replace(frame, metadata={**metadata, "a1_persistent_state": state.payload()})
        with self.assertRaises(EngineInputError):
            dispatch_registered_attempt(
                self._attempt("A1_COMPRESSION_V2", "persisted-a1"),
                (broken,), registry_by_id={"persisted-a1": self._attempt("A1_COMPRESSION_V2", "persisted-a1")},
                payoff_provider=ShadowPayoffProvider("stage24-a1-persisted-state"),
            )

    def test_production_a1_checkpoint_is_order_and_restart_invariant(self) -> None:
        row = self._attempt("A1_COMPRESSION_V2", "persisted-a1-replay")
        start = datetime(2025, 5, 1, tzinfo=timezone.utc)
        frames = []
        for anchor in (start, start + timedelta(days=14)):
            frame = a1_frame(row["config"], anchor=anchor)
            frames.append(replace(frame, metadata={
                **frame.metadata,
                "production_input": True,
                "a1_persistent_state": initial_state().payload(),
            }))
        forward = dispatch_registered_attempt(
            row, frames, registry_by_id={row["executable_attempt_id"]: row},
            payoff_provider=ShadowPayoffProvider("stage24-a1-checkpoint"),
        )
        reverse = dispatch_registered_attempt(
            row, list(reversed(frames)), registry_by_id={row["executable_attempt_id"]: row},
            payoff_provider=ShadowPayoffProvider("stage24-a1-checkpoint"),
        )
        self.assertEqual(forward["a1_persistent_state_checkpoints"], reverse["a1_persistent_state_checkpoints"])
        first = dispatch_registered_attempt(
            row, frames[:1], registry_by_id={row["executable_attempt_id"]: row},
            payoff_provider=ShadowPayoffProvider("stage24-a1-checkpoint"),
        )
        first_checkpoint = first["a1_persistent_state_checkpoints"][frames[0].content_sha256()]
        resumed_frame = replace(frames[1], metadata={**frames[1].metadata, "a1_persistent_state": first_checkpoint})
        resumed = dispatch_registered_attempt(
            row, [resumed_frame], registry_by_id={row["executable_attempt_id"]: row},
            payoff_provider=ShadowPayoffProvider("stage24-a1-checkpoint"),
        )
        combined_final = forward["a1_persistent_state_checkpoints"][frames[1].content_sha256()]
        resumed_final = next(iter(resumed["a1_persistent_state_checkpoints"].values()))
        self.assertEqual(combined_final, resumed_final)
        self.assertEqual(frames[1].decision_ts, combined_final["last_valid_ts"])

    def test_production_gate_a1_evidence_is_canonical_json_serializable(self) -> None:
        evidence = _a1_state_gate()
        self.assertEqual("pass", evidence["status"])
        payload = pretty_json_bytes(evidence)
        self.assertIn(b'"last_valid_ts": "2025-01-01T00:25:00+00:00"', payload)

    def test_full_cache_warm_replay_releases_cold_frames_first(self) -> None:
        class Frame:
            metadata: dict[str, object] = {}

        cold_refs: list[weakref.ReferenceType[Frame]] = []
        factory_calls = {"count": 0}

        class Cache:
            def __init__(self, generation: int) -> None:
                self.generation = generation

            def load_frames(self, _manifest: object, paths: list[str]) -> tuple[dict[str, object], tuple[Frame, ...]]:
                frames = tuple(Frame() for _ in paths)
                if self.generation == 1:
                    cold_refs.extend(weakref.ref(frame) for frame in frames)
                return {}, frames

        def factory() -> Cache:
            factory_calls["count"] += 1
            if factory_calls["count"] == 2 and any(reference() is not None for reference in cold_refs):
                raise AssertionError("cold cache frames remained live during warm replay")
            return Cache(factory_calls["count"])

        records = [
            {"path": f"frame-{index}", "campaign_partition": {"phase": "inner_validation" if index < 3 else "outer_evaluation", "outer_fold_id": "2024Q1", "inner_fold_id": f"M_{index // 3}"}}
            for index in range(6)
        ]
        cold, warm, frames, protected = _bounded_cold_warm_replay(factory, {}, records)
        self.assertEqual((2, 2, 0), (factory_calls["count"], len(frames), protected))
        self.assertGreaterEqual(cold, 0.0)
        self.assertGreaterEqual(warm, 0.0)

    def test_benchmark_releases_inner_frames_before_fork(self) -> None:
        class Frame:
            def __init__(self, phase: str, fold: str) -> None:
                self.metadata = {"campaign_partition": {"phase": phase, "outer_fold_id": fold}}

        frames = tuple(
            Frame(phase, fold)
            for fold in ("2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4")
            for phase in ("inner_validation", "outer_evaluation")
        )
        selected = _outer_benchmark_frames(frames)
        self.assertEqual(8, len(selected))
        self.assertTrue(all(frame.metadata["campaign_partition"]["phase"] == "outer_evaluation" for frame in selected))

    def test_cache_restores_a1_cooldown_deadline_as_utc_datetime(self) -> None:
        value = _restore_metadata({"cooldown_until": "2025-01-01T01:00:00+00:00"})
        self.assertEqual(datetime(2025, 1, 1, 1, tzinfo=timezone.utc), value["cooldown_until"])

    def test_typed_kda_cache_unavailability_becomes_terminal_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.json"
            atomic_write_json(source, {"fixture": True})
            source_record = {"role": "fixture", "path": "source.json", "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            authority = {
                "platform": "kraken_native_linear_pf",
                "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
                "source_manifest_sha256": source_record["sha256"], "pit_universe_sha256": "b" * 64,
                "funding_manifest_sha256": "c" * 64, "cache_contract_sha256": "d" * 64,
                "fold_graph_sha256": "e" * 64, "rankable_funding_package_sha256": "f" * 64,
                "source_records": [source_record], "cache_manifest_contract": {"schema": "stage22_semantic_cache_manifest_v1"},
            }
            writer = SemanticCacheWriter(root / "cache", authority, authority_root=root, synthetic_only=False)
            partition = {
                "phase": "outer_evaluation", "outer_fold_id": "2024Q1", "inner_fold_id": None,
                "training_start": datetime(2023, 1, 1, tzinfo=timezone.utc),
                "training_end_exclusive": datetime(2023, 12, 22, tzinfo=timezone.utc),
                "evaluation_start": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "evaluation_end_exclusive": datetime(2024, 4, 1, tzinfo=timezone.utc),
            }
            record = writer.add_unavailable(
                family_id="KDA02B_SURVIVOR_ADJUDICATION_V1", partition=partition,
                reason="exact raw decision fields unavailable", authority_sha256="a" * 64,
            )
            cache = {"artifacts": [], "typed_unavailable": [record]}
            row = self._attempt("KDA02B_SURVIVOR_ADJUDICATION_V1", "kda-row")
            orchestrator = object.__new__(CampaignOrchestrator)
            jobs = list(orchestrator._kda_jobs([row], {"kda-row": row}, cache))
            self.assertEqual(1, len(jobs))
            result = jobs[0][1]()
            self.assertEqual(("unavailable_data", "explicit_empty_unavailable_observation"), (result["status"], result["materialization"]))
            self.assertEqual("a" * 64, result["authority_sha256"])

    def test_early_inner_fold_persists_long_a4_features_as_unavailable(self) -> None:
        frame = a4_frame()
        bars = frame.five_minute_bars
        daily = frame.daily_bars
        populations, unavailable = _thresholds(
            {"PF_XBTUSD": bars, "PF_ETHUSD": bars},
            {"PF_XBTUSD": daily, "PF_ETHUSD": daily},
            target="PF_XBTUSD",
            training_start=bars[0].open_ts,
            training_end=frame.decision_ts,
        )
        name = "A4_ensemble:ema_slope:lookback=180:volatility=close_to_close"
        self.assertNotIn(name, populations)
        self.assertIn(name, {row["feature_signature"] for row in unavailable})

    def test_production_thresholds_enumerate_every_boundary_and_match_a1_feature_formula(self) -> None:
        frame = a1_frame()
        bars = tuple(bar for bar in frame.five_minute_bars if bar.close_ts < frame.decision_ts)
        populations, _ = _thresholds(
            {"PF_XBTUSD": bars, "PF_ETHUSD": bars}, {"PF_XBTUSD": frame.daily_bars, "PF_ETHUSD": frame.daily_bars}, target="PF_XBTUSD",
            training_start=bars[0].open_ts, training_end=frame.decision_ts,
        )
        impulse = populations[a1_compression.impulse_population_key("6h", "symbol_side", 1)]
        self.assertEqual(len(bars) - 72, len(impulse.values))
        expected = []
        for index in range(47, len(bars)):
            baseline = [bar.close for bar in bars[index - 47:index - 23]]
            base = [bar.close for bar in bars[index - 23:index + 1]]
            try:
                expected.append(a1_compression.features(base, base, baseline, 1)["contraction_ratio"])
            except EngineInputError:
                pass
        actual = populations[a1_compression.contraction_population_key("2h", "adjacent_equal_duration", "symbol")].values
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertAlmostEqual(left, right, places=12)

    def test_columnar_a1_features_match_engine_formula_and_fail_closed_at_gap(self) -> None:
        import numpy as np

        size = 400
        times = np.arange(size, dtype="<i8") * 300_000 + 1_735_689_600_000
        closes = 100.0 * np.exp(np.sin(np.arange(size) / 13.0) * 0.01 + np.arange(size) * 0.0001)
        arrays = _feature_arrays(times, closes)
        index = 300
        expected = a1_compression.features(
            closes[index - 72:index + 1], closes[index - 23:index + 1], closes[index - 47:index - 23], 1,
        )
        self.assertAlmostEqual(expected["side_signed_impulse"], arrays["A1_impulse:window=6h"][index], places=14)
        self.assertAlmostEqual(expected["contraction_ratio"], arrays["A1_contraction:base=2h:baseline=adjacent_equal_duration"][index], places=12)
        self.assertAlmostEqual(expected["base_smoothness"], arrays["A1_smoothness:base=2h"][index], places=13)
        gapped_times = times.copy(); gapped_times[250:] += 300_000
        gapped = _feature_arrays(gapped_times, closes)
        self.assertTrue(np.isnan(gapped["A1_impulse:window=6h"][300]))
        self.assertTrue(np.isnan(gapped["A1_contraction:base=2h:baseline=adjacent_equal_duration"][270]))

    def test_sparse_a3_population_records_exact_first_pit_crossing_and_rejects_gap(self) -> None:
        import numpy as np

        day = datetime(2025, 1, 1, tzinfo=timezone.utc)
        daily = tuple(
            DailyBar(
                day - timedelta(days=299 - index),
                100.0, 110.0, 90.0, 100.0,
                day - timedelta(days=299 - index),
                day - timedelta(days=299 - index),
                True,
            )
            for index in range(300)
        )
        day_ms = int(day.timestamp() * 1000)
        times = np.asarray([day_ms, day_ms + 300_000, day_ms + 600_000], dtype="<i8")
        closes = np.asarray([109.0, 111.0, 112.0], dtype="<f8")
        pit = {day_ms: {"average_liquidity_rank": 3.0, "eligible_population": 20}}
        events = _a3_symbol_events(times, closes, daily, pit)
        for lookback in (20, 60, 120, 250):
            for atr in (10, 20, 40, 60):
                self.assertEqual(
                    [(day_ms + 600_000, 2, 0.05)],
                    events[f"A3_breakout:lookback={lookback}:atr={atr}:side=1"],
                )
                self.assertEqual([], events[f"A3_breakout:lookback={lookback}:atr={atr}:side=-1"])

        gapped_times = times.copy()
        gapped_times[1:] += 300_000
        gapped = _a3_symbol_events(gapped_times, closes, daily, pit)
        self.assertTrue(all(not rows for rows in gapped.values()))

    def test_vectorized_a2_proximity_matches_exact_level_and_wilder_atr(self) -> None:
        from tools.core_liquid_campaign.family_engines import a2_context
        from tools.core_liquid_campaign.family_engines.common import wilder_atr

        frame = a3_frame()
        times, arrays = _a2_proximity_feature_arrays(frame.five_minute_bars, frame.daily_bars)
        name = a2_context.proximity_population_key(20, 10, 1)
        finite = [index for index, value in enumerate(arrays[name]) if __import__("math").isfinite(float(value))]
        self.assertTrue(finite)
        index = finite[-1]
        day = datetime.fromtimestamp(int(times[index]) / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        daily_index = next(i for i, row in enumerate(frame.daily_bars) if row.close_ts == day)
        prior = frame.daily_bars[daily_index - 19:daily_index + 1]
        atr_rows = frame.daily_bars[daily_index - 10:daily_index + 1]
        atr = wilder_atr(
            [row.high for row in atr_rows], [row.low for row in atr_rows], [row.close for row in atr_rows], 10,
        )
        level = max(row.high for row in prior)
        expected = (frame.five_minute_bars[index].close - level) / atr
        self.assertAlmostEqual(expected, float(arrays[name][index]), places=14)

    def test_a1_population_authority_resolves_global_and_signed_views(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); table_root = root / "population_tables/a1"; table_root.mkdir(parents=True)
            day0 = 1_672_531_200_000
            values = np.asarray([float(index + 1) for index in range(120)], dtype="<f8")
            timestamps = np.asarray([day0 + 86_400_000] * 60 + [day0 + 2 * 86_400_000] * 60, dtype="<i8")
            symbols = np.asarray([1 + index % 5 for index in range(120)], dtype="<u2")
            deciles = np.asarray([1] * 120, dtype="u1")
            paths = {}
            for name, array in (("values", values), ("timestamps", timestamps), ("symbols", symbols), ("deciles", deciles)):
                path = table_root / f"{name}.npy"; np.save(path, array, allow_pickle=False); paths[name] = path
            counts = np.zeros((3, 16), dtype="<i4"); counts[1:, 0] = 60; counts[1:, 1] = 60
            for code in range(1, 6): counts[1:, 10 + code] = 12
            count_path = table_root / "counts.npy"; np.save(count_path, counts, allow_pickle=False)
            def record(path: Path) -> dict[str, object]:
                return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": 120}
            feature = {**record(paths["values"]), "daily_counts_path": count_path.relative_to(root).as_posix(), "daily_counts_bytes": count_path.stat().st_size, "daily_counts_sha256": sha256_file(count_path)}
            manifest = {
                "schema": "stage24_a1_exact_pit_population_table_v1", "protected_rows": 0,
                "rows": 120, "symbol_codes": {f"S{code}": code for code in range(1, 6)},
                "common": {name: record(paths[name]) for name in ("timestamps", "symbols", "deciles")},
                "features": {"A1_impulse:window=6h": feature},
                "daily_count_rankable_start_day_ms": day0, "daily_count_columns": {
                    "global": 0, "liquidity_deciles": {str(value): value for value in range(1, 11)},
                    "symbols": {f"S{code}": 10 + code for code in range(1, 6)},
                },
            }
            manifest_path = table_root / "A1_POPULATION_TABLE_MANIFEST.json"; atomic_write_json(manifest_path, manifest)
            authority = A1PopulationTableAuthority(root, manifest_path)
            start = datetime(2023, 1, 1, tzinfo=timezone.utc); end = datetime(2023, 1, 4, tzinfo=timezone.utc)
            positive = authority.population(
                "A1_impulse:window=6h:scope=global_side:side=1", target_symbol="S1", target_decile=1,
                training_start=start, training_end=end,
            )
            negative = authority.population(
                "A1_impulse:window=6h:scope=global_side:side=-1", target_symbol="S1", target_decile=1,
                training_start=start, training_end=end,
            )
            positive.validate(pooled=True, decision_ts=end)
            negative.validate(pooled=True, decision_ts=end)
            self.assertEqual(120, len(positive.values)); self.assertEqual(120, len(negative.values))
            self.assertEqual(-120.0, negative.values[0]); self.assertEqual(1.0, positive.values[0])

    def test_shadow_provider_uses_actual_accounting_without_real_post_entry_data(self) -> None:
        config = baseline_config("A4_TSMOM_V7")
        _, address = economic_address("A4_TSMOM_V7", config)
        attempt_id = "stage24-shadow-a4"
        row = {
            "campaign_id": CAMPAIGN_ID,
            "family_id": "A4_TSMOM_V7",
            "config": config,
            "execution_disposition": "execute_once",
            "executable_attempt_id": attempt_id,
            "canonical_economic_address_sha256": address,
            "duplicate_of_executable_attempt_id": None,
        }
        provider = ShadowPayoffProvider("stage24-test")
        result = dispatch_registered_attempt(
            row,
            (a4_frame(config),),
            registry_by_id={attempt_id: row},
            payoff_provider=provider,
        )
        self.assertEqual("complete", result["status"])
        self.assertGreater(provider.calls, 0)
        self.assertFalse(provider.attestation()["economic_outcomes_opened"])
        self.assertTrue(all(item["shadow_only"] for item in result["ledger"]))
        self.assertTrue(all(item["actual_accounting_path_executed"] for item in result["ledger"]))
        self.assertTrue(all(item["synthetic_funding_rows"] == 242 for item in result["ledger"]))
        self.assertTrue(all(item["real_post_entry_rows_opened"] == 0 for item in result["ledger"]))

    def test_every_deterministic_control_receives_parent_frames(self) -> None:
        for family, controls in CONTROL_IDS.items():
            engine_family = family
            config = baseline_config(engine_family)
            parent = {
                "family_id": engine_family,
                "config": config,
                "executable_attempt_id": f"parent-{family}",
                "canonical_economic_address_sha256": economic_address(engine_family, config)[1],
            }
            frames = [frame_for_family(engine_family, config)]
            for control_id in controls:
                if control_id in {
                    "A4_SIGN_PERMUTED_MAIN_NULL", "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL",
                    "A2_CONTEXT_PERMUTED_MAIN_NULL", "A3_RETEST_TIME_PERMUTED_MAIN_NULL",
                    "A3_MATCHED_PSEUDO_EVENT",
                }:
                    continue
                control = {"control_id": control_id, "effective_seed": 1, "economic_address_sha256": "a" * 64}
                transformed, directives, unavailable = derive_control_inputs(
                    control, parent, {"observations": [], "ledger": []}, frames,
                )
                self.assertEqual(transformed, frames, control_id)
                self.assertEqual(directives, {}, control_id)
                self.assertEqual(unavailable, [], control_id)

    def test_all_twenty_controls_execute_nonempty_and_replay_invariant(self) -> None:
        utc = timezone.utc
        anchors = (datetime(2025, 6, 1, tzinfo=utc), datetime(2025, 6, 15, tzinfo=utc))
        fixtures: dict[str, tuple[dict[str, object], list[object], dict[str, dict[str, object]], dict[str, object] | None]] = {}
        a4 = self._attempt("A4_TSMOM_V7", "parent-a4")
        fixtures["A4_TSMOM_V7"] = (
            a4,
            [a4_frame(a4["config"], signal_sign=1, anchor=anchors[0]), a4_frame(a4["config"], signal_sign=-1, anchor=anchors[1])],
            {"parent-a4": a4},
            None,
        )
        a1 = self._attempt("A1_COMPRESSION_V2", "parent-a1")
        fixtures["A1_COMPRESSION_V2"] = (
            a1,
            [a1_frame(a1["config"], anchor=anchors[0]), a1_frame(a1["config"], anchor=anchors[1])],
            {"parent-a1": a1},
            None,
        )
        a3 = self._attempt("A3_STARTER_RETEST_V3", "parent-a3")
        fixtures["A3_STARTER_RETEST_V3"] = (
            a3,
            [a3_frame(a3["config"], anchor=anchors[0]), a3_frame(a3["config"], anchor=anchors[1])],
            {"parent-a3": a3},
            None,
        )
        a2_parent = self._attempt("A1_COMPRESSION_V2", "a2-parent")
        a2_config = normalize_config("A2_PRIOR_HIGH_RS_CONTEXT_V1", baseline_config("A2_PRIOR_HIGH_RS_CONTEXT_V1"))
        template = canonical_hash({"mode": "beam_slot", "parent_slot": "A1_COMPRESSION_V2:2024Q1:beam:01"})
        a2 = {
            "campaign_id": CAMPAIGN_ID, "family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1",
            "config": a2_config, "execution_disposition": "execute_if_parent_available",
            "executable_attempt_id": "parent-a2", "canonical_economic_address_sha256": economic_address("A2_PRIOR_HIGH_RS_CONTEXT_V1", a2_config)[1],
            "duplicate_of_executable_attempt_id": None, "parent_binding_template_id": template,
            "parent_only_counterpart_id": "parent-only", "overlay_counterpart_id": "overlay",
        }
        first = a1_frame(a2_parent["config"], anchor=anchors[0])
        second = a1_frame(a2_parent["config"], anchor=anchors[1])
        by_lookback = {key: dict(value) for key, value in second.context.cross_section_returns_by_lookback.items()}
        by_lookback[20][second.symbol] = -0.08
        second_context = replace(
            second.context,
            cross_section_returns=dict(by_lookback[20]),
            cross_section_returns_by_lookback=by_lookback,
            source_sha256=canonical_hash({"stage24_control_fixture": "second_context"}),
        )
        second = replace(second, context=second_context)
        binding = {
            "parent_binding_template_id": template, "parent_executable_attempt_id": "a2-parent",
            "parent_only_counterpart_id": "parent-only", "overlay_counterpart_id": "overlay",
        }
        fixtures["A2_PRIOR_HIGH_RS_CONTEXT_V1"] = (a2, [first, second], {"parent-a2": a2, "a2-parent": a2_parent}, binding)

        executed = []
        for family, control_ids in CONTROL_IDS.items():
            parent, frames, registry, binding = fixtures[family]
            for control_id in control_ids:
                seed = 3 if control_id in {"A4_SIGN_PERMUTED_MAIN_NULL", "A2_CONTEXT_PERMUTED_MAIN_NULL", "A3_RETEST_TIME_PERMUTED_MAIN_NULL"} else 1
                control = {
                    "family": family, "control_id": control_id, "effective_seed": seed,
                    "economic_address_sha256": canonical_hash({"control": control_id}),
                    "control_attempt_id": f"stage24-{control_id}", "execution_status": "execute_once",
                }
                kwargs = {
                    "registry_by_id": registry,
                    "payoff_provider": ShadowPayoffProvider("stage24-all-controls"),
                }
                if binding is not None:
                    kwargs.update({"parent_binding": binding, "parent_frames": frames})
                forward = execute_control(control, parent, frames, **kwargs)
                reverse = execute_control(control, parent, list(reversed(frames)), **kwargs)
                self.assertEqual("complete", forward["status"], control_id)
                self.assertGreater(len(forward["observations"]), 0, control_id)
                self.assertEqual(self._control_result_signature(forward), self._control_result_signature(reverse), control_id)
                executed.append(control_id)
        self.assertEqual(20, len(executed))
        self.assertEqual(20, len(set(executed)))

    def test_missing_a2_parent_is_an_explicit_empty_fold_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage = root / "inner_development"
            artifact = stage / "artifacts/result.json"
            payload = {
                "status": "unavailable_no_parent",
                "registered_attempt_id": "a2",
                "registered_job_id": "inner:2024Q1:M_202307:a2",
                "aggregate": {}, "observation_count": 0,
                "day_base_net_bps": {}, "event_ids": [],
            }
            atomic_write_json(artifact, {"result": payload})
            atomic_write_json(stage / "markers/marker.json", {
                "artifact": "artifacts/result.json", "artifact_sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
            })
            config = baseline_config("A2_PRIOR_HIGH_RS_CONTEXT_V1")
            row = {
                "family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "config": config,
                "executable_attempt_id": "a2", "canonical_economic_address_sha256": economic_address("A2_PRIOR_HIGH_RS_CONTEXT_V1", config)[1],
                "selection_role": "conditional_parent_overlay_template",
            }
            orchestrator = object.__new__(CampaignOrchestrator)
            orchestrator.run_root = root
            self.assertEqual(orchestrator._freeze_beams([row]), [])

    def test_completed_terminal_requires_forensics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(TerminalContractError):
                terminal_package(
                    Path(raw), attempt_ids=["a"], control_ids=[],
                    attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}], control_rows=[],
                    routes=[{"family": "A4_TSMOM_V7", "route": "translation_rejected"}],
                    forensics=[], all_workers_stopped=True,
                    job_reconciliation={"pass": True},
                )

    def test_terminal_inventory_is_last_and_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            terminal_package(
                root,
                attempt_ids=["a"], control_ids=["c"],
                attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}],
                control_rows=[{"control_attempt_id": "c", "terminal_status": "completed"}],
                routes=[{"family": "A4_TSMOM_V7", "route": "translation_rejected"}],
                forensics=[{"family": "A4_TSMOM_V7", "event_count": 0}],
                    all_workers_stopped=True,
                    job_reconciliation={"pass": True},
                )
            self.assertEqual("pass", verify_terminal_inventory(root)["status"])
            (root / "FORENSIC_RECORDS.json").write_bytes(b"tampered")
            with self.assertRaises(TerminalContractError):
                verify_terminal_inventory(root)

    def test_terminal_independent_recomputation_round_trips_frozen_sources(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload = terminal_package(
                root, attempt_ids=["a"], control_ids=["c"],
                attempt_rows=[{"attempt_id": "a", "terminal_status": "completed"}],
                control_rows=[{"control_attempt_id": "c", "terminal_status": "unavailable_no_parent"}],
                routes=[{"family": "fixture", "route": "shadow_verified"}],
                forensics=[{"family": "fixture", "status": "shadow_verified"}],
                all_workers_stopped=True, job_reconciliation={"pass": True},
            )
            stored = json.loads((root / "INDEPENDENT_RECOMPUTATION.json").read_text())
            replay = independent_terminal_recomputation(root, attempt_ids=["a"], control_ids=["c"], require_complete=True)
            self.assertEqual(stored, replay)
            self.assertEqual(sha256_file(root / "INDEPENDENT_RECOMPUTATION.json"), payload["independent_recomputation_sha256"])
            self.assertEqual("pass", verify_terminal_inventory(root)["status"])

    def test_real_shadow_bound_stop_uses_generation_scoped_terminal_builder(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = _write_shadow_bound_stop(root, generation=3, attempt_id="a", all_workers_stopped=True)
            replay = _write_shadow_bound_stop(root, generation=3, attempt_id="a", all_workers_stopped=True)
            self.assertEqual(first, replay)
            package = json.loads((root / "terminal_bound_stops/generation-000003/TERMINAL_PACKAGE.json").read_text())
            self.assertEqual(("global_bound_stop_incomplete", True), (package["status"], package["resumable"]))

    def test_detached_shadow_service_uses_supported_campaign_cli(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repository = root / "repository"
            repository.mkdir()
            spec_path = root / "SHADOW_SERVICE_SPEC.json"
            atomic_write_json(spec_path, {"schema": "stage24_shadow_service_spec_v1"})
            telegram = root / "telegram.env"
            telegram.write_text("TOKEN=fixture\n", encoding="utf-8")
            telegram.chmod(0o600)
            service = detached_shadow_service_spec(
                repository,
                root / "run",
                spec_path,
                sha256_file(spec_path),
                telegram_env_file=telegram,
            )
            self.assertEqual(
                [
                    str(repository / ".venv/bin/python"),
                    "-m",
                    "tools.run_stage22_core_liquid_campaign",
                    "shadow-run",
                    "--spec",
                    str(spec_path),
                ],
                service["exec_start"],
            )
            self.assertIsNone(service["environment"]["PYTHONPATH"])

    def test_stale_scheduled_heartbeat_stops_workers_without_late_commit(self) -> None:
        import time

        class Clock:
            value = 0.0

            def __call__(self) -> float:
                self.value += 1000.0
                return self.value

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            late = root / "late-write"
            deliveries = {"count": 0}

            def heartbeat(_payload: object) -> bool:
                deliveries["count"] += 1
                return deliveries["count"] == 1

            def slow() -> dict[str, object]:
                time.sleep(1.0)
                late.write_text("unsafe\n", encoding="utf-8")
                return {"registered_attempt_id": "slow", "status": "complete", "aggregate": {}}

            limits = ResourceLimits(
                max_workers=1, max_jobs_in_flight=1, max_output_bytes=32 * 1024**2,
                minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0,
                heartbeat_seconds=1, monitor_interval_seconds=0.001,
            )
            state = LazySupervisor(root, limits, heartbeat=heartbeat, monotonic=Clock()).run(iter([("slow", slow)]))
            time.sleep(0.05)
            self.assertEqual("global_resumable_bound_stop_heartbeat_stale", state["status"])
            self.assertTrue(state["all_workers_stopped"])
            self.assertFalse(late.exists())

    def test_abrupt_worker_pipe_eof_requeues_and_recovers_once(self) -> None:
        import os

        class Clock:
            value = 0.0

            def __call__(self) -> float:
                self.value += 100.0
                return self.value

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first_attempt = root / "first-attempt"

            def abrupt_then_complete() -> dict[str, object]:
                if not first_attempt.exists():
                    first_attempt.write_text("abrupt worker exit\n", encoding="utf-8")
                    os._exit(17)
                return {"registered_attempt_id": "abrupt", "status": "complete", "aggregate": {}}

            limits = ResourceLimits(
                max_workers=1, max_jobs_in_flight=1, max_output_bytes=32 * 1024**2,
                minimum_free_disk_bytes=1, minimum_free_disk_fraction=0.0,
                heartbeat_seconds=1800, monitor_interval_seconds=0.001,
            )
            state = LazySupervisor(root, limits, heartbeat=lambda _payload: True, monotonic=Clock()).run(
                iter([("abrupt", abrupt_then_complete)])
            )
            self.assertEqual("complete", state["status"])
            self.assertEqual(2, state["attempts"]["abrupt"])
            self.assertEqual(1, state["completed_count"])
            self.assertTrue(state["all_workers_stopped"])
            self.assertEqual([], state["worker_pids"])


if __name__ == "__main__":
    unittest.main()
