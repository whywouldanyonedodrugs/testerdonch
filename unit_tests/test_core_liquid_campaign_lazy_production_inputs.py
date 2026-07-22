from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.core_liquid_campaign.canonical import sha256_file
from tools.core_liquid_campaign.engine_types import ContextInputs, SignalBar
from tools.core_liquid_campaign.lazy_production_inputs import (
    FamilyDecisionLocator,
    LazyProductionFamilyInputAdapter,
    LazyProductionInputError,
    SUPPORTED_FAMILIES,
    _fold_partitions,
)
from tools.core_liquid_campaign.production_cache import SourcePart
from tools.core_liquid_campaign.production_population_tables import PopulationTableError


UTC = timezone.utc


def _row(symbol: str, rank: float, *, day_ms: int, decision_count: int = 288) -> dict[str, object]:
    population = 2
    return {
        "symbol": symbol,
        "day_open_ms": day_ms,
        "decision_count_5m": decision_count,
        "average_liquidity_rank": rank,
        "lagged_30d_median_canonical_quote_notional": 3.0 - rank,
        "eligible_population": population,
        "top_10": True,
        "top_20": True,
        "top_40": True,
    }


class _EnumeratorAdapter:
    iter_decision_locators = LazyProductionFamilyInputAdapter.iter_decision_locators


class _ScheduleAdapter:
    _execution_schedule_bar = LazyProductionFamilyInputAdapter._execution_schedule_bar

    def __init__(self, part: SourcePart) -> None:
        self.part = part

    def _verified_trade_parts(self, symbol, start, end):
        return (self.part,)


class _FrameAdapter:
    frame = LazyProductionFamilyInputAdapter.frame

    def __init__(self, root: Path, decision: datetime) -> None:
        self.construction_mode = "shadow_no_outcome"
        day_ms = int(decision.replace(hour=0, minute=0).timestamp() * 1000)
        self._pit_by_day = {day_ms: (_row("PF_XBTUSD", 1.0, day_ms=day_ms), _row("PF_ETHUSD", 2.0, day_ms=day_ms))}
        self.virtual_cache_manifest_path = root / "virtual.json"
        self.a1_population_manifest_path = root / "a1.json"
        self.a3_population_manifest_path = root / "a3.json"
        for path in (self.virtual_cache_manifest_path, self.a1_population_manifest_path, self.a3_population_manifest_path):
            path.write_text("{}\n", encoding="utf-8")
        self.authority = {
            "source_manifest_sha256": "1" * 64,
            "pit_universe_sha256": "2" * 64,
            "funding_manifest_sha256": "3" * 64,
            "cache_contract_sha256": "4" * 64,
            "fold_graph_sha256": "5" * 64,
            "source_records": [],
            "rankable_funding_package_sha256": "6" * 64,
        }
        self.completed = SignalBar(
            decision - timedelta(minutes=5), decision,
            100.0, 101.0, 99.0, 100.5,
            decision, decision, True, True, 1000.0,
        )

    def _partition(self, locator):
        return {
            "phase": locator.phase,
            "outer_fold_id": locator.outer_fold_id,
            "inner_fold_id": locator.inner_fold_id,
            "training_start": datetime(2023, 1, 1, tzinfo=UTC),
            "training_end_exclusive": datetime(2023, 12, 20, tzinfo=UTC),
            "evaluation_start": datetime(2024, 1, 1, tzinfo=UTC),
            "evaluation_end_exclusive": datetime(2024, 4, 1, tzinfo=UTC),
        }

    def _populations(self, locator, partition, decile):
        return {}, []

    def _bars(self, symbol, cutoff):
        return (self.completed,)

    def _execution_schedule_bar(self, symbol, decision, completed):
        self.assert_completed = tuple(completed)
        return SignalBar(decision, decision + timedelta(minutes=5), 100.5, 100.5, 100.5, 100.5, decision, decision)

    def _daily(self, symbol, cutoff):
        return ()

    def _funding(self, symbol, cutoff):
        return ()

    def _context_inputs(self, locator, partition, snapshot):
        return ContextInputs()

    def _parts(self, symbol):
        return (SourcePart("historical_trade_candles_5m", symbol, "/bound/source.parquet", "7" * 64, 1, "2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"),)


class LazyProductionInputTests(unittest.TestCase):
    def test_a3_population_requests_match_frozen_5_10_20_60_registry(self) -> None:
        class Recorder:
            def __init__(self): self.names = []
            def population(self, name, **kwargs): self.names.append(name); return object()

        adapter = object.__new__(LazyProductionFamilyInputAdapter)
        adapter._a3 = Recorder()
        populations, unavailable = adapter._table_populations(
            "A3_STARTER_RETEST_V3", "PF_XBTUSD", 1,
            {"training_start": datetime(2023, 1, 1, tzinfo=UTC), "training_end_exclusive": datetime(2024, 1, 1, tzinfo=UTC)},
        )
        lookbacks = {int(name.split("lookback=", 1)[1].split(":", 1)[0]) for name in populations}
        self.assertEqual(lookbacks, {5, 10, 20, 60})
        self.assertEqual(len(populations), 4 * 4 * 2 * 3)
        self.assertEqual(unavailable, [])

    def test_missing_registered_a3_signature_is_compiler_failure_not_typed_unavailable(self) -> None:
        class Missing:
            def population(self, name, **kwargs): raise PopulationTableError("A3 population feature is absent")

        adapter = object.__new__(LazyProductionFamilyInputAdapter)
        adapter._a3 = Missing()
        with self.assertRaisesRegex(LazyProductionInputError, "compiler output"):
            adapter._table_populations(
                "A3_STARTER_RETEST_V3", "PF_XBTUSD", 1,
                {"training_start": datetime(2023, 1, 1, tzinfo=UTC), "training_end_exclusive": datetime(2024, 1, 1, tzinfo=UTC)},
            )

    def test_frozen_fold_graph_expands_all_132_positions(self) -> None:
        graph = {"outer_folds": []}
        inner_count = 0
        for outer_index in range(8):
            count = 16 if outer_index < 4 else 15
            inner = []
            for item in range(count):
                inner_count += 1
                inner.append({
                    "inner_fold_id": f"I{inner_count:03d}",
                    "training_start": "2023-01-01T00:00:00Z",
                    "training_latest_exit_exclusive": "2023-06-20T00:00:00Z",
                    "validation_start": "2023-07-01T00:00:00Z",
                    "validation_end_exclusive": "2023-08-01T00:00:00Z",
                })
            graph["outer_folds"].append({
                "outer_fold_id": f"O{outer_index}",
                "development_start": "2023-01-01T00:00:00Z",
                "purge_days": 10,
                "outer_evaluation_start": "2024-01-01T00:00:00Z",
                "outer_evaluation_end_exclusive": "2024-04-01T00:00:00Z",
                "inner_folds": inner,
            })
        partitions = _fold_partitions(graph)
        self.assertEqual(inner_count, 124)
        self.assertEqual(len(partitions), 132)

    def test_bounded_schedule_enumeration_and_parent_bound_a2(self) -> None:
        adapter = _EnumeratorAdapter()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        key = ("outer_evaluation", "2024Q1", None)
        adapter.partitions = {key: {"evaluation_start": start, "evaluation_end_exclusive": start + timedelta(days=1)}}
        day_ms = int(start.timestamp() * 1000)
        adapter._pit_by_day = {day_ms: (_row("PF_XBTUSD", 1.0, day_ms=day_ms), _row("PF_ETHUSD", 2.0, day_ms=day_ms))}
        common = {
            "partition_key": key,
            "executable_attempt_id": "attempt",
            "canonical_economic_address_sha256": "a" * 64,
        }
        a1 = list(adapter.iter_decision_locators(family_id="A1_COMPRESSION_V2", config={"PIT_liquidity_top_n": 10}, **common))
        a3 = list(adapter.iter_decision_locators(family_id="A3_STARTER_RETEST_V3", config={"PIT_liquidity_top_n": 10}, **common))
        a4_daily = list(adapter.iter_decision_locators(family_id="A4_TSMOM_V7", config={"PIT_liquidity_top_n": 10, "rebalance": "1d"}, **common))
        a4_8h = list(adapter.iter_decision_locators(family_id="A4_TSMOM_V7", config={"PIT_liquidity_top_n": 10, "rebalance": "8h"}, **common))
        self.assertEqual((len(a1), len(a3), len(a4_daily), len(a4_8h)), (576, 576, 2, 6))
        self.assertEqual([row.decision_ts.hour for row in a4_8h], [0, 0, 8, 8, 16, 16])
        with self.assertRaisesRegex(LazyProductionInputError, "parent-event-bound"):
            list(adapter.iter_decision_locators(family_id="A2_PRIOR_HIGH_RS_CONTEXT_V1", config={"PIT_liquidity_top_n": 10}, **common))

    def test_schedule_bar_reads_only_timestamp_and_uses_last_completed_close(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = datetime(2024, 1, 2, tzinfo=UTC)
            path = root / "schedule.parquet"
            # Extreme future OHLC values are intentionally not present: the
            # schedule probe is physically capable of reading only `time`.
            pq.write_table(pa.table({"time": [int(decision.timestamp() * 1000)]}), path)
            part = SourcePart("historical_trade_candles_5m", "PF_XBTUSD", str(path), sha256_file(path), 1, decision.isoformat(), (decision + timedelta(minutes=5)).isoformat())
            completed = SignalBar(decision - timedelta(minutes=5), decision, 99.0, 101.0, 98.0, 100.0, decision, decision)
            schedule = _ScheduleAdapter(part)._execution_schedule_bar("PF_XBTUSD", decision, (completed,))
            self.assertEqual((schedule.open, schedule.high, schedule.low, schedule.close), (100.0, 100.0, 100.0, 100.0))
            self.assertEqual(schedule.source_close_ts, decision)
            self.assertEqual(schedule.open_ts, decision)

    def test_frame_path_never_carries_real_post_entry_ohlc_for_all_a1_a4_families(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            decision = datetime(2024, 1, 20, 12, 0, tzinfo=UTC)
            adapter = _FrameAdapter(Path(temporary), decision)
            for family in sorted(SUPPORTED_FAMILIES):
                locator = FamilyDecisionLocator(family, "outer_evaluation", "2024Q1", None, "PF_XBTUSD", decision, "attempt", "b" * 64)
                frame = adapter.frame(locator)
                real_rows = frame.five_minute_bars[:-1]
                schedule = frame.five_minute_bars[-1]
                self.assertTrue(all(row.close_ts <= decision for row in real_rows))
                self.assertEqual((schedule.open, schedule.high, schedule.low, schedule.close), (100.5, 100.5, 100.5, 100.5))
                self.assertEqual(frame.metadata["shadow_ohlc_attestation"]["post_entry_real_ohlc_rows"], 0)
                self.assertTrue(frame.metadata["shadow_ohlc_attestation"]["execution_open_timestamp_only"])

    def test_fail_closed_on_bad_locator_pit_and_explicit_hash(self) -> None:
        with self.assertRaisesRegex(LazyProductionInputError, "timezone-naive"):
            FamilyDecisionLocator("A1_COMPRESSION_V2", "outer_evaluation", "O", None, "PF_XBTUSD", datetime(2024, 1, 1))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for label in ("execution_input_authority", "virtual_cache_manifest", "fold_graph", "a1_population_manifest", "a3_population_manifest"):
                path = root / f"{label}.json"; path.write_text("{}\n", encoding="utf-8"); paths[label] = path
            adapter = object.__new__(LazyProductionFamilyInputAdapter)
            adapter.execution_authority_path = paths["execution_input_authority"]
            adapter.virtual_cache_manifest_path = paths["virtual_cache_manifest"]
            adapter.fold_graph_path = paths["fold_graph"]
            adapter.a1_population_manifest_path = paths["a1_population_manifest"]
            adapter.a3_population_manifest_path = paths["a3_population_manifest"]
            expected = {label: sha256_file(path) for label, path in paths.items()}
            expected["fold_graph"] = "0" * 64
            with self.assertRaisesRegex(LazyProductionInputError, "fold_graph SHA-256 differs"):
                adapter._verify_explicit_hashes(expected)

    def test_non_shadow_construction_mode_fails_closed_before_authority_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            with self.assertRaisesRegex(LazyProductionInputError, "shadow_no_outcome"):
                LazyProductionFamilyInputAdapter(
                    repository_root=Path(temporary),
                    execution_authority_path=missing,
                    virtual_cache_manifest_path=missing,
                    virtual_cache_root=Path(temporary),
                    fold_graph_path=missing,
                    a1_population_manifest_path=missing,
                    a3_population_manifest_path=missing,
                    expected_sha256={},
                    construction_mode="economic",
                )


if __name__ == "__main__":
    unittest.main()
