from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools import build_kda01_contract_closure as builder
from tools.build_kraken_c01_foundation import AuthorityRow
from tools.qlmg_kda01_contract_closure import (
    MAX_ENTRY_DELAY,
    MAX_EXIT_DELAY,
    attach_market_cluster_identity,
    execution_records,
    expected_next_open,
    frozen_contract_hash,
    locate_execution_availability,
    market_cluster_identity,
    normalized_bar_times,
)


def fixture_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = pd.DataFrame([
        {"parent_episode_id": "p1", "attempt": "primary", "parent_onset_ts": pd.Timestamp("2024-03-01T01:00Z")},
        {"parent_episode_id": "p2", "attempt": "primary", "parent_onset_ts": pd.Timestamp("2024-03-01T18:00Z")},
        {"parent_episode_id": "p3", "attempt": "robustness", "parent_onset_ts": pd.Timestamp("2024-03-01T01:00Z")},
    ])
    events = pd.DataFrame([
        {"event_id": "e1", "economic_address": "a1", "parent_episode_id": "p1", "attempt": "primary", "branch_id": "b", "symbol": "PF_AUSD", "decision_ts": pd.Timestamp("2024-03-01T01:05Z")},
        {"event_id": "e2", "economic_address": "a2", "parent_episode_id": "p2", "attempt": "primary", "branch_id": "b", "symbol": "PF_BUSD", "decision_ts": pd.Timestamp("2024-03-01T18:05Z")},
        {"event_id": "e3", "economic_address": "a3", "parent_episode_id": "p3", "attempt": "robustness", "branch_id": "b", "symbol": "PF_AUSD", "decision_ts": pd.Timestamp("2024-03-01T01:05Z")},
    ])
    return events, episodes


class KDA01ContractClosureTests(unittest.TestCase):
    def test_deterministic_utc_cluster_ids_and_attempt_separation(self):
        first = market_cluster_identity("primary", "2024-03-01T01:15:00Z")
        second = market_cluster_identity("primary", "2024-03-01T18:00:00Z")
        robustness = market_cluster_identity("robustness", "2024-03-01T01:15:00Z")
        self.assertEqual(first[0], "2024-03-01")
        self.assertEqual(first[1], second[1])
        self.assertNotEqual(first[2], second[2])
        self.assertNotEqual(first[1], robustness[1])
        self.assertEqual(first[3], pd.Timestamp("2024-03-01T00:00:00Z"))

    def test_same_day_cross_symbol_cluster_and_replay(self):
        events, episodes = fixture_events()
        first = attach_market_cluster_identity(events, episodes)
        second = attach_market_cluster_identity(events.sample(frac=1, random_state=4), episodes.sample(frac=1, random_state=5))
        first = first.sort_values("event_id").reset_index(drop=True)
        second = second.sort_values("event_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)
        primary = first[first.attempt.eq("primary")]
        self.assertEqual(primary.market_day_cluster_id.nunique(), 1)
        self.assertEqual(primary.market_6h_cluster_id.nunique(), 2)

    def test_expected_next_open_is_strict_grid(self):
        self.assertEqual(expected_next_open("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:05:00Z"))
        self.assertEqual(expected_next_open("2024-01-01T00:02:00Z"), pd.Timestamp("2024-01-01T00:05:00Z"))
        self.assertEqual(expected_next_open("2024-01-01T00:05:00Z"), pd.Timestamp("2024-01-01T00:10:00Z"))

    def test_exact_entry_delay_cap(self):
        at_cap = normalized_bar_times(["2024-01-01T00:15Z", "2024-01-01T01:15Z"])
        result = locate_execution_availability("2024-01-01T00:00Z", 1, at_cap)
        self.assertEqual(result.entry_delay_minutes, MAX_ENTRY_DELAY.total_seconds() / 60)
        self.assertEqual(result.status, "eligible")
        outside = normalized_bar_times(["2024-01-01T00:20Z", "2024-01-01T01:20Z"])
        self.assertEqual(locate_execution_availability("2024-01-01T00:00Z", 1, outside).status, "entry_delay_exceeded")

    def test_exact_exit_delay_cap(self):
        at_cap = normalized_bar_times(["2024-01-01T00:05Z", "2024-01-01T01:15Z"])
        result = locate_execution_availability("2024-01-01T00:00Z", 1, at_cap)
        self.assertEqual(result.exit_delay_minutes, MAX_EXIT_DELAY.total_seconds() / 60)
        self.assertEqual(result.status, "eligible")
        outside = normalized_bar_times(["2024-01-01T00:05Z", "2024-01-01T01:20Z"])
        self.assertEqual(locate_execution_availability("2024-01-01T00:00Z", 1, outside).status, "exit_delay_exceeded")

    def test_protected_target_rejected_without_protected_bar(self):
        bars = normalized_bar_times(["2025-12-31T23:55Z"])
        result = locate_execution_availability("2025-12-31T23:50Z", 1, bars)
        self.assertEqual(result.status, "protected_boundary_crossing")
        with self.assertRaisesRegex(ValueError, "non-rankable"):
            normalized_bar_times(["2026-01-01T00:00Z"])

    def test_actual_exit_non_overlap_is_definition_local(self):
        events = pd.DataFrame([
            {"event_id": "e1", "economic_address": "a1", "branch_id": "b", "symbol": "PF_AUSD", "decision_ts": pd.Timestamp("2024-01-01T00:00Z")},
            {"event_id": "e2", "economic_address": "a2", "branch_id": "b", "symbol": "PF_AUSD", "decision_ts": pd.Timestamp("2024-01-01T00:30Z")},
            {"event_id": "e3", "economic_address": "a3", "branch_id": "b", "symbol": "PF_AUSD", "decision_ts": pd.Timestamp("2024-01-01T01:05Z")},
        ])
        definitions = pd.DataFrame([
            {"definition_id": "d1", "definition_contract_hash": "h1", "branch_id": "b", "timeout_hours": 1},
            {"definition_id": "d6", "definition_contract_hash": "h6", "branch_id": "b", "timeout_hours": 6},
        ])
        bars = normalized_bar_times(pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC"))
        result = execution_records(events, definitions, {"PF_AUSD": bars})
        d1 = result[result.definition_id.eq("d1")]
        d6 = result[result.definition_id.eq("d6")]
        self.assertEqual(d1.accepted.tolist(), [True, False, True])
        self.assertEqual(d6.accepted.tolist(), [True, False, False])
        self.assertEqual(d1.loc[~d1.accepted, "status"].tolist(), ["actual_position_overlap"])

    def test_timestamp_loader_requests_no_price_columns(self):
        requested = []

        class FakeSchema:
            names = list(builder.TIMESTAMP_ONLY_COLUMNS)

        class FakeParquet:
            schema_arrow = FakeSchema()

            def __init__(self, _path):
                pass

            def read(self, columns):
                requested.extend(columns)
                return FakeTable()

        class FakeTable:
            def to_pandas(self):
                return pd.DataFrame({
                    "time": [pd.Timestamp("2024-01-01T00:00Z").timestamp() * 1000],
                    "venue_symbol": ["PF_AUSD"],
                    "resolution": ["5m"],
                    "rankable_pre_holdout": [True],
                    "contains_protected_period": [False],
                })

        row = AuthorityRow("historical_trade_candles_5m", "PF_AUSD", pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-01-02T00:00Z"), Path("/tmp/fake.parquet"), "x", 1)
        with patch.object(builder.pq, "ParquetFile", FakeParquet):
            times, _ = builder.load_timestamp_only_bars([row], "PF_AUSD")
        self.assertEqual(len(times), 1)
        self.assertEqual(set(requested), set(builder.TIMESTAMP_ONLY_COLUMNS))
        self.assertFalse({"open", "high", "low", "close", "volume"} & set(requested))

    def test_non_bar_manifest_envelope_is_skipped_before_read(self):
        requested = []

        class Schema:
            def __init__(self, names):
                self.names = names

        class FakeParquet:
            def __init__(self, path):
                self.is_bar = "bar" in str(path)
                self.schema_arrow = Schema(list(builder.TIMESTAMP_ONLY_COLUMNS) if self.is_bar else ["message"])

            def read(self, columns):
                self.assert_bar()
                requested.append(tuple(columns))
                return FakeTable()

            def assert_bar(self):
                if not self.is_bar:
                    raise AssertionError("non-bar envelope reader invoked")

        class FakeTable:
            def to_pandas(self):
                return pd.DataFrame({
                    "time": [pd.Timestamp("2024-01-01T00:00Z").timestamp() * 1000],
                    "venue_symbol": ["PF_AUSD"], "resolution": ["5m"],
                    "rankable_pre_holdout": [True], "contains_protected_period": [False],
                })

        rows = [
            AuthorityRow("historical_trade_candles_5m", "PF_AUSD", pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-01-02T00:00Z"), Path("/tmp/envelope.parquet"), "x", 1),
            AuthorityRow("historical_trade_candles_5m", "PF_AUSD", pd.Timestamp("2024-01-02T00:00Z"), pd.Timestamp("2024-01-03T00:00Z"), Path("/tmp/bar.parquet"), "y", 1),
        ]
        with patch.object(builder.pq, "ParquetFile", FakeParquet):
            times, _ = builder.load_timestamp_only_bars(rows, "PF_AUSD")
        self.assertEqual(len(times), 1)
        self.assertEqual(len(requested), 1)

    def test_contract_serialization_is_idempotent(self):
        contract = {"b": [2, 1], "a": {"x": True}}
        self.assertEqual(frozen_contract_hash(contract), frozen_contract_hash({"a": {"x": True}, "b": [2, 1]}))
        with_hash = {**contract, "level3_contract_hash": "ignored"}
        self.assertEqual(frozen_contract_hash(contract), frozen_contract_hash(with_hash))

    def test_source_has_no_price_or_outcome_reader(self):
        source = Path(builder.__file__).read_text(encoding="utf-8").lower()
        self.assertIn("sys.path.insert(0, str(repository_root))", source)
        self.assertNotIn("candidate_return", source)
        self.assertNotIn("forward_return", source)
        self.assertNotIn("calculate_pnl", source)
        self.assertNotIn('columns=["time", "open"', source)

    def test_actual_stage8b_manifest_reconciles(self):
        archive = Path("/opt/testerdonch-stage8b-20260719/docs/agent/task_archive/20260719_donch_bt_stage_8b_kda01_prerun_freeze_20260719_v1")
        if not archive.exists():
            self.skipTest("authoritative local Stage 8B archive not mounted")
        result = builder.verify_stage8b(archive)
        self.assertEqual(result["archive_objects"], 52)
        self.assertEqual(result["cache_objects"], 748)
        self.assertEqual(builder.sha256_file(result["manifest_path"]), builder.SOURCE_MANIFEST_FILE_SHA256)


if __name__ == "__main__":
    unittest.main()
