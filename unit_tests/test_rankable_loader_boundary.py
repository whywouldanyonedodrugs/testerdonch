from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as sweep


TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2025-12-31T23:59:59Z")


class RankableLoaderBoundaryTests(unittest.TestCase):
    def _paths(self, root: Path, file_path: Path, authority: dict) -> dict:
        for rel in ["trade", "mark", "funding"]:
            (root / rel / "PF_TESTUSD").mkdir(parents=True, exist_ok=True)
        return {
            "trade_5m": root / "trade",
            "alt_trade": root / "unused_trade",
            "mark_5m": root / "mark",
            "alt_mark": root / "unused_mark",
            "funding": root / "funding",
            "rankable_file_authority": {str(file_path): authority},
        }

    @staticmethod
    def _authority(
        *,
        purpose: str = "rankable_research",
        start: str = "2023-01-01T00:00:00Z",
        end: str = "2025-12-31T23:59:59Z",
        venue: str = "kraken",
    ) -> dict:
        return {
            "purpose": purpose,
            "venue": venue,
            "start_ts": start,
            "end_ts": end,
            "funding_type": "exact",
        }

    def test_market_rejects_protected_mixed_and_unrankable_files_before_reader(self):
        cases = [
            ("protected", self._authority(start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z")),
            ("mixed", self._authority(purpose="mixed_rankable_protected")),
            ("calibration", self._authority(purpose="execution_calibration_only")),
            ("prospective", self._authority(purpose="prospective_shadow")),
            ("external", self._authority(purpose="external_unrankable")),
            ("unknown", self._authority(purpose="unknown")),
            ("unprovable", None),
        ]
        for label, authority in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                path = root / "trade/PF_TESTUSD/20250101T000000_fixture.parquet"
                path.parent.mkdir(parents=True)
                path.touch()
                paths = self._paths(root, path, authority)
                reader = mock.Mock(return_value=pd.DataFrame())
                downstream = mock.Mock()
                with mock.patch.object(sweep.pd, "read_parquet", reader):
                    with self.assertRaisesRegex(RuntimeError, "rankable file authority"):
                        frame = sweep.load_symbol_bars(paths, "PF_TESTUSD", TRAIN_START, TRAIN_END)
                        downstream(frame)
                self.assertEqual(reader.call_count, 0)
                self.assertEqual(downstream.call_count, 0)

    def test_market_authorized_file_filters_pretrain_and_non_kraken_before_downstream(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "trade/PF_TESTUSD/20220101T000000_fixture.parquet"
            path.parent.mkdir(parents=True)
            path.touch()
            paths = self._paths(
                root,
                path,
                self._authority(start="2022-01-01T00:00:00Z"),
            )
            rows = pd.DataFrame(
                [
                    {"timestamp": "2022-12-31T23:55:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 1, "venue_symbol": "PF_TESTUSD"},
                    {"timestamp": "2024-01-01T00:00:00Z", "open": 2, "high": 3, "low": 1.5, "close": 2.5, "volume": 1, "venue_symbol": "BYBIT:TESTUSDT"},
                    {"timestamp": "2024-01-01T00:05:00Z", "open": 3, "high": 4, "low": 2.5, "close": 3.5, "volume": 1, "venue_symbol": "PF_TESTUSD"},
                ]
            )
            reader = mock.Mock(return_value=rows)
            downstream = mock.Mock()
            with mock.patch.object(sweep.pd, "read_parquet", reader):
                frame = sweep.load_symbol_bars(
                    paths,
                    "PF_TESTUSD",
                    pd.Timestamp("2022-01-01T00:00:00Z"),
                    TRAIN_END,
                )
                downstream(frame)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(downstream.call_count, 1)
            consumed = downstream.call_args.args[0]
            self.assertEqual(consumed["ts"].tolist(), [pd.Timestamp("2024-01-01T00:05:00Z")])
            self.assertEqual(consumed["venue_symbol"].tolist(), ["PF_TESTUSD"])

    def test_market_valid_rankable_kraken_fixture_reaches_downstream(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "trade/PF_TESTUSD/20240101T000000_fixture.parquet"
            path.parent.mkdir(parents=True)
            path.touch()
            paths = self._paths(root, path, self._authority())
            rows = pd.DataFrame(
                [{"timestamp": "2024-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 1, "venue_symbol": "PF_TESTUSD"}]
            )
            reader = mock.Mock(return_value=rows)
            downstream = mock.Mock()
            with mock.patch.object(sweep.pd, "read_parquet", reader):
                downstream(sweep.load_symbol_bars(paths, "PF_TESTUSD", TRAIN_START, TRAIN_END))
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(downstream.call_count, 1)
            self.assertEqual(len(downstream.call_args.args[0]), 1)

    def test_market_rejects_unrankable_mark_file_before_mark_reader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trade_path = root / "trade/PF_TESTUSD/20240101T000000_trade.parquet"
            mark_path = root / "mark/PF_TESTUSD/20240101T000000_mark.parquet"
            trade_path.parent.mkdir(parents=True)
            mark_path.parent.mkdir(parents=True)
            trade_path.touch()
            mark_path.touch()
            paths = self._paths(root, trade_path, self._authority())
            paths["rankable_file_authority"][str(mark_path)] = self._authority(purpose="execution_calibration_only")
            trade_rows = pd.DataFrame(
                [{"timestamp": "2024-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "venue_symbol": "PF_TESTUSD"}]
            )
            trade_reader = mock.Mock(return_value=trade_rows)
            mark_reader = mock.Mock(return_value=pd.DataFrame())

            def reader(path: Path) -> pd.DataFrame:
                return mark_reader(path) if path == mark_path else trade_reader(path)

            downstream = mock.Mock()
            with mock.patch.object(sweep.pd, "read_parquet", side_effect=reader):
                with self.assertRaisesRegex(RuntimeError, "rankable file authority"):
                    downstream(sweep.load_symbol_bars(paths, "PF_TESTUSD", TRAIN_START, TRAIN_END))
            self.assertEqual(trade_reader.call_count, 1)
            self.assertEqual(mark_reader.call_count, 0)
            self.assertEqual(downstream.call_count, 0)

    def test_funding_rejects_protected_mixed_and_unrankable_files_before_reader(self):
        cases = [
            ("protected", self._authority(start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z")),
            ("mixed", self._authority(purpose="mixed_rankable_protected")),
            ("calibration", self._authority(purpose="execution_calibration_only")),
            ("prospective", self._authority(purpose="prospective_shadow")),
            ("external", self._authority(purpose="external_unrankable")),
            ("unknown", self._authority(purpose="unknown")),
            ("unprovable", None),
        ]
        for label, authority in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                path = root / "funding/PF_TESTUSD/20250101T000000_fixture.parquet"
                path.parent.mkdir(parents=True)
                path.touch()
                paths = self._paths(root, path, authority)
                reader = mock.Mock(return_value=pd.DataFrame())
                downstream = mock.Mock()
                with mock.patch.object(sweep.pd, "read_parquet", reader):
                    with self.assertRaisesRegex(RuntimeError, "rankable file authority"):
                        frame = sweep.load_funding(paths, "PF_TESTUSD", TRAIN_END)
                        downstream(frame)
                self.assertEqual(reader.call_count, 0)
                self.assertEqual(downstream.call_count, 0)

    def test_funding_rejects_imputed_file_before_reader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "funding/PF_TESTUSD/20240101T000000_fixture.parquet"
            path.parent.mkdir(parents=True)
            path.touch()
            authority = self._authority()
            authority["funding_type"] = "imputed"
            paths = self._paths(root, path, authority)
            reader = mock.Mock(return_value=pd.DataFrame())
            downstream = mock.Mock()
            with mock.patch.object(sweep.pd, "read_parquet", reader):
                with self.assertRaisesRegex(RuntimeError, "rankable file authority"):
                    downstream(sweep.load_funding(paths, "PF_TESTUSD", TRAIN_END))
            self.assertEqual(reader.call_count, 0)
            self.assertEqual(downstream.call_count, 0)

    def test_funding_filters_pretrain_non_kraken_and_signal_ineligible_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "funding/PF_TESTUSD/20220101T000000_fixture.parquet"
            path.parent.mkdir(parents=True)
            path.touch()
            paths = self._paths(
                root,
                path,
                self._authority(start="2022-01-01T00:00:00Z"),
            )
            rows = pd.DataFrame(
                [
                    {"timestamp": "2022-12-31T23:00:00Z", "fundingRate": 0.01, "venue": "kraken", "funding_exact": True},
                    {"timestamp": "2024-01-01T00:00:00Z", "fundingRate": 0.02, "venue": "bybit", "funding_exact": True},
                    {"timestamp": "2024-01-01T01:00:00Z", "fundingRate": 0.03, "venue": "kraken", "funding_exact": "false"},
                    {"timestamp": "2024-01-01T02:00:00Z", "fundingRate": 0.04, "venue": "kraken", "funding_exact": True},
                ]
            )
            reader = mock.Mock(return_value=rows)
            downstream = mock.Mock()
            with mock.patch.object(sweep.pd, "read_parquet", reader):
                downstream(sweep.load_funding(paths, "PF_TESTUSD", TRAIN_END))
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(downstream.call_count, 1)
            consumed = downstream.call_args.args[0]
            self.assertEqual(consumed["timestamp"].tolist(), [pd.Timestamp("2024-01-01T02:00:00Z")])
            self.assertTrue(bool(consumed.iloc[0]["funding_exact"]))

    def test_funding_valid_exact_rankable_kraken_fixture_reaches_downstream(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "funding/PF_TESTUSD/20240101T000000_fixture.parquet"
            path.parent.mkdir(parents=True)
            path.touch()
            paths = self._paths(root, path, self._authority())
            rows = pd.DataFrame(
                [{"timestamp": "2024-01-01T00:00:00Z", "fundingRate": 0.01, "venue": "kraken", "funding_exact": True}]
            )
            reader = mock.Mock(return_value=rows)
            downstream = mock.Mock()
            with mock.patch.object(sweep.pd, "read_parquet", reader):
                downstream(sweep.load_funding(paths, "PF_TESTUSD", TRAIN_END))
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(downstream.call_count, 1)
            self.assertEqual(len(downstream.call_args.args[0]), 1)


if __name__ == "__main__":
    unittest.main()
