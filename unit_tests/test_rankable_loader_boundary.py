from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
            "rankable_pre_holdout": True,
            "contains_protected_period": False,
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

    @staticmethod
    def _reader_rows(*, ohlcv: bool) -> pd.DataFrame:
        rows = [
            {"timestamp": "2022-12-31T23:55:00Z", "close": 1.0, "venue": "kraken", "venue_symbol": "PF_TESTUSD"},
            {"timestamp": "2024-01-01T00:00:00Z", "close": 2.0, "venue": "bybit", "venue_symbol": "BYBIT:TESTUSDT"},
            {"timestamp": "2024-01-01T00:05:00Z", "close": 3.0, "venue": "kraken", "venue_symbol": "PF_TESTUSD"},
        ]
        if ohlcv:
            for row in rows:
                row.update({"open": row["close"], "high": row["close"] + 1, "low": row["close"] - 0.5, "volume": 1})
        return pd.DataFrame(rows)

    @staticmethod
    def _projecting_reader(frame: pd.DataFrame):
        def read(_path: Path, columns=None) -> pd.DataFrame:
            if columns is None:
                return frame.copy()
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(f"missing synthetic columns: {sorted(missing)}")
            return frame[list(columns)].copy()
        return read

    @staticmethod
    def _call_remaining_reader(name: str, paths: dict) -> pd.DataFrame:
        start = pd.Timestamp("2022-01-01T00:00:00Z")
        if name == "signal":
            return sweep.load_symbol_signal_bars(paths, "PF_TESTUSD", start, TRAIN_END)
        if name == "rank":
            return sweep.load_symbol_rank_close_window(paths, "PF_TESTUSD", start, TRAIN_END)
        if name == "a1":
            return sweep.a1_load_symbol_bars_window(paths, "PF_TESTUSD", start, TRAIN_END)
        raise AssertionError(name)

    def test_remaining_readers_reject_unrankable_files_before_payload_reader(self):
        authorities = {
            "protected": self._authority(start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z"),
            "mixed": self._authority(purpose="mixed_rankable_protected"),
            "unknown": None,
        }
        for reader_name in ["signal", "rank", "a1"]:
            for label, authority in authorities.items():
                with self.subTest(reader=reader_name, authority=label), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    path = root / "trade/PF_TESTUSD/20250101T000000_fixture.parquet"
                    path.parent.mkdir(parents=True)
                    path.touch()
                    paths = self._paths(root, path, authority)
                    payload_reader = mock.Mock(side_effect=self._projecting_reader(self._reader_rows(ohlcv=reader_name == "a1")))
                    downstream = mock.Mock()
                    with mock.patch.object(sweep.pd, "read_parquet", payload_reader):
                        with self.assertRaisesRegex(RuntimeError, "rankable file authority"):
                            downstream(self._call_remaining_reader(reader_name, paths))
                    self.assertEqual(payload_reader.call_count, 0)
                    self.assertEqual(downstream.call_count, 0)

    def test_remaining_readers_filter_pretrain_and_non_kraken_rows_before_downstream(self):
        for reader_name in ["signal", "rank", "a1"]:
            with self.subTest(reader=reader_name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                path = root / "trade/PF_TESTUSD/20220101T000000_fixture.parquet"
                path.parent.mkdir(parents=True)
                path.touch()
                paths = self._paths(root, path, self._authority(start="2022-01-01T00:00:00Z"))
                payload_reader = mock.Mock(side_effect=self._projecting_reader(self._reader_rows(ohlcv=reader_name == "a1")))
                downstream = mock.Mock()
                with mock.patch.object(sweep.pd, "read_parquet", payload_reader):
                    downstream(self._call_remaining_reader(reader_name, paths))
                self.assertGreaterEqual(payload_reader.call_count, 1)
                self.assertEqual(downstream.call_count, 1)
                consumed = downstream.call_args.args[0]
                self.assertEqual(consumed["ts"].tolist(), [pd.Timestamp("2024-01-01T00:05:00Z")])

    def test_data_paths_binds_existing_manifest_authority_for_remaining_readers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file_path = root / "parquet/historical_trade_candles_5m/PF_TESTUSD/20230101T000000_fixture.parquet"
            file_path.parent.mkdir(parents=True)
            file_path.touch()
            manifest = root / "manifests/synthetic_download_manifest.csv"
            manifest.parent.mkdir(parents=True)
            pd.DataFrame([{
                "dataset": "historical_trade_candles_5m",
                "symbol": "PF_TESTUSD",
                "parquet_path": f"/stale/acquisition/root/{file_path.name}",
                "status": "downloaded",
                "chunk_start": "2023-01-01T00:00:00Z",
                "chunk_end": "2026-01-01T00:00:00Z",
                "rankable_pre_holdout": True,
                "contains_protected_period": False,
            }]).to_csv(manifest, index=False)
            ctx = SimpleNamespace(args=SimpleNamespace(kraken_data_root=str(root)))
            paths = sweep.data_paths(ctx)
            self.assertIn(str(file_path), paths["rankable_file_authority"])
            payload_reader = mock.Mock(side_effect=self._projecting_reader(self._reader_rows(ohlcv=True).tail(1)))
            with mock.patch.object(sweep.pd, "read_parquet", payload_reader):
                for reader_name in ["signal", "rank", "a1"]:
                    with self.subTest(reader=reader_name):
                        frame = self._call_remaining_reader(reader_name, paths)
                        self.assertEqual(len(frame), 1)


if __name__ == "__main__":
    unittest.main()
