from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd

from tools import run_kraken_c02_spot_reference_authority as c02


class C02SpotReferenceAuthorityTests(unittest.TestCase):
    def test_usd_only_identity_and_ticker_migration(self):
        pairs = {
            "XXBTZUSD": {"wsname": "XBT/USD", "altname": "XBTUSD", "base": "XXBT", "quote": "ZUSD", "status": "online"},
            "XETHZUSD": {"wsname": "ETH/USD", "altname": "ETHUSD", "base": "XETH", "quote": "ZUSD", "status": "online"},
            "AAVEEUR": {"wsname": "AAVE/EUR", "altname": "AAVEEUR", "base": "AAVE", "quote": "ZEUR", "status": "online"},
            "AAVEUSD": {"wsname": "AAVE/USD", "altname": "AAVEUSD", "base": "AAVE", "quote": "ZUSD", "status": "online"},
        }
        cohort = pd.DataFrame([{"symbol": "PF_AAVEUSD", "base": "AAVE"}])
        result = c02.build_pair_authority(cohort, pairs)
        self.assertEqual(result.set_index("PF_symbol").loc["PF_XBTUSD", "Kraken_spot_pair"], "XBTUSD")
        self.assertEqual(result.set_index("PF_symbol").loc["PF_AAVEUSD", "Kraken_spot_pair"], "AAVEUSD")
        self.assertNotIn("AAVEEUR", result.Kraken_spot_pair.tolist())

    def test_unknown_identity_remains_unknown(self):
        result = c02.build_pair_authority(
            pd.DataFrame([{"symbol": "PF_UNKNOWNUSD", "base": "UNKNOWN"}]), {}
        )
        row = result[result.PF_symbol.eq("PF_UNKNOWNUSD")].iloc[0]
        self.assertEqual(row.historical_authority_status, "pending_archive_observation")
        self.assertEqual(row.inclusion_or_exclusion_reason, "no_unique_official_USD_pair")

    def test_archive_boundary_rejected_before_zip_open(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.zip"
            path.touch()
            spec = c02.ArchiveSpec(
                "bad", path, c02.FULL_ARCHIVE_URL, c02.TRAIN_START, c02.PROTECTED_START + pd.Timedelta(days=1)
            )
            with self.assertRaisesRegex(ValueError, "protected"):
                c02.validate_archive_spec(spec)

    def test_non_official_archive_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("AAVEUSD.csv", "1672531200,1,2\n")
            spec = c02.ArchiveSpec("bad", path, "https://example.com/file", c02.TRAIN_START, c02.PROTECTED_START)
            with self.assertRaisesRegex(ValueError, "non-official"):
                c02.validate_archive_spec(spec)

    def test_trade_to_5m_aggregation_is_deterministic(self):
        chunk = pd.DataFrame(
            [[1672531201, 10, 2], [1672531210, 12, 3], [1672531500, 11, 4]],
            columns=["timestamp", "price", "volume"],
        )
        first, stats = c02.aggregate_trade_chunks([chunk.copy()])
        second, _ = c02.aggregate_trade_chunks([chunk.copy()])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.iloc[0][["open", "high", "low", "close", "volume", "trade_count"]].tolist(), [10, 12, 10, 12, 5, 2])
        self.assertEqual(stats["trade_rows"], 3)

    def test_duplicate_tuple_reported_without_volume_removal(self):
        chunk = pd.DataFrame(
            [[1672531201, 10, 2], [1672531201, 10, 2]],
            columns=["timestamp", "price", "volume"],
        )
        bars, stats = c02.aggregate_trade_chunks([chunk])
        self.assertEqual(stats["duplicate_tuple_rows"], 1)
        self.assertEqual(bars.iloc[0].volume, 4)
        self.assertEqual(bars.iloc[0].trade_count, 2)

    def test_gap_is_not_filled(self):
        chunk = pd.DataFrame(
            [[1672531201, 10, 2], [1672531801, 12, 3]],
            columns=["timestamp", "price", "volume"],
        )
        bars, _ = c02.aggregate_trade_chunks([chunk])
        self.assertEqual(len(bars), 2)
        self.assertEqual((bars.timestamp.iloc[1] - bars.timestamp.iloc[0]), pd.Timedelta(minutes=10))
        gaps = c02.internal_gap_intervals(bars)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.iloc[0].missing_5m_slots, 1)
        self.assertEqual(gaps.iloc[0].gap_start_ts, pd.Timestamp("2023-01-01T00:05:00Z"))

    def test_boundary_rows_excluded_from_selected_archive(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "folder/AAVEUSD.csv",
                    "1672531199,1,1\n1672531200,2,1\n1767225599,3,1\n1767225600,4,1\n",
                )
            spec = c02.ArchiveSpec("fixture", path, c02.FULL_ARCHIVE_URL, c02.TRAIN_START, c02.PROTECTED_START)
            bars, _ = c02.read_pair_archive(spec, "AAVEUSD", None)
            final = c02.finalize_bars([bars], "AAVEUSD")
            self.assertEqual(len(final), 2)
            self.assertTrue(final.timestamp.min() >= c02.TRAIN_START)
            self.assertTrue(final.timestamp.max() < c02.PROTECTED_START)

    def test_archive_chunks_are_consumed_as_stream(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("folder/AAVEUSD.csv", "1672531200,1,1\n1672531500,2,1\n")
            spec = c02.ArchiveSpec("fixture", path, c02.FULL_ARCHIVE_URL, c02.TRAIN_START, c02.PROTECTED_START)
            original = c02.aggregate_trade_chunks

            def assertion(chunks):
                self.assertNotIsInstance(chunks, list)
                return original(chunks)

            with mock.patch.object(c02, "aggregate_trade_chunks", side_effect=assertion):
                bars, _ = c02.read_pair_archive(spec, "AAVEUSD", None)
            self.assertEqual(len(bars), 2)

    def test_ambiguous_member_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("one/AAVEUSD.csv", "1672531200,1,1\n")
                archive.writestr("two/AAVEUSD.csv", "1672531200,1,1\n")
            with zipfile.ZipFile(path) as archive:
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    c02.find_member(archive, "AAVEUSD")

    def test_manifest_hash_is_order_stable(self):
        self.assertEqual(c02.canonical_hash({"a": 1, "b": 2}), c02.canonical_hash({"b": 2, "a": 1}))

    def test_out_of_order_rows_are_reported(self):
        chunk = pd.DataFrame(
            [[1672531202, 10, 2], [1672531201, 11, 3]],
            columns=["timestamp", "price", "volume"],
        )
        _, stats = c02.aggregate_trade_chunks([chunk])
        self.assertEqual(stats["out_of_order_rows"], 1)
        with self.assertRaisesRegex(ValueError, "out-of-order official trades"):
            c02.validate_trade_order(stats, "fixture", "AAVEUSD")


if __name__ == "__main__":
    unittest.main()
