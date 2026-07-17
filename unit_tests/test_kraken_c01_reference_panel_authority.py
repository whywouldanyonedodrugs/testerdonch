import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_kraken_c01_reference_panel_authority as c01


def candle_payload(times, **extra):
    value = {
        "candles": [
            {"time": time, "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "3"}
            for time in times
        ],
        "more_candles": False,
        **extra,
    }
    return json.dumps(value, separators=(",", ":")).encode()


class C01FinalDayTests(unittest.TestCase):
    def test_exact_request_bounds_and_identity(self):
        url = c01.build_candle_url("PF_XBTUSD", "trade")
        self.assertIn("from=1767139200", url)
        self.assertIn("to=1767225599", url)
        c01.assert_request_identity(url, "PF_XBTUSD", "trade")
        with self.assertRaises(ValueError):
            c01.assert_request_identity(url.replace("PF_XBTUSD", "PF_ETHUSD"), "PF_XBTUSD", "trade")
        with self.assertRaises(ValueError):
            c01.assert_request_identity(url.replace("1767225599", "1767225600"), "PF_XBTUSD", "trade")

    def test_protected_timestamp_rejected_before_normalization(self):
        payload = candle_payload([c01.FROM_MS, c01.PROTECTED_START_MS])
        with self.assertRaisesRegex(ValueError, "protected-period"):
            c01.parse_candle_payload(
                payload, symbol="PF_XBTUSD", tick_type="trade",
                source_url=c01.build_candle_url("PF_XBTUSD", "trade"),
            )

    def test_wrong_response_identity_and_pagination_fail(self):
        url = c01.build_candle_url("PF_XBTUSD", "mark")
        with self.assertRaisesRegex(ValueError, "symbol"):
            c01.parse_candle_payload(
                candle_payload([c01.FROM_MS], symbol="PF_ETHUSD", tick_type="mark"),
                symbol="PF_XBTUSD", tick_type="mark", source_url=url,
            )
        with self.assertRaisesRegex(ValueError, "pagination"):
            c01.parse_candle_payload(
                json.dumps({"candles": json.loads(candle_payload([c01.FROM_MS]))["candles"], "more_candles": True}).encode(),
                symbol="PF_XBTUSD", tick_type="mark", source_url=url,
            )

    def test_duplicate_and_out_of_order_rows_fail(self):
        url = c01.build_candle_url("PF_ETHUSD", "trade")
        for times in ([c01.FROM_MS, c01.FROM_MS], [c01.FROM_MS + 300_000, c01.FROM_MS]):
            with self.assertRaises(ValueError):
                c01.parse_candle_payload(candle_payload(times), symbol="PF_ETHUSD", tick_type="trade", source_url=url)

    def test_gaps_reported_not_filled_and_hash_stable(self):
        url = c01.build_candle_url("PF_ETHUSD", "mark")
        rows = c01.parse_candle_payload(
            candle_payload([c01.FROM_MS, c01.FROM_MS + 600_000]),
            symbol="PF_ETHUSD", tick_type="mark", source_url=url,
        )
        self.assertEqual(len(rows), 2)
        self.assertIn("2025-12-31T00:05:00Z", c01.missing_intervals(rows))
        first = c01.sha256_bytes(c01.canonical_csv_bytes(rows))
        second = c01.sha256_bytes(c01.canonical_csv_bytes(rows))
        self.assertEqual(first, second)
        self.assertTrue(all(row["venue_symbol"] == "PF_ETHUSD" for row in rows))

    def test_trade_and_mark_remain_separate(self):
        trade = c01.parse_candle_payload(
            candle_payload([c01.FROM_MS]), symbol="PF_XBTUSD", tick_type="trade",
            source_url=c01.build_candle_url("PF_XBTUSD", "trade"),
        )
        mark = c01.parse_candle_payload(
            candle_payload([c01.FROM_MS]), symbol="PF_XBTUSD", tick_type="mark",
            source_url=c01.build_candle_url("PF_XBTUSD", "mark"),
        )
        self.assertNotEqual(trade[0]["source_url"], mark[0]["source_url"])


class C01LifecycleTests(unittest.TestCase):
    FIXTURE = b"""
    <html><body><table><tr><th>Symbol</th><th>Settlement Date</th><th>Observation</th><th>Settlement</th></tr>
    <tr><td>PF_DELISTEDUSD*</td><td>1-Jan-2025</td><td>one hour</td><td>1 USD</td></tr></table>
    <p>PF_RESUMEDUSD trading resumed after review.</p></body></html>
    """

    def test_terminal_parser_classifies_delisted_resumed_and_absent(self):
        rows, resumed = c01.parse_terminal_lifecycle_html(self.FIXTURE)
        self.assertEqual(c01.terminal_status("PF_DELISTEDUSD", rows, resumed), "delisted_and_settled_in_official_terminal_ledger")
        self.assertEqual(c01.terminal_status("PF_RESUMEDUSD", rows, resumed), "resumed_after_official_suspension_note")
        self.assertEqual(c01.terminal_status("PF_INCLUDEDUSD", rows, resumed), "not_listed_in_official_terminal_ledger_as_of_access")

    def test_absence_never_becomes_no_outage_claim(self):
        rows, resumed = c01.parse_terminal_lifecycle_html(self.FIXTURE)
        status = c01.terminal_status("PF_XBTUSD", rows, resumed)
        self.assertNotIn("continuous", status)
        self.assertNotIn("no_outage", status)

    def test_no_candidate_return_funding_or_protected_reader(self):
        with patch.object(c01.pd, "read_parquet", side_effect=AssertionError("downstream reader called")) as spy:
            c01.parse_candle_payload(
                candle_payload([c01.FROM_MS]), symbol="PF_XBTUSD", tick_type="trade",
                source_url=c01.build_candle_url("PF_XBTUSD", "trade"),
            )
        spy.assert_not_called()
        imported = set(c01.__dict__)
        for forbidden in ("load_candidate", "load_funding", "compute_return", "score_candidate"):
            self.assertNotIn(forbidden, imported)

    def test_official_instrument_identity_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instruments.json"
            path.write_text(json.dumps({"instruments": [
                {"symbol": "PF_XBTUSD", "type": "flexible_futures", "openingDate": "2022-03-22T13:15:36Z"},
                {"symbol": "PF_ETHUSD", "type": "flexible_futures", "openingDate": "2022-03-22T13:18:45Z"},
            ]}), encoding="utf-8")
            rows = c01.load_official_instruments(path)
            self.assertEqual(set(rows), set(c01.SYMBOLS))

    def test_prior_coverage_authority_must_reconcile_exactly(self):
        fields = [
            "Kraken_symbol", "trade_coverage_start_utc", "trade_coverage_end_utc",
            "mark_coverage_start_utc", "mark_coverage_end_utc", "identity_confidence",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.csv"
            rows = [
                {
                    "Kraken_symbol": symbol,
                    "trade_coverage_start_utc": c01.RANKABLE_START,
                    "trade_coverage_end_utc": "2025-12-31T00:00:00Z",
                    "mark_coverage_start_utc": c01.RANKABLE_START,
                    "mark_coverage_end_utc": "2025-12-31T00:00:00Z",
                    "identity_confidence": "high",
                }
                for symbol in c01.SYMBOLS
            ]
            c01.write_csv(path, rows, fields)
            self.assertEqual(set(c01.load_prior_coverage_authority(path)), set(c01.SYMBOLS))
            rows[0]["trade_coverage_end_utc"] = c01.PROTECTED_START
            c01.write_csv(path, rows, fields)
            with self.assertRaisesRegex(ValueError, "prior authority mismatch"):
                c01.load_prior_coverage_authority(path)


if __name__ == "__main__":
    unittest.main()
