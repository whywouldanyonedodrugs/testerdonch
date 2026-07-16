import gzip
import json
import unittest

from tools import build_kraken_u2_lifecycle_authority as u2


class TestKrakenU2LifecycleAuthority(unittest.TestCase):
    def _payload(self, symbol="pf_xbtusd", opening="2022-03-22T13:15:36Z", tradeable=True):
        return json.dumps({"instruments": [{
            "symbol": symbol,
            "type": "flexible_futures",
            "openingDate": opening,
            "tradeable": tradeable,
        }]}).encode()

    def test_plain_and_gzip_instrument_parser_normalize_symbol(self):
        for payload in (self._payload(), gzip.compress(self._payload())):
            rows = u2.load_instrument_payload(payload)
            self.assertEqual(rows[0]["symbol"], "PF_XBTUSD")

    def test_conflicting_identity_or_unknown_tradeable_fails(self):
        base = {
            "symbol": "PF_XBTUSD", "type": "flexible_futures", "tradeable": True,
            "openingDate": "2022-03-22T13:15:36Z", "observed_at": "2023-01-01T00:00:00Z",
        }
        conflict = {**base, "openingDate": "2022-03-23T00:00:00Z"}
        with self.assertRaisesRegex(ValueError, "conflicting identity"):
            u2.validate_identity_history("PF_XBTUSD", [base, conflict])
        with self.assertRaisesRegex(ValueError, "non-tradeable or unknown"):
            u2.validate_identity_history("PF_XBTUSD", [{**base, "tradeable": None}])

    def test_coverage_intervals_are_sorted_merged_and_gaps_reported(self):
        rows = [
            {"chunk_start": "2023-01-02T00:00:00Z", "chunk_end": "2023-01-03T00:00:00Z"},
            {"chunk_start": "2023-01-01T00:00:00Z", "chunk_end": "2023-01-02T00:00:00Z"},
            {"chunk_start": "2023-01-04T00:00:00Z", "chunk_end": "2023-01-05T00:00:00Z"},
        ]
        start, end, gaps = u2.merge_coverage_intervals(rows)
        self.assertEqual(start, "2023-01-01T00:00:00Z")
        self.assertEqual(end, "2023-01-05T00:00:00Z")
        self.assertEqual(gaps, [("2023-01-03T00:00:00Z", "2023-01-04T00:00:00Z")])

    def test_invalid_or_protected_rankable_intervals_fail_closed(self):
        for start, end in [
            ("2022-12-31T00:00:00Z", "2023-01-02T00:00:00Z"),
            ("2025-12-31T00:00:00Z", "2026-01-02T00:00:00Z"),
            ("2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z"),
        ]:
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "rankable interval rejected"):
                u2.assert_rankable_interval(start, end)

    def test_non_kraken_or_non_perpetual_identity_fails_closed(self):
        for venue, symbol in [("bybit", "PF_XBTUSD"), ("kraken", "PI_XBTUSD")]:
            with self.subTest(venue=venue, symbol=symbol), self.assertRaisesRegex(ValueError, "non-Kraken perpetual"):
                u2.assert_kraken_identity(venue, symbol)

    def test_unknown_lifecycle_and_incomplete_coverage_exclude(self):
        included, reasons = u2.candidate_is_includable(
            official_start="2022-03-22T00:00:00Z",
            claimed_start="2023-01-01T00:00:00Z",
            claimed_end="2026-01-01T00:00:00Z",
            trade_start="2023-01-01T00:00:00Z",
            trade_end="2025-12-31T00:00:00Z",
            mark_start="2023-01-01T00:00:00Z",
            mark_end="2025-12-31T00:00:00Z",
            trade_gaps=[], mark_gaps=[], lifecycle_continuity_verified=False,
        )
        self.assertFalse(included)
        self.assertIn("lifecycle_interval_continuity_unproven", reasons)
        self.assertIn("trade_coverage_incomplete", reasons)
        self.assertIn("mark_coverage_incomplete", reasons)

    def test_official_start_after_claimed_start_excludes(self):
        included, reasons = u2.candidate_is_includable(
            official_start="2024-01-01T00:00:00Z",
            claimed_start="2023-01-01T00:00:00Z",
            claimed_end="2025-01-01T00:00:00Z",
            trade_start="2023-01-01T00:00:00Z", trade_end="2025-01-01T00:00:00Z",
            mark_start="2023-01-01T00:00:00Z", mark_end="2025-01-01T00:00:00Z",
            trade_gaps=[], mark_gaps=[], lifecycle_continuity_verified=True,
        )
        self.assertFalse(included)
        self.assertEqual(reasons, ["official_start_after_claimed_start"])

    def test_cohort_hash_is_deterministic_across_row_and_key_order(self):
        a = {"Kraken_symbol": "PF_XBTUSD", "eligible_start_utc": "2023-01-01T00:00:00Z"}
        b = {"eligible_start_utc": "2023-01-01T00:00:00Z", "Kraken_symbol": "PF_ETHUSD"}
        self.assertEqual(u2.deterministic_cohort_hash([a, b]), u2.deterministic_cohort_hash([dict(reversed(list(b.items()))), dict(reversed(list(a.items())))]) )


if __name__ == "__main__":
    unittest.main()
