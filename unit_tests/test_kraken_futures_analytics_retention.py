import json
import unittest

from tools.probe_kraken_futures_analytics_retention import (
    Budget, MAX_BYTES, PROTECTED_START, RequestSpec, build_matrix, build_url,
    classify_response, compare_replay, decision, validate_url,
)


class FuturesAnalyticsRetentionTests(unittest.TestCase):
    def setUp(self):
        self.spec = RequestSpec("x", "PF_XBTUSD", "open-interest", "2023", 3600, 1686787200, 1686873600)

    def test_matrix_is_exact_and_deterministic(self):
        first, second = build_matrix(), build_matrix()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(len({x.request_id for x in first}), 24)

    def test_url_has_explicit_safe_bounds(self):
        url = build_url(self.spec)
        self.assertIn("interval=3600", url)
        self.assertIn("since=1686787200", url)
        self.assertIn("to=1686873600", url)
        validate_url(url)

    def test_current_or_protected_request_rejected_before_reader(self):
        called = False
        def reader():
            nonlocal called
            called = True
        bad = RequestSpec("x", "PF_XBTUSD", "funding", "x", 3600, 1, PROTECTED_START + 1)
        with self.assertRaises(ValueError):
            build_url(bad)
        self.assertFalse(called)
        with self.assertRaises(ValueError):
            validate_url("https://futures.kraken.com/api/charts/v1/analytics/PF_XBTUSD/funding?interval=3600&since=1")

    def test_valid_empty_response(self):
        row = classify_response(self.spec, 200, "application/json", b'{"result":{"timestamp":[],"data":[]},"errors":[]}')
        self.assertEqual(row["classification"], "empty_valid_response")

    def test_valid_rows_seconds_and_milliseconds(self):
        for values in ([1686787200, 1686790800], [1686787200000, 1686790800000]):
            body = json.dumps({"result": {"timestamp": values, "data": [1, 2]}, "errors": []}).encode()
            row = classify_response(self.spec, 200, "application/json", body)
            self.assertEqual(row["classification"], "verified_historical_rows")
            self.assertEqual(row["row_count"], 2)

    def test_ignored_upper_bound_and_protected_payload(self):
        body = json.dumps({"result": {"timestamp": [PROTECTED_START], "data": [999]}, "errors": []}).encode()
        row = classify_response(self.spec, 200, "application/json", body)
        self.assertEqual(row["classification"], "recent_only_or_bound_ignored")
        self.assertEqual(row["protected_2026_rows"], 1)
        self.assertEqual(row["null_nonfinite_count"], "")

    def test_schema_change_is_not_historical_evidence(self):
        row = classify_response(self.spec, 200, "application/json", b'{"unexpected":true}')
        self.assertEqual(row["classification"], "schema_or_unit_ambiguous")

    def test_replay_comparison(self):
        base = {"request_id": "x", "classification": "verified_historical_rows", "schema_keys": "a",
                "row_count": 2, "minimum_timestamp": "a", "maximum_timestamp": "b", "response_sha256": "1"}
        same = dict(base, response_sha256="2")
        result = compare_replay([base], [same])[0]
        self.assertTrue(result["structurally_stable"])
        self.assertFalse(result["response_hash_equal"])

    def test_decision_rules(self):
        rows = [{"classification": "verified_historical_rows", "upper_bound_honored": True}] * 24
        replay = [{"structurally_stable": True}] * 24
        self.assertEqual(decision(rows, replay), "ready_for_bounded_historical_analytics_audit")
        rows[0] = {"classification": "empty_valid_response", "upper_bound_honored": True}
        self.assertEqual(decision(rows, replay), "partial_historical_analytics_requires_review")
        self.assertEqual(decision([{"classification": "request_failed", "upper_bound_honored": False}] * 24, replay),
                         "historical_public_analytics_unavailable")

    def test_request_and_byte_caps(self):
        budget = Budget()
        budget.requests = 48
        with self.assertRaises(RuntimeError):
            budget.charge(0)
        budget = Budget()
        budget.bytes = MAX_BYTES
        with self.assertRaises(RuntimeError):
            budget.charge(1)

    def test_no_economic_output_contract(self):
        forbidden = {"pnl", "return", "mae", "mfe", "sharpe", "rank"}
        self.assertFalse(forbidden & set(classify_response(self.spec, 200, "application/json", b'{"result":{"timestamp":[]},"errors":[]}')))


if __name__ == "__main__":
    unittest.main()
