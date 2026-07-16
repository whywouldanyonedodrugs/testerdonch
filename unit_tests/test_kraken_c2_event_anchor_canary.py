import unittest

import pandas as pd

from tools import run_kraken_c2_event_anchor_canary as c2


def event(state, first="2024-01-01", confirm="unknown", effective="unknown"):
    return pd.Series({"event_state": state, "first_public_ts_utc": first, "official_confirm_ts_utc": confirm, "effective_ts_utc": effective, "source_confidence": "high"})


class C2EventAnchorCanaryTests(unittest.TestCase):
    def test_confirmed_uses_earliest_verified_public_timestamp(self):
        result = c2.resolve_event_anchor(event("confirmed", "2024-01-02", "2024-01-03"))
        self.assertEqual(result["event_anchor_source"], "first_public_ts_utc")
        self.assertEqual(result["anchor_precision"], "date_only")

    def test_executed_uses_verified_effective_timestamp(self):
        result = c2.resolve_event_anchor(event("executed", "2024-01-01", effective="2024-02-01T12:30:00Z"))
        self.assertEqual(result["event_anchor_source"], "effective_ts_utc")
        self.assertEqual(result["anchor_precision"], "intraday_explicit_utc")

    def test_unknown_does_not_become_timestamp(self):
        result = c2.resolve_event_anchor(event("executed", "unknown"))
        self.assertEqual(result["anchor_policy_status"], "fail")
        self.assertTrue(pd.isna(result["event_anchor_ts"]))

    def test_4h_definitions_only_for_intraday_anchor(self):
        events = pd.DataFrame([
            {"legacy_event_id": "D", "anchor_precision": "date_only"},
            {"legacy_event_id": "I", "anchor_precision": "intraday_explicit_utc"},
        ])
        definitions = c2.build_definitions(events)
        self.assertFalse(((definitions.legacy_event_id == "D") & (definitions.reaction_exclusion == "4h")).any())
        self.assertTrue(((definitions.legacy_event_id == "I") & (definitions.reaction_exclusion == "4h")).any())


if __name__ == "__main__":
    unittest.main()
