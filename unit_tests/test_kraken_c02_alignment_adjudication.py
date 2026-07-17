import unittest

import pandas as pd

from tools import adjudicate_kraken_c02_alignment as a


def frame(spot, perp, gap_at=None):
    timestamps = pd.date_range("2024-01-01", periods=len(spot), freq="5min", tz="UTC")
    if gap_at is not None:
        timestamps = timestamps.delete(gap_at)
        spot = [x for i, x in enumerate(spot) if i != gap_at]
        perp = [x for i, x in enumerate(perp) if i != gap_at]
    return pd.DataFrame({"timestamp": timestamps, "spot_z_15m": spot, "perp_z_15m": perp})


class ResolutionAwareLeadershipTests(unittest.TestCase):
    def test_exact_ten_minute_boundary_resolves_spot(self):
        data = frame([0, 1.6, 1.7, 1.8], [0, 0, 0, 1.6])
        state, spot, perp = a.resolution_aware_state(data, 3, 1, 15)
        self.assertEqual(state, "resolved_spot_led")
        self.assertEqual(perp - spot, pd.Timedelta(minutes=10))

    def test_five_minute_separation_is_unresolved(self):
        data = frame([0, 0, 1.6, 1.7], [0, 0, 0, 1.6])
        self.assertEqual(a.resolution_aware_state(data, 3, 1, 15)[0], "coincident_or_unresolved")

    def test_same_bar_is_unresolved(self):
        data = frame([0, 0, 0, 1.6], [0, 0, 0, 1.6])
        self.assertEqual(a.resolution_aware_state(data, 3, 1, 15)[0], "coincident_or_unresolved")

    def test_sparse_gap_does_not_create_crossing(self):
        data = frame([0, 0, 1.6, 1.7, 1.8], [0, 0, 0, 0, 1.6], gap_at=2)
        state, spot, _ = a.resolution_aware_state(data, len(data) - 1, 1, 30)
        self.assertEqual(state, "coincident_or_unresolved")
        self.assertIsNone(spot)

    def test_negative_perp_lead_resolves(self):
        data = frame([0, 0, 0, -1.6], [0, -1.6, -1.7, -1.8])
        self.assertEqual(a.resolution_aware_state(data, 3, -1, 15)[0], "resolved_perp_led")

    def test_15m_and_30m_are_frozen_separate_lookbacks(self):
        data = frame([0, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1], [0, 0, 0, 0, 0, 0, 1.6])
        self.assertEqual(a.resolution_aware_state(data, 6, 1, 15)[0], "coincident_or_unresolved")
        self.assertEqual(a.resolution_aware_state(data, 6, 1, 30)[0], "resolved_spot_led")

    def test_transition_categories(self):
        row = pd.Series({"same_episode_and_direction": True, "exact_leadership_state": "simultaneous", "shifted_leadership_state": "spot_led", "spot_shift_minutes": -5})
        self.assertEqual(a.transition_category(row), "simultaneous_to_spot_led_under_minus5m")

    def test_completed_failure_requires_resolved_perp_leadership(self):
        self.assertEqual(a.retained_failure_state("resolved_perp_led", True), "completed_trade_and_mark")
        self.assertEqual(a.retained_failure_state("resolved_perp_led", False), "unconfirmed")
        self.assertEqual(a.retained_failure_state("resolved_spot_led", True), "not_applicable")
        self.assertEqual(a.retained_failure_state("coincident_or_unresolved", True), "not_applicable")

    def test_prohibited_outcome_schema_fails(self):
        with self.assertRaises(ValueError):
            a.assert_safe_schema(["event_id", "post_decision_pnl"])

    def test_identity_is_deterministic(self):
        value = {"source_event_id": "a", "contract_version": a.CONTRACT_VERSION, "leadership_state": "resolved_spot_led"}
        self.assertEqual(a.c02.stable_hash(value), a.c02.stable_hash(dict(reversed(list(value.items())))))


if __name__ == "__main__":
    unittest.main()
