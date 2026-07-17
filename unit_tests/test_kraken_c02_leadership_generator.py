from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools import build_kraken_c02_leadership_generator as c02


def panel(n: int = 9000, start: str = "2023-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    close = np.exp(np.linspace(0, .05, n))
    return pd.DataFrame({
        "timestamp": ts, "spot_close": close, "spot_volume": 1.0,
        "perp_close": close, "perp_high": close * 1.001, "perp_low": close * .999,
        "perp_volume": 1.0, "mark_close": close,
        "feature_available_ts": ts + pd.Timedelta(minutes=5),
    })


class C02LeadershipGeneratorTests(unittest.TestCase):
    def test_attempt_registry_is_frozen_and_complete(self):
        registry = c02.make_attempt_register()
        self.assertEqual(len(registry), 60)
        self.assertEqual(registry.attempt_id.nunique(), 60)
        self.assertTrue(registry.registered_before_generation.all())

    def test_exact_sparse_intersection_does_not_fill(self):
        spot = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T00:10Z"]),
            "close": [1., 2.], "volume": [1., 1.],
            "source_close_ts": pd.to_datetime(["2024-01-01T00:05Z", "2024-01-01T00:15Z"]),
            "feature_available_ts": pd.to_datetime(["2024-01-01T00:05Z", "2024-01-01T00:15Z"]),
        })
        trade = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
                              "close": [1., 1., 2.], "high": [1., 1., 2.], "low": [1., 1., 2.], "volume": [1., 1., 1.]})
        mark = trade[["timestamp", "close"]].copy()
        out = c02.align_exact(spot, trade, mark)
        self.assertEqual(out.timestamp.tolist(), spot.timestamp.tolist())
        self.assertNotIn(pd.Timestamp("2024-01-01T00:05Z"), out.timestamp.tolist())

    def test_spot_boundary_rejects_non_kraken_and_protected_rows(self):
        base = pd.DataFrame([{
            "timestamp": "2024-01-01T00:00Z", "close": 1, "volume": 1,
            "source_close_ts": "2024-01-01T00:05Z", "feature_available_ts": "2024-01-01T00:05Z",
            "Kraken_spot_pair": "TESTUSD", "venue": "kraken",
        }])
        self.assertEqual(len(c02.validate_spot_frame(base, "TESTUSD")), 1)
        external = base.copy(); external["venue"] = "bybit"
        with self.assertRaises(ValueError): c02.validate_spot_frame(external, "TESTUSD")
        protected = base.copy()
        protected[["timestamp", "source_close_ts", "feature_available_ts"]] = [[
            "2026-01-01T00:00Z", "2026-01-01T00:05Z", "2026-01-01T00:05Z"]]
        with self.assertRaises(ValueError): c02.validate_spot_frame(protected, "TESTUSD")

    def test_exact_15m_return_requires_all_four_consecutive_bars(self):
        frame = panel(5)
        result = c02.complete_return(frame, "spot_close")
        self.assertAlmostEqual(result.iloc[3], np.log(frame.spot_close.iloc[3] / frame.spot_close.iloc[0]))
        broken = frame.drop(index=1).reset_index(drop=True)
        self.assertTrue(c02.complete_return(broken, "spot_close").isna().all())

    def test_prior_day_scale_ignores_current_and_future_day(self):
        frame = panel(9000)
        returns = c02.complete_return(frame, "spot_close")
        cutoff = pd.Timestamp("2023-01-20", tz="UTC")
        before = c02.prior_daily_scale(returns, frame.timestamp).set_index("utc_day")
        changed = returns.copy()
        changed.loc[frame.timestamp >= cutoff] = 100
        after = c02.prior_daily_scale(changed, frame.timestamp).set_index("utc_day")
        self.assertEqual(before.at[cutoff, "scale"], after.at[cutoff, "scale"])

    def test_prior_day_volume_median_ignores_current_day(self):
        frame = panel(9000)
        values = pd.Series(np.arange(len(frame), dtype=float))
        cutoff = pd.Timestamp("2023-01-20", tz="UTC")
        before = c02.prior_daily_median(values, frame.timestamp).set_index("utc_day")
        values.loc[frame.timestamp >= cutoff] = 1e20
        after = c02.prior_daily_median(values, frame.timestamp).set_index("utc_day")
        self.assertEqual(before.at[cutoff, "median"], after.at[cutoff, "median"])

    def test_daily_eligibility_records_top100_and_lifecycle_reasons(self):
        days = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        audit = pd.DataFrame({"utc_day": days, "coverage_eligible": [True, True],
                              "price_unit_equivalence_verified": [True, True]})
        cohort = pd.DataFrame({"utc_day": days, "top_100_eligible": [True, False], "rank": [1, 101]})
        out = c02.complete_daily_eligibility(audit, cohort, [(days[0] - pd.Timedelta(days=1), days[0] + pd.Timedelta(hours=1))])
        self.assertEqual(out.eligibility_reason.tolist(), ["known_lifecycle_invalid_interval", "not_stage2c_daily_top100"])
        self.assertFalse(out.eligible_before_scale.any())

    def test_impulse_threshold_and_first_onset_not_peak(self):
        frame = panel(20)
        frame["spot_r_15m"] = 0.0; frame["perp_r_15m"] = 0.0
        frame["spot_z_15m"] = 0.0; frame["perp_z_15m"] = 0.0
        frame.loc[12, ["spot_r_15m", "perp_r_15m", "spot_z_15m", "perp_z_15m"]] = [1, 1, 3, 1.5]
        frame.loc[13, ["spot_r_15m", "perp_r_15m", "spot_z_15m", "perp_z_15m"]] = [1, 1, 9, 8]
        self.assertEqual(c02.onset_indices(frame), [(12, 1)])

    def test_leadership_crossings_primary_and_robustness(self):
        frame = panel(10)
        frame["spot_z_15m"] = 0.; frame["perp_z_15m"] = 0.
        frame.loc[5:, "spot_z_15m"] = 1.5
        frame.loc[7:, "perp_z_15m"] = 1.5
        self.assertEqual(c02.classify_leadership(frame, 7, 1, 15), "spot_led")
        self.assertEqual(c02.classify_leadership(frame, 7, 1, 30), "spot_led")
        frame.loc[5:, "perp_z_15m"] = 1.5
        self.assertEqual(c02.classify_leadership(frame, 7, 1, 15), "simultaneous")
        frame.loc[:, "perp_z_15m"] = 0
        self.assertEqual(c02.classify_leadership(frame, 7, 1, 15), "ambiguous")

    def test_already_above_at_lookback_boundary_is_ambiguous(self):
        frame = panel(10)
        frame["spot_z_15m"] = 2.; frame["perp_z_15m"] = 2.
        self.assertEqual(c02.classify_leadership(frame, 9, 1, 15), "ambiguous")

    def test_completed_failure_requires_trade_and_mark_close(self):
        frame = panel(80)
        onset_open = frame.timestamp.iloc[20]
        event = pd.DataFrame([{"event_id": "e", "leadership_state": "perp_led", "direction": 1,
                              "impulse_onset_ts": onset_open + c02.BAR, "feature_available_ts": onset_open + c02.BAR}])
        frame.loc[25, "perp_close"] = frame.loc[18:20, "perp_low"].min() * .9
        self.assertTrue(c02.generate_failures(event, frame).empty)
        frame.loc[25, "mark_close"] = frame.loc[18:20, "perp_low"].min() * .9
        failures = c02.generate_failures(event, frame)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures.iloc[0].decision_ts, frame.timestamp.iloc[25] + c02.BAR)

    def test_deterministic_ids_and_episode_merging(self):
        base = pd.DataFrame([
            {"event_id": "a", "PF_symbol": "PF_AUSD", "impulse_start": "2024-01-01T00:00Z", "impulse_onset_ts": "2024-01-01T00:15Z"},
            {"event_id": "b", "PF_symbol": "PF_AUSD", "impulse_start": "2024-01-01T05:00Z", "impulse_onset_ts": "2024-01-01T05:15Z"},
            {"event_id": "c", "PF_symbol": "PF_AUSD", "impulse_start": "2024-01-02T00:00Z", "impulse_onset_ts": "2024-01-02T00:15Z"},
        ])
        first = c02.cluster_episodes(base, pd.DataFrame())
        second = c02.cluster_episodes(base.sample(frac=1, random_state=4), pd.DataFrame())
        self.assertEqual(first.set_index("event_id").canonical_episode_id.to_dict(), second.set_index("event_id").canonical_episode_id.to_dict())
        ids = first.set_index("event_id").canonical_episode_id
        self.assertEqual(ids.a, ids.b); self.assertNotEqual(ids.a, ids.c)

    def test_alignment_diagnostic_reports_state_changes(self):
        exact = pd.DataFrame([{"event_id": "e", "PF_symbol": "PF_AUSD", "direction": 1,
                              "impulse_onset_ts": pd.Timestamp("2024-01-01T00:15Z"), "leadership_state": "spot_led"}])
        shifted = pd.DataFrame([{"event_id": "s", "PF_symbol": "PF_AUSD", "direction": 1,
                                "impulse_onset_ts": pd.Timestamp("2024-01-01T00:20Z"), "leadership_state": "simultaneous"}])
        out = c02.alignment_comparison(exact, {-5: shifted, 5: shifted})
        self.assertTrue(out.same_episode_and_direction.all())
        self.assertFalse(out.same_leadership_state.any())

    def test_no_outcome_schema(self):
        c02.assert_no_outcome_fields(["event_id", "spot_r_15m"])
        with self.assertRaises(ValueError):
            c02.assert_no_outcome_fields(["event_id", "forward_return_6h"])

    def test_future_changes_do_not_alter_past_features(self):
        frame = panel(12000)
        before = c02.add_features(frame)
        cutoff = pd.Timestamp("2023-02-01", tz="UTC")
        changed = frame.copy(); changed.loc[changed.timestamp >= cutoff, "spot_close"] *= 100
        after = c02.add_features(changed)
        cols = ["spot_r_15m", "spot_prior_scale", "spot_z_15m"]
        pd.testing.assert_frame_equal(before.loc[before.timestamp < cutoff, cols], after.loc[after.timestamp < cutoff, cols])


if __name__ == "__main__":
    unittest.main()
