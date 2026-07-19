from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.qlmg_kda03_v1 import (
    extend_causal_features,
    generate_parent_episodes_and_events,
    parent_mask,
)


def raw_frame(periods: int = 80 * 288) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=periods, freq="5min", tz="UTC")
    trend = np.arange(periods, dtype=float)
    return pd.DataFrame({
        "timestamp_utc": ts,
        "basis_decimal": .001 + np.sin(trend / 31) * .0001 + trend * 1e-10,
        "trade_open": 100 + trend * .0002,
        "trade_close": 100.01 + trend * .0002 + np.sin(trend / 17) * .01,
        "mark_open": 100.02 + trend * .0002,
        "mark_close": 100.03 + trend * .0002 + np.cos(trend / 19) * .01,
        "oi_close_base_units": 1000 + trend * .001 + np.sin(trend / 23),
        "liquidation_base_units_5m": .1 + np.abs(np.sin(trend / 29)),
        "eligible": True, "known_lifecycle_mask": True,
        "trade_coverage": True, "mark_coverage": True, "analytics_coverage": True,
    })


def normalized_frame(periods: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=periods, freq="5min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp_utc": ts, "basis_decimal": 0.0, "trade_close": 100.0,
        "mark_close": 100.0, "basis_change_15m": 0.0, "prior_basis_level": 0.0,
        "onset_trade_open": 100.0, "onset_mark_open": 100.0,
        "trade_return_15m": 0.0, "mark_return_15m": 0.0,
        "basis_change_15m_robust_z": 0.0, "basis_change_15m_percentile": .5,
        "prior_basis_level_robust_z": 0.0, "prior_basis_level_percentile": .5,
        "trade_displacement_15m_robust_z": 0.0, "trade_displacement_15m_percentile": .5,
        "mark_displacement_15m_robust_z": 0.0, "mark_displacement_15m_percentile": .5,
        "oi_change_15m_robust_z": 0.0, "oi_change_15m_percentile": .5,
        "liquidation_intensity_15m_robust_z": 0.0,
        "liquidation_intensity_15m_percentile": .5,
        "eligible": True, "known_lifecycle_mask": True,
        "trade_coverage": True, "mark_coverage": True, "analytics_coverage": True,
        "exact_contiguous_15m_valid": True,
    })
    for name in (
        "basis_change_15m", "prior_basis_level", "trade_displacement_15m",
        "mark_displacement_15m", "oi_change_15m", "liquidation_intensity_15m",
    ):
        frame[f"{name}_normalization_valid"] = True
    return frame


def activate(frame: pd.DataFrame, index: int, kind: str, direction: int = 1) -> None:
    frame.loc[index, ["basis_change_15m", "basis_change_15m_robust_z"]] = [direction * .01, direction * 3]
    frame.loc[index, "basis_change_15m_percentile"] = .99 if direction == 1 else .01
    if kind == "catchup":
        frame.loc[index, ["trade_displacement_15m_percentile", "mark_displacement_15m_percentile"]] = [.25, .25]
        frame.loc[index, "oi_change_15m_robust_z"] = 1.0
        frame.loc[index, "liquidation_intensity_15m_robust_z"] = .999
    else:
        frame.loc[index, ["trade_return_15m", "mark_return_15m"]] = [direction * .02, direction * .02]
        frame.loc[index, ["trade_displacement_15m_percentile", "mark_displacement_15m_percentile"]] = [.75, .75]
        frame.loc[index, ["oi_change_15m_robust_z", "oi_change_15m_percentile"]] = [2.0, .95]
        frame.loc[index, "prior_basis_level_robust_z"] = 1.49
        frame.loc[index, "liquidation_intensity_15m_robust_z"] = 1.999


class KDA03FeatureTests(unittest.TestCase):
    def test_exact_window_uses_window_open_and_pre_window_state(self):
        frame = raw_frame(6)
        frame["basis_decimal"] = [0, 1, 2, 3, 4, 5]
        frame["trade_open"] = [100, 101, 102, 103, 104, 105]
        frame["trade_close"] = [100, 101, 102, 106, 104, 105]
        result = extend_causal_features(frame)
        self.assertEqual(result.loc[3, "basis_change_15m"], 3)
        self.assertEqual(result.loc[3, "prior_basis_level"], 0)
        self.assertEqual(result.loc[3, "onset_trade_open"], 101)
        self.assertAlmostEqual(result.loc[3, "trade_return_15m"], 106 / 101 - 1)

    def test_irregular_and_duplicate_timestamps_fail_closed(self):
        frame = raw_frame(6)
        frame.loc[2, "timestamp_utc"] += pd.Timedelta(minutes=1)
        result = extend_causal_features(frame)
        self.assertFalse(result.exact_contiguous_15m_valid.any())
        duplicate = raw_frame(6)
        duplicate.loc[2, "timestamp_utc"] = duplicate.loc[1, "timestamp_utc"]
        with self.assertRaisesRegex(ValueError, "duplicate timestamp"):
            extend_causal_features(duplicate)

    def test_future_and_later_same_day_rows_do_not_change_past_features(self):
        frame = raw_frame()
        cutoff = 70 * 288 + 100
        first = extend_causal_features(frame.iloc[:cutoff].copy())
        changed = frame.copy()
        changed.loc[cutoff:, ["basis_decimal", "liquidation_base_units_5m"]] = [10.0, 1e12]
        second = extend_causal_features(changed)
        for column in (
            "basis_change_15m_robust_z", "basis_change_15m_percentile",
            "trade_displacement_15m_percentile", "oi_change_15m_robust_z",
            "liquidation_intensity_15m_robust_z",
        ):
            pd.testing.assert_series_equal(
                first[column].reset_index(drop=True), second[column].iloc[:cutoff].reset_index(drop=True)
            )

    def test_signed_catchup_boundaries_and_direction(self):
        frame = normalized_frame(2)
        activate(frame, 0, "catchup", 1)
        activate(frame, 1, "catchup", -1)
        self.assertTrue(parent_mask(frame, "primary", "catchup", 1).iloc[0])
        self.assertTrue(parent_mask(frame, "robustness", "catchup", -1).iloc[1])
        frame.loc[0, "oi_change_15m_robust_z"] = 1.000001
        self.assertFalse(parent_mask(frame, "primary", "catchup", 1).iloc[0])

    def test_impulse_exact_boundaries_and_both_price_signs(self):
        frame = normalized_frame(1)
        activate(frame, 0, "impulse", 1)
        self.assertTrue(parent_mask(frame, "primary", "impulse", 1).iloc[0])
        self.assertTrue(parent_mask(frame, "robustness", "impulse", 1).iloc[0])
        frame.loc[0, "mark_return_15m"] = -.02
        self.assertFalse(parent_mask(frame, "primary", "impulse", 1).iloc[0])

    def test_immediate_directions_and_completed_rejection(self):
        frame = normalized_frame()
        activate(frame, 20, "catchup", 1)
        activate(frame, 40, "impulse", 1)
        frame.loc[40, ["prior_basis_level", "onset_trade_open", "onset_mark_open"]] = [0.0, 100.0, 100.0]
        frame.loc[43, ["basis_decimal", "trade_close", "mark_close"]] = [0.0, 99.0, 99.0]
        episodes, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a",
            cohort_hash="c", source_refs="r",
        )
        primary = events[events.attempt.eq("primary")]
        self.assertEqual(primary[primary.mechanism.eq("reference_led_catchup")].trade_direction.tolist(), [-1])
        self.assertEqual(primary[primary.mechanism.eq("basis_impulse_continuation")].trade_direction.tolist(), [1])
        rejection = primary[primary.mechanism.eq("completed_basis_impulse_rejection")]
        self.assertEqual(rejection.trade_direction.tolist(), [-1])
        self.assertEqual(rejection.state_ts.iloc[0], frame.timestamp_utc.iloc[43])
        impulse_episode = episodes[(episodes.attempt.eq("primary")) & episodes.parent_kind.eq("impulse")].iloc[0]
        self.assertEqual(impulse_episode.episode_candidate_count, 2)

    def test_rejection_requires_basis_trade_and_mark_completed_closes(self):
        frame = normalized_frame()
        activate(frame, 20, "impulse", -1)
        frame.loc[20, ["prior_basis_level", "onset_trade_open", "onset_mark_open"]] = [0.0, 100.0, 100.0]
        frame.loc[22, ["basis_decimal", "trade_close", "mark_close"]] = [0.0, 101.0, 99.0]
        episodes, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a",
            cohort_hash="c", source_refs="r",
        )
        self.assertFalse(events.mechanism.eq("completed_basis_impulse_rejection").any())
        self.assertTrue((episodes.rejection_candidate_count == 0).all())

    def test_episode_reset_end_and_deterministic_replay(self):
        frame = normalized_frame(180)
        activate(frame, 20, "impulse", 1)
        activate(frame, 30, "impulse", 1)  # less than 60 quiet minutes: not a new episode
        activate(frame, 100, "impulse", 1)
        first = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a",
            cohort_hash="c", source_refs="r",
        )
        second = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a",
            cohort_hash="c", source_refs="r",
        )
        primary_impulse = first[0][(first[0].attempt.eq("primary")) & first[0].parent_kind.eq("impulse")]
        self.assertEqual(len(primary_impulse), 2)
        self.assertEqual(primary_impulse.iloc[0].episode_active_end_ts, frame.timestamp_utc.iloc[26] + pd.Timedelta(minutes=5))
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])

    def test_pretrain_and_protected_rows_fail_closed(self):
        frame = raw_frame(2)
        frame["timestamp_utc"] = pd.to_datetime(["2022-12-31T23:55Z", "2026-01-01T00:00Z"], utc=True)
        with self.assertRaisesRegex(ValueError, "non-rankable"):
            extend_causal_features(frame)


if __name__ == "__main__":
    unittest.main()
