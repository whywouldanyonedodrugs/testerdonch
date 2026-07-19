from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.qlmg_kda02_v2 import (
    FEATURE_EXTENSION_HASH,
    GENERATOR_HASH,
    deterministic_episode_id,
    extend_causal_features,
    generate_parent_episodes_and_events,
    hysteresis_mask,
    parent_mask,
    _contiguous,
)


def normalized_frame(rows: int = 150) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp_utc": ts,
        "trade_return_15m": 0.0,
        "mark_return_15m": 0.0,
        "liquidation_base_units_15m": 0.0,
        "liquidation_to_lagged_oi_15m": 0.0,
        "oi_log_change_15m": 0.0,
        "price_displacement_15m": 0.0,
        "liquidation_intensity_15m_robust_z": 0.0,
        "liquidation_intensity_15m_percentile": .50,
        "oi_change_15m_robust_z": 0.0,
        "oi_change_15m_percentile": .50,
        "price_displacement_15m_robust_z": 0.0,
        "price_displacement_15m_percentile": .50,
        "liquidation_intensity_15m_normalization_valid": True,
        "oi_change_15m_normalization_valid": True,
        "price_displacement_15m_normalization_valid": True,
        "exact_contiguous_15m_valid": True,
        "pre_window_oi_close_base_units": 1000.0,
        "oi_close_base_units": 1000.0,
        "eligible": True,
        "known_lifecycle_mask": True,
        "trade_coverage": True,
        "mark_coverage": True,
        "analytics_coverage": True,
        "trade_open": 100.0,
        "trade_high": 101.0,
        "trade_low": 99.0,
        "trade_close": 100.0,
        "mark_open": 100.0,
        "mark_high": 101.0,
        "mark_low": 99.0,
        "mark_close": 100.0,
    })
    return frame


def activate_parent(frame: pd.DataFrame, onset: int = 20, direction: int = -1) -> None:
    frame.loc[onset, ["trade_return_15m", "mark_return_15m"]] = .02 * direction
    frame.loc[onset, "liquidation_intensity_15m_robust_z"] = 2.0
    frame.loc[onset, "liquidation_intensity_15m_percentile"] = .95
    frame.loc[onset, "oi_change_15m_robust_z"] = -2.0
    frame.loc[onset, "oi_change_15m_percentile"] = .05
    frame.loc[onset, "price_displacement_15m_robust_z"] = 1.0
    frame.loc[onset, "price_displacement_15m_percentile"] = .75
    frame.loc[onset:, "oi_close_base_units"] = 900.0
    frame.loc[onset:onset + 8, "liquidation_intensity_15m_robust_z"] = 1.1
    frame.loc[onset:onset + 8, "liquidation_intensity_15m_percentile"] = .80
    frame.loc[onset, "liquidation_intensity_15m_robust_z"] = 2.0
    frame.loc[onset, "liquidation_intensity_15m_percentile"] = .95


class KDA02V2Tests(unittest.TestCase):
    def test_contract_hashes_and_identity_are_stable(self):
        self.assertEqual(len(FEATURE_EXTENSION_HASH), 64)
        self.assertEqual(len(GENERATOR_HASH), 64)
        self.assertEqual(
            deterministic_episode_id("PF_XBTUSD", "primary", -1, "2024-01-01T00:00Z"),
            deterministic_episode_id("PF_XBTUSD", "primary", -1, "2024-01-01T00:00Z"),
        )

    def test_exact_15m_features_and_unit_cancelling_ratio(self):
        ts = pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC")
        frame = pd.DataFrame({
            "timestamp_utc": ts,
            "trade_close": np.arange(100, 140, dtype=float),
            "mark_close": np.arange(200, 240, dtype=float),
            "oi_close_base_units": np.arange(1000, 1040, dtype=float),
            "liquidation_base_units_5m": 2.0,
            "eligible": True, "known_lifecycle_mask": True,
            "trade_coverage": True, "mark_coverage": True, "analytics_coverage": True,
        })
        # Daily rows are deliberately not contiguous five-minute windows.
        result = extend_causal_features(frame)
        self.assertFalse(result.exact_contiguous_15m_valid.any())
        self.assertTrue(result.liquidation_to_lagged_oi_15m.isna().all())
        dense = frame.iloc[:6].copy()
        dense["timestamp_utc"] = pd.date_range("2023-01-01", periods=6, freq="5min", tz="UTC")
        dense["oi_close_base_units"] = [1000, 1000, 1000, 1000, 1000, 1000]
        scored = extend_causal_features(dense)
        self.assertAlmostEqual(scored.loc[3, "liquidation_base_units_15m"], 6.0)
        self.assertAlmostEqual(scored.loc[3, "liquidation_to_lagged_oi_15m"], .006)

    def test_irregular_interior_and_duplicate_timestamps_fail_closed(self):
        irregular = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T00:04Z", "2024-01-01T00:11Z", "2024-01-01T00:15Z"], utc=True),
            "trade_close": [100, 99, 98, 97], "mark_close": [100, 99, 98, 97],
            "oi_close_base_units": [1000, 990, 980, 970], "liquidation_base_units_5m": 1.0,
            "eligible": True, "known_lifecycle_mask": True, "trade_coverage": True,
            "mark_coverage": True, "analytics_coverage": True,
        })
        result = extend_causal_features(irregular)
        self.assertFalse(result.exact_contiguous_15m_valid.any())
        self.assertTrue(result.liquidation_base_units_15m.isna().all())
        duplicate = irregular.copy(); duplicate.loc[1, "timestamp_utc"] = duplicate.loc[0, "timestamp_utc"]
        with self.assertRaisesRegex(ValueError, "duplicate timestamp"):
            extend_causal_features(duplicate)
        microseconds = pd.Series(pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")).astype("datetime64[us, UTC]")
        self.assertTrue(_contiguous(microseconds, 0, 3))

    def test_future_and_same_day_later_rows_cannot_change_prior_features(self):
        ts = pd.date_range("2023-01-01", periods=80 * 288, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "timestamp_utc": ts,
            "trade_close": np.linspace(100, 180, len(ts)),
            "mark_close": np.linspace(100, 179, len(ts)),
            "oi_close_base_units": np.linspace(1000, 1200, len(ts)),
            "liquidation_base_units_5m": np.linspace(0, 5, len(ts)),
            "eligible": True, "known_lifecycle_mask": True,
            "trade_coverage": True, "mark_coverage": True, "analytics_coverage": True,
        })
        cutoff = 70 * 288
        first = extend_causal_features(frame.iloc[:cutoff])
        frame.loc[cutoff:, "liquidation_base_units_5m"] = 1e12
        second = extend_causal_features(frame)
        pd.testing.assert_series_equal(
            first.liquidation_intensity_15m_robust_z.reset_index(drop=True),
            second.liquidation_intensity_15m_robust_z.iloc[:cutoff].reset_index(drop=True),
        )

    def test_parent_exact_boundaries_and_proxy_direction(self):
        frame = normalized_frame(2)
        activate_parent(frame, 0, -1)
        activate_parent(frame, 1, 1)
        self.assertTrue(parent_mask(frame, "primary", -1).iloc[0])
        self.assertTrue(parent_mask(frame, "robustness", 1).iloc[1])
        frame.loc[0, "mark_return_15m"] = .02
        self.assertFalse(parent_mask(frame, "primary", -1).iloc[0])

    def test_hysteresis_is_either_liquidation_or_oi(self):
        frame = normalized_frame(2)
        frame.loc[0, "liquidation_intensity_15m_robust_z"] = 1.0
        frame.loc[1, "oi_change_15m_robust_z"] = -1.0
        self.assertEqual(hysteresis_mask(frame, "primary").tolist(), [True, True])

    def test_continuation_requires_trade_and_mark_extreme_break(self):
        frame = normalized_frame(); activate_parent(frame)
        frame.loc[23, ["trade_close", "mark_close"]] = [98.0, 98.0]
        episodes, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        primary = events[(events.attempt == "primary") & (events.event_type == "active_purge_continuation")]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary.iloc[0].trade_direction, -1)
        self.assertEqual(episodes[episodes.attempt == "primary"].iloc[0].price_inferred_liquidation_side, "long_liquidation_proxy")
        frame.loc[23, "mark_close"] = 100.0
        _, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        self.assertFalse(((events.attempt == "primary") & (events.event_type == "active_purge_continuation")).any())

    def test_reversal_requires_three_bar_liquidation_cooldown_and_both_reclaims(self):
        frame = normalized_frame(); activate_parent(frame)
        frame.loc[29:, ["liquidation_intensity_15m_robust_z", "liquidation_intensity_15m_percentile"]] = [0.0, .50]
        frame.loc[31, ["trade_close", "mark_close"]] = [101.0, 101.0]
        _, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        reversal = events[(events.attempt == "primary") & (events.event_type == "completed_purge_reversal")]
        self.assertEqual(len(reversal), 1)
        self.assertEqual(reversal.iloc[0].trade_direction, 1)
        frame.loc[31, "mark_close"] = 99.0
        _, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        self.assertFalse(((events.attempt == "primary") & (events.event_type == "completed_purge_reversal")).any())

    def test_reversal_retained_oi_materiality_rejects_epsilon_and_accepts_onset_depth_equality(self):
        frame = normalized_frame(); activate_parent(frame)
        frame.loc[29:, ["liquidation_intensity_15m_robust_z", "liquidation_intensity_15m_percentile"]] = [0.0, .50]
        frame.loc[31, ["trade_close", "mark_close"]] = [101.0, 101.0]
        frame.loc[31:, "oi_close_base_units"] = 999.999999
        _, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        self.assertFalse(((events.attempt == "primary") & (events.event_type == "completed_purge_reversal")).any())
        frame.loc[31:, "oi_close_base_units"] = 900.0
        _, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        self.assertTrue(((events.attempt == "primary") & (events.event_type == "completed_purge_reversal")).any())

    def test_oi_vacuum_without_liquidation_cannot_enter_kda02a(self):
        frame = normalized_frame()
        frame.loc[20, ["trade_return_15m", "mark_return_15m"]] = [-.02, -.02]
        frame.loc[20, ["oi_change_15m_robust_z", "oi_change_15m_percentile"]] = [-3.0, .01]
        frame.loc[20, ["price_displacement_15m_robust_z", "price_displacement_15m_percentile"]] = [2.0, .90]
        episodes, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        self.assertTrue(episodes.empty)
        self.assertTrue(events.empty)

    def test_one_continuation_and_one_reversal_per_episode(self):
        frame = normalized_frame(); activate_parent(frame)
        frame.loc[23:24, ["trade_close", "mark_close"]] = [98.0, 98.0]
        frame.loc[29:, ["liquidation_intensity_15m_robust_z", "liquidation_intensity_15m_percentile"]] = [0.0, .50]
        frame.loc[31:32, ["trade_close", "mark_close"]] = [101.0, 101.0]
        episodes, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        primary = events[events.attempt == "primary"]
        self.assertEqual(primary.event_type.value_counts().to_dict(), {"active_purge_continuation": 1, "completed_purge_reversal": 1})
        self.assertEqual(len(episodes[episodes.attempt == "primary"]), 1)

    def test_episode_ends_after_three_outside_bars_and_continuation_is_capped_at_sixty_minutes(self):
        frame = normalized_frame(); activate_parent(frame)
        frame.loc[32, ["trade_close", "mark_close"]] = [98.0, 98.0]
        episodes, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        primary_episode = episodes[episodes.attempt == "primary"].iloc[0]
        self.assertEqual(primary_episode.episode_active_end_ts, frame.timestamp_utc.iloc[31] + pd.Timedelta(minutes=5))
        self.assertFalse(((events.attempt == "primary") & (events.event_type == "active_purge_continuation")).any())

    def test_six_hour_cap_and_sixty_minute_parent_reset(self):
        frame = normalized_frame(180); activate_parent(frame, 20); activate_parent(frame, 100)
        frame.loc[101:, ["liquidation_intensity_15m_robust_z", "liquidation_intensity_15m_percentile"]] = [0.0, .50]
        frame.loc[172, ["trade_close", "mark_close"]] = [101.0, 101.0]
        episodes, events = generate_parent_episodes_and_events(
            frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r"
        )
        self.assertEqual(len(episodes[episodes.attempt == "primary"]), 2)
        first_episode = episodes[(episodes.attempt == "primary")].iloc[0].parent_episode_id
        self.assertFalse(((events.parent_episode_id == first_episode) & (events.state_ts >= frame.timestamp_utc.iloc[92])).any())

    def test_deterministic_replay_primary_and_robustness_separate(self):
        frame = normalized_frame(); activate_parent(frame); frame.loc[23, ["trade_close", "mark_close"]] = [98.0, 98.0]
        first = generate_parent_episodes_and_events(frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r")
        second = generate_parent_episodes_and_events(frame, symbol="PF_TESTUSD", semantic_hash="s", analytics_manifest_hash="a", cohort_hash="c", source_refs="r")
        pd.testing.assert_frame_equal(first[0], second[0]); pd.testing.assert_frame_equal(first[1], second[1])
        self.assertEqual(set(first[0].attempt), {"primary", "robustness"})
        self.assertFalse(first[1].event_id.duplicated().any())

    def test_pretrain_and_protected_rows_fail_closed(self):
        frame = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2022-12-31T23:55Z", "2026-01-01T00:00Z"], utc=True),
            "trade_close": 100.0, "mark_close": 100.0, "oi_close_base_units": 1000.0,
            "liquidation_base_units_5m": 0.0, "eligible": True, "known_lifecycle_mask": True,
            "trade_coverage": True, "mark_coverage": True, "analytics_coverage": True,
        })
        with self.assertRaisesRegex(ValueError, "non-rankable"):
            extend_causal_features(frame)


if __name__ == "__main__":
    unittest.main()
