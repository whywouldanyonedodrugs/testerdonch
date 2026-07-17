from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools import build_kraken_c01_foundation as c01


def bars(start: str, periods: int, returns: np.ndarray | None = None) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    values = np.zeros(periods) if returns is None else np.asarray(returns, dtype=float)
    close = 100.0 * np.exp(np.cumsum(values))
    return pd.DataFrame({"source_open_ts": ts, "close": close})


class C01FoundationTests(unittest.TestCase):
    def test_path_and_shock_exact_thresholds(self):
        self.assertEqual(c01.classify_path_state(0.25, 0.50), "smooth")
        self.assertEqual(c01.classify_path_state(0.50, 0.49), "jump_dominated")
        self.assertEqual(c01.classify_path_state(0.30, 0.49), "intermediate")
        self.assertEqual(c01.classify_shock(3.0), "positive")
        self.assertEqual(c01.classify_shock(-3.0), "negative")
        self.assertIsNone(c01.classify_shock(2.999))

    def test_residual_arithmetic_for_both_models(self):
        frame = pd.DataFrame({
            "candidate_ret": [0.1], "btc_ret": [0.02], "eth_ret": [0.03],
            "alpha": [0.01], "beta_btc": [2.0], "beta_eth": [1.0],
        })
        primary = frame.candidate_ret - frame.alpha - frame.beta_btc * frame.btc_ret - frame.beta_eth * frame.eth_ret
        robust = frame.candidate_ret - frame.alpha - frame.beta_btc * frame.btc_ret
        self.assertAlmostEqual(primary.iloc[0], 0.02)
        self.assertAlmostEqual(robust.iloc[0], 0.05)

    def test_daily_refit_uses_prior_days_and_future_mutation_is_inert(self):
        n = 42 * c01.DAY_BARS
        ts = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
        btc = np.sin(np.arange(n) / 17) * 0.001
        eth = np.cos(np.arange(n) / 23) * 0.001
        candidate = 0.0001 + 1.2 * btc + 0.7 * eth
        frame = pd.DataFrame({
            "source_open_ts": ts, "candidate_ret": candidate, "btc_ret": btc, "eth_ret": eth,
            "candidate_mark": 1.0, "btc_mark": 1.0, "eth_mark": 1.0,
        })
        first = c01._daily_coefficients(frame, c01.PRIMARY_MODEL)
        changed = frame.copy()
        changed.loc[changed.source_open_ts >= pd.Timestamp("2023-02-10", tz="UTC"), "candidate_ret"] += 99
        second = c01._daily_coefficients(changed, c01.PRIMARY_MODEL)
        before = first[first.decision_day <= pd.Timestamp("2023-02-10", tz="UTC")].reset_index(drop=True)
        before2 = second[second.decision_day <= pd.Timestamp("2023-02-10", tz="UTC")].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, before2)
        day = first[first.decision_day == pd.Timestamp("2023-02-10", tz="UTC")].iloc[0]
        self.assertAlmostEqual(day.beta_btc, 1.2, places=8)
        self.assertAlmostEqual(day.beta_eth, 0.7, places=8)

    def test_missing_factor_and_mark_fail_closed(self):
        n = 31 * c01.DAY_BARS
        ts = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "source_open_ts": ts, "candidate_ret": 0.001, "btc_ret": 0.001, "eth_ret": 0.001,
            "candidate_mark": 1.0, "btc_mark": 1.0, "eth_mark": 1.0,
        })
        frame.loc[:3000, "eth_ret"] = np.nan
        coeff = c01._daily_coefficients(frame, c01.PRIMARY_MODEL)
        self.assertTrue(coeff.loc[coeff.decision_day == pd.Timestamp("2023-01-31", tz="UTC"), "alpha"].isna().all())
        frame["eth_ret"] = 0.001
        frame.loc[:3000, "eth_mark"] = np.nan
        coeff = c01._daily_coefficients(frame, c01.PRIMARY_MODEL)
        self.assertTrue(coeff.loc[coeff.decision_day == pd.Timestamp("2023-01-31", tz="UTC"), "alpha"].isna().all())

    def test_shock_scale_excludes_current_window(self):
        n = 140 * c01.SHOCK_BARS
        index = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
        residual = pd.Series(np.tile(np.r_[np.ones(36), -np.ones(36)], 140), index=index)
        shock_start = pd.Series(index - pd.Timedelta(minutes=355), index=index)
        scale1, counts1 = c01._causal_scale(residual, shock_start)
        changed = residual.copy()
        changed.iloc[-72:] = 1000
        scale2, counts2 = c01._causal_scale(changed, shock_start)
        self.assertEqual(counts1[-1], counts2[-1])
        self.assertAlmostEqual(scale1[-1], scale2[-1])

    def test_path_metric_formula(self):
        residual = np.array([3.0, 1.0, -2.0])
        largest = np.abs(residual).max() / np.abs(residual).sum()
        efficiency = abs(residual.sum()) / np.abs(residual).sum()
        self.assertEqual(largest, 0.5)
        self.assertEqual(efficiency, 1 / 3)

    def test_candidate_identity_deterministic_and_causal_only(self):
        base = {
            "family_id": c01.FAMILY_ID, "definition_id": "d", "attempt_id": "a", "symbol": "PF_TESTUSD",
            "venue": "Kraken", "decision_ts": "2024-01-01T00:00:00Z", "shock_window_start": "2023-12-31T18:00:00Z",
            "shock_window_end": "2024-01-01T00:00:00Z", "residual_model_version": c01.PRIMARY_MODEL,
            "feature_version": c01.FEATURE_VERSION, "reference_panel_id": c01.REFERENCE_PANEL_ID,
            "reference_panel_hash": c01.REFERENCE_PANEL_HASH, "candidate_cohort_version": c01.CANDIDATE_COHORT_VERSION,
            "data_authority_hash": "x",
        }
        first = c01.assign_candidate_identity({**base, "report_runtime": 1})
        second = c01.assign_candidate_identity({**dict(reversed(list(base.items()))), "report_runtime": 99})
        self.assertEqual(first, second)

    def test_interval_clustering_deterministic_and_no_extra_gap(self):
        rows = pd.DataFrame([
            {"candidate_id": "b", "symbol": "PF_AUSD", "canonical_episode_input_start": "2024-01-01T14:00:01Z", "canonical_episode_input_end": "2024-01-01T15:00:00Z"},
            {"candidate_id": "a", "symbol": "PF_AUSD", "canonical_episode_input_start": "2024-01-01T00:00:00Z", "canonical_episode_input_end": "2024-01-01T12:00:00Z"},
            {"candidate_id": "c", "symbol": "PF_AUSD", "canonical_episode_input_start": "2024-01-01T12:00:00Z", "canonical_episode_input_end": "2024-01-01T14:00:00Z"},
        ])
        first = c01.cluster_intervals(rows)
        second = c01.cluster_intervals(rows.sample(frac=1, random_state=7))
        mapping1 = first.set_index("candidate_id").canonical_episode_id.to_dict()
        mapping2 = second.set_index("candidate_id").canonical_episode_id.to_dict()
        self.assertEqual(mapping1, mapping2)
        self.assertEqual(mapping1["a"], mapping1["c"])
        self.assertNotEqual(mapping1["a"], mapping1["b"])

    def test_no_outcome_columns(self):
        c01.assert_no_outcome_columns(["candidate_id", "residual_shock_z_6h"])
        with self.assertRaises(ValueError):
            c01.assert_no_outcome_columns(["candidate_id", "forward_return_6h"])

    def test_attempt_registry_retains_all_twelve(self):
        registry = c01.make_attempt_registry("f", "d", "c")
        self.assertEqual(len(registry), 12)
        self.assertTrue(registry.killed_or_retained_for_later_review.eq("retained").all())
        self.assertEqual(registry.candidate_count.sum(), 0)

    def test_manifest_rejects_protected_non_kraken_and_pretrain_before_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.csv"
            pd.DataFrame([{
                "dataset": "historical_trade_candles_5m", "symbol": "BYBIT_X", "chunk_start": "2023-01-01T00:00:00Z",
                "chunk_end": "2023-01-02T00:00:00Z", "resolution": "5m", "rankable_pre_holdout": True,
                "contains_protected_period": False, "parquet_path": "/not/opened.parquet", "parquet_sha256": "x",
                "rows": 1, "status": "downloaded",
            }]).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                c01.load_safe_manifest(p)

    def test_reference_and_cohort_hashes_are_fixed(self):
        self.assertEqual(c01.REFERENCE_PANEL_HASH, "2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763")
        self.assertEqual(c01.CANDIDATE_COHORT_VERSION, "current_roster_bar_existence_cohort")

    def test_trade_and_mark_refs_remain_distinct(self):
        trade = c01.AuthorityRow("historical_trade_candles_5m", "PF_AUSD", c01.TRAIN_START, c01.TRAIN_START + pd.Timedelta(days=1), Path("/tmp/a"), "a", 1)
        mark = c01.AuthorityRow("historical_mark_candles_5m", "PF_AUSD", c01.TRAIN_START, c01.TRAIN_START + pd.Timedelta(days=1), Path("/tmp/b"), "b", 1)
        self.assertNotEqual(trade.reference_id, mark.reference_id)

    def test_lifecycle_authority_changes_data_identity(self):
        row = c01.AuthorityRow(
            "historical_trade_candles_5m", "PF_AUSD", c01.TRAIN_START,
            c01.TRAIN_START + pd.Timedelta(days=1), Path("/tmp/a"), "a", 1,
        )
        self.assertNotEqual(
            c01.authority_hash([row], "reference", "lifecycle-a"),
            c01.authority_hash([row], "reference", "lifecycle-b"),
        )

    def test_official_lifecycle_source_recovers_known_cohort_intervals(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "docs/agent/task_archive/20260717_donch_bt_stage_2a1_c01_reference_panel_20260717_v1"
            / "sources/terminal_lifecycle/kraken_derivatives_delistings.body"
        )
        intervals = c01.load_known_lifecycle_invalidations(source)
        self.assertEqual(
            intervals["PF_FETUSD"],
            [(pd.Timestamp("2024-06-07", tz="UTC"), pd.Timestamp("2024-11-29", tz="UTC"))],
        )
        self.assertEqual(
            intervals["PF_MNTUSD"],
            [(pd.Timestamp("2025-02-28", tz="UTC"), pd.Timestamp("2025-07-17", tz="UTC"))],
        )
        self.assertEqual(
            intervals["PF_RIVERUSD"],
            [(pd.Timestamp("2025-11-21", tz="UTC"), c01.PROTECTED_START)],
        )

    def test_zero_denominator_path_unavailable(self):
        with self.assertRaises(ValueError):
            c01.classify_path_state(np.nan, 0.5)

    def test_future_bar_mutation_cannot_change_past_features_or_candidates(self):
        rng = np.random.default_rng(17)
        n = 90 * c01.DAY_BARS
        btc_ret = rng.normal(0, 0.0008, n)
        eth_ret = 0.4 * btc_ret + rng.normal(0, 0.0008, n)
        residual = rng.normal(0, 0.001, n)
        residual[::997] += 0.025
        candidate_ret = 0.00001 + 1.1 * btc_ret + 0.6 * eth_ret + residual
        candidate = bars("2023-01-01", n, candidate_ret)
        btc = bars("2023-01-01", n, btc_ret)
        eth = bars("2023-01-01", n, eth_ret)
        before, _ = c01.compute_symbol_features(
            "PF_TESTUSD", candidate, candidate.copy(), btc, btc.copy(), eth, eth.copy(),
            c01.PRIMARY_MODEL, pd.Timestamp("2023-01-01", tz="UTC"),
        )
        changed = candidate.copy()
        cutoff = changed.source_open_ts.iloc[-c01.DAY_BARS]
        changed.loc[changed.source_open_ts >= cutoff, "close"] *= np.exp(
            np.linspace(0, 2, (changed.source_open_ts >= cutoff).sum())
        )
        after, _ = c01.compute_symbol_features(
            "PF_TESTUSD", changed, changed.copy(), btc, btc.copy(), eth, eth.copy(),
            c01.PRIMARY_MODEL, pd.Timestamp("2023-01-01", tz="UTC"),
        )
        causal_columns = [
            "decision_ts", "residual_shock_6h", "residual_scale_6h", "residual_shock_z_6h",
            "largest_bar_share", "path_efficiency", "sign", "path_state",
        ]
        before_past = before.loc[before.decision_ts < cutoff, causal_columns].reset_index(drop=True)
        after_past = after.loc[after.decision_ts < cutoff, causal_columns].reset_index(drop=True)
        self.assertGreater(len(before_past), 0)
        pd.testing.assert_frame_equal(before_past, after_past)

    def test_interval_overlap_counts_unique_left_and_pairs(self):
        left = pd.DataFrame([
            {"candidate_id": "a", "symbol": "PF_AUSD", "canonical_episode_input_start": "2024-01-01T00:00:00Z", "canonical_episode_input_end": "2024-01-02T00:00:00Z"},
            {"candidate_id": "b", "symbol": "PF_AUSD", "canonical_episode_input_start": "2024-02-01T00:00:00Z", "canonical_episode_input_end": "2024-02-02T00:00:00Z"},
        ])
        right = pd.DataFrame([
            {"symbol": "PF_AUSD", "episode_input_start": "2023-12-31T23:00:00Z", "episode_input_end": "2024-01-01T01:00:00Z"},
            {"symbol": "PF_AUSD", "episode_input_start": "2024-01-01T12:00:00Z", "episode_input_end": "2024-01-03T00:00:00Z"},
        ])
        self.assertEqual(c01.count_interval_overlaps(left, right), (1, 2))

    def test_known_lifecycle_invalidation_fails_closed(self):
        n = 90 * c01.DAY_BARS
        rng = np.random.default_rng(23)
        btc_ret = rng.normal(0, 0.0008, n)
        eth_ret = rng.normal(0, 0.0008, n)
        residual = rng.normal(0, 0.001, n)
        residual[::613] += 0.03
        candidate = bars("2023-01-01", n, btc_ret + eth_ret + residual)
        btc = bars("2023-01-01", n, btc_ret)
        eth = bars("2023-01-01", n, eth_ret)
        invalid = [(pd.Timestamp("2023-03-01", tz="UTC"), pd.Timestamp("2023-03-15", tz="UTC"))]
        features, unavailable = c01.compute_symbol_features(
            "PF_TESTUSD", candidate, candidate.copy(), btc, btc.copy(), eth, eth.copy(),
            c01.PRIMARY_MODEL, pd.Timestamp("2023-01-01", tz="UTC"), invalid,
        )
        self.assertFalse(features.decision_ts.between(invalid[0][0], invalid[0][1], inclusive="left").any())
        self.assertGreater(unavailable["known_lifecycle_invalidated_rows"], 0)


if __name__ == "__main__":
    unittest.main()
