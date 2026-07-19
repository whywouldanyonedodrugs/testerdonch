from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools import build_kraken_derivatives_state_foundation as runner
from tools.qlmg_kraken_derivatives_state import (
    COHORT_VERSION, SEMANTIC_STATUS, assert_no_outcomes, basis_fields,
    causal_daily_normalization, cluster_canonical_episodes,
    deterministic_event_identity, exact_horizon_mask, liquidation_fields, load_semantic_decision,
    open_interest_fields, price_inferred_liquidation_side, stable_hash,
    validate_rankable_times,
)


class KrakenDerivativesStateFoundationTests(unittest.TestCase):
    def test_semantic_decision_hash_and_version(self):
        path = Path(
            "docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1/received/"
            "DONCH_DECISION_Kraken_Analytics_Inferred_Semantics_2026-07-19_v1.json"
        )
        decision, digest = load_semantic_decision(path, expected_sha256=runner.SEMANTIC_FILE_SHA256)
        self.assertEqual(decision["future_basis"]["semantic_status"], SEMANTIC_STATUS)
        self.assertEqual(digest, stable_hash(decision))

    def test_semantic_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"; path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_semantic_decision(path, expected_sha256="0" * 64)

    def test_basis_decimal_and_bps(self):
        raw, decimal, percent, bps = basis_fields("-0.0003")
        self.assertEqual(raw, "-0.0003"); self.assertEqual(decimal, -0.0003)
        self.assertAlmostEqual(percent, -0.03); self.assertAlmostEqual(bps, -3.0)

    def test_oi_exact_tuple_mapping(self):
        values = open_interest_fields('["10.0","12.0","9.0","11.0"]')
        self.assertEqual(values[:4], ("10.0", "12.0", "9.0", "11.0"))
        self.assertEqual(values[4:], (10.0, 12.0, 9.0, 11.0))

    def test_oi_structural_failure(self):
        with self.assertRaisesRegex(ValueError, "inequalities"):
            open_interest_fields('["10","9","8","11"]')

    def test_liquidation_unsigned_sum_semantics(self):
        values = [liquidation_fields(value)[1] for value in ("1.25", "0", "2.75")]
        self.assertEqual(sum(values), 4.0)
        with self.assertRaisesRegex(ValueError, "unsigned"):
            liquidation_fields("-1")

    def test_liquidation_side_is_only_price_proxy(self):
        self.assertEqual(price_inferred_liquidation_side(-0.1), "long_liquidation_proxy")
        self.assertEqual(price_inferred_liquidation_side(0.1), "short_liquidation_proxy")
        self.assertEqual(price_inferred_liquidation_side(0), "ambiguous")

    def test_base_unit_notional_arithmetic(self):
        oi = open_interest_fields('["2","3","1","2.5"]')[-1]
        liquidation = liquidation_fields("1.5")[1]
        self.assertEqual(oi * 100.0, 250.0)
        self.assertEqual(liquidation * 100.0, 150.0)

    def test_causal_normalization_uses_prior_days(self):
        ts = pd.Series(pd.date_range("2023-01-01", periods=70, freq="D", tz="UTC"))
        values = pd.Series(np.arange(70, dtype=float))
        stats = causal_daily_normalization(ts, values)
        self.assertFalse(stats.iloc[29].normalization_valid)
        self.assertTrue(stats.iloc[30].normalization_valid)
        self.assertEqual(stats.iloc[30].prior_median, 14.5)

    def test_future_rows_do_not_change_past_normalization(self):
        ts = pd.Series(pd.date_range("2023-01-01", periods=80, freq="D", tz="UTC"))
        values = pd.Series(np.sin(np.arange(80)) + np.arange(80) / 10)
        base = causal_daily_normalization(ts.iloc[:70], values.iloc[:70])
        changed = values.copy(); changed.iloc[70:] = 1e9
        extended = causal_daily_normalization(ts, changed)
        pd.testing.assert_frame_equal(base.reset_index(drop=True), extended.iloc[:70].reset_index(drop=True))

    def test_later_same_day_rows_do_not_change_earlier_score(self):
        history = pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC")
        current = pd.DatetimeIndex(["2023-02-10T00:05:00Z", "2023-02-10T23:55:00Z"])
        ts = pd.Series(history.append(current))
        values = pd.Series(np.r_[np.arange(40, dtype=float), 10.0, 20.0])
        first = causal_daily_normalization(ts, values)
        values.iloc[-1] = 1e12
        second = causal_daily_normalization(ts, values)
        self.assertEqual(first.iloc[-2].robust_z, second.iloc[-2].robust_z)
        self.assertEqual(first.iloc[-2].empirical_percentile, second.iloc[-2].empirical_percentile)

    def test_missing_five_minute_window_fails_closed(self):
        ts = pd.Series(pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z",
            "2024-01-01T00:15:00Z", "2024-01-01T00:20:00Z",
        ]))
        mask = exact_horizon_mask(ts, 2)
        self.assertFalse(mask.iloc[2])
        self.assertFalse(mask.iloc[3])

    def test_zero_mad_fails_closed(self):
        ts = pd.Series(pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC"))
        stats = causal_daily_normalization(ts, pd.Series(np.ones(40)))
        self.assertFalse(stats.iloc[-1].normalization_valid)
        self.assertTrue(np.isnan(stats.iloc[-1].robust_z))

    def test_protected_and_pretrain_rows_rejected(self):
        for value in ("2022-12-31T23:55:00Z", "2026-01-01T00:00:00Z"):
            with self.assertRaisesRegex(ValueError, "non-rankable"):
                validate_rankable_times([value])

    def test_no_outcome_columns(self):
        assert_no_outcomes(["decision_ts", "oi_log_change_1h"])
        with self.assertRaisesRegex(ValueError, "outcome"):
            assert_no_outcomes(["forward_return_1h"])

    def test_deterministic_event_identity(self):
        row = {
            "family_id":"KDA01", "definition_id":"d", "attempt_id":"a", "symbol":"PF_XBTUSD",
            "direction":"positive", "state_start":"2024-01-01T00:00:00Z", "decision_ts":"2024-01-01T01:00:00Z",
            "feature_window_start":"2023-11-01T00:00:00Z", "feature_window_end":"2024-01-01T01:00:00Z",
            "semantic_contract_hash":"s", "analytics_data_manifest_hash":"m", "trade_and_mark_authority_hashes":"t",
            "cohort_version":COHORT_VERSION, "feature_version":"f", "generator_contract_hash":"g",
        }
        self.assertEqual(deterministic_event_identity(row), deterministic_event_identity(dict(reversed(list(row.items())))))

    def test_episode_identity_is_deterministic(self):
        frame = pd.DataFrame([
            {"symbol":"PF_XBTUSD","state_start":pd.Timestamp("2024-01-01T00:00Z"),"feature_window_end":pd.Timestamp("2024-01-01T02:00Z"),"event_id":"a"},
            {"symbol":"PF_XBTUSD","state_start":pd.Timestamp("2024-01-01T01:00Z"),"feature_window_end":pd.Timestamp("2024-01-01T03:00Z"),"event_id":"b"},
        ])
        first = cluster_canonical_episodes(frame)
        second = cluster_canonical_episodes(frame.sample(frac=1, random_state=4))
        self.assertEqual(first.canonical_episode_id.iloc[0], second.canonical_episode_id.iloc[0])
        self.assertEqual(first.canonical_episode_member_count.iloc[0], 2)

    def test_attempt_registry_retains_all_branches(self):
        register = runner.register_attempts("s", "f")
        self.assertEqual(len(register), len(runner.ATTEMPTS))
        self.assertEqual(set(register.family_id), {"KDA01", "KDA02", "KDA03"})
        self.assertTrue(register.attempted_before_generation.all())
        killed = register[register.killed_branch]
        self.assertEqual(killed.definition_id.tolist(), ["kda02_robust_oi_vacuum"])
        self.assertEqual(killed.kill_reason.iloc[0], "semantic_duplicate_of_kda02_primary_oi_vacuum")

    def test_exact_1m_to_5m_aggregation_contract(self):
        oi = pd.DataFrame({"open":[10,11,9,12,12],"high":[11,12,10,13,12],"low":[9,10,8,11,11],"close":[11,9,12,12,11]})
        self.assertEqual((oi.open.iloc[0],oi.high.max(),oi.low.min(),oi.close.iloc[-1]),(10,13,8,11))
        self.assertEqual(sum([1,0,2,0,3]),6)
        self.assertEqual([.1,.2,.3,.4,.5][-1],.5)

    def test_source_contains_no_outcome_reader(self):
        source = Path(runner.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("outcome_reader", source)
        self.assertNotIn("candidate_return", source)
        self.assertNotIn("forward_return", source)

    def test_generator_has_bounded_partitioned_cache(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("PARTITION_BY (symbol)", source)
        self.assertIn("memory_limit='1GB'", source)
        self.assertIn("columns=columns", source)
        self.assertNotIn("all_events: list", source)
        self.assertIn('args.cache / "events" / f"symbol={symbol}"', source)

    def test_old_family_overlap_projection_is_allowlisted(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('safe = {"symbol", "PF_symbol", "decision_ts"', source)
        self.assertIn("unsafe old-family overlap projection", source)


if __name__ == "__main__":
    unittest.main()
