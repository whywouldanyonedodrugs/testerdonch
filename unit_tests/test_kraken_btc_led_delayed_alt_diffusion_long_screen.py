import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools import run_kraken_btc_led_delayed_alt_diffusion_long_screen as screen


class BTCLedDelayedAltDiffusionTests(unittest.TestCase):
    def test_manifest_is_frozen_12_with_two_raw_and_four_selected_policies(self):
        manifest = screen.frozen_manifest()
        self.assertEqual(len(manifest), 12)
        self.assertEqual(manifest.selected_key_policy_hash.nunique(), 4)
        self.assertTrue((manifest.groupby("selected_key_policy_hash").size() == 3).all())
        self.assertEqual(set(manifest.lag_profile), {"moderate_lag", "deep_lag"})
        self.assertEqual(set(manifest.exit_policy), {"fixed_4h", "fixed_8h", "fixed_12h"})

    def test_two_hour_bars_require_24_source_bars_and_completed_right_boundary(self):
        source = inspect.getsource(screen.completed_two_hour)
        self.assertIn('resample("2h", label="right", closed="right")', source)
        self.assertIn("execution_bar_count.eq(24)", source)
        self.assertIn("decision_ts.diff().eq(pd.Timedelta(hours=2))", source)

    def test_btc_impulse_references_are_shifted_and_eth_is_aligned(self):
        source = inspect.getsource(screen.btc_eth_reference_frames)
        self.assertIn("rolling(60, min_periods=60).mean().shift(1)", source)
        self.assertIn("rolling(60, min_periods=60).std().shift(1)", source)
        self.assertIn('how="inner"', source)
        self.assertIn("refs.eth_return_2h.ge(0)", source)

    def test_beta_is_prior_only_min30_and_clipped(self):
        source = inspect.getsource(screen.feature_frames)
        self.assertIn("rolling(60, min_periods=30).cov(frame.btc_return_2h).shift(1)", source)
        self.assertIn("rolling(60, min_periods=30).var().shift(1)", source)
        self.assertIn("frame.beta_raw.clip(0, 3)", source)
        self.assertIn("int(row.beta_observations) < 30", inspect.getsource(screen.cross_section_at_timestamp))

    def test_current_alt_return_does_not_change_current_beta(self):
        ts = pd.date_range("2024-01-01T02:00Z", periods=70, freq="2h")
        base_returns = np.linspace(-0.01, 0.02, len(ts))
        close = 100 * np.cumprod(1 + base_returns)
        alt = pd.DataFrame({"decision_ts": ts, "open": close, "high": close + 1, "low": close - 1, "close": close,
                            "volume": 1.0, "execution_bar_count": 24, "return_2h": base_returns,
                            "daily_source_ts": ts, "atr_14d": 5.0, "ema_10": close, "daily_close": close})
        refs = pd.DataFrame({"btc_source_ts": ts, "btc_close": 100.0, "btc_return_2h": np.linspace(-0.008, 0.015, len(ts)),
                             "btc_prior60_return_mean": 0.0, "btc_prior60_return_std": 0.01, "btc_impulse": False,
                             "eth_source_ts": ts, "eth_return_2h": 0.0, "qualified_btc_impulse": False})
        bars = pd.DataFrame({"ts": ts, "open": close})
        def attach(frame):
            out = frame.copy(); out["parent_state"] = "both_up"; out["parent_source_ts"] = out.decision_ts; return out
        with patch.object(screen, "completed_two_hour", return_value=(alt.copy(), pd.DataFrame(), pd.DataFrame())), patch.object(screen.parent, "attach_parent_state", side_effect=attach):
            first = screen.feature_frames(bars, refs)[0]
        altered = alt.copy(); altered.loc[altered.index[-1], "return_2h"] = 5.0
        with patch.object(screen, "completed_two_hour", return_value=(altered, pd.DataFrame(), pd.DataFrame())), patch.object(screen.parent, "attach_parent_state", side_effect=attach):
            second = screen.feature_frames(bars, refs)[0]
        self.assertAlmostEqual(first.iloc[-1].beta_clipped, second.iloc[-1].beta_clipped)

    def test_lag_profiles_use_frozen_rank_and_residual_thresholds(self):
        moderate = {"residual_return": -0.001, "residual_rank_pct": 0.5, "trailing_residual_std": 0.01}
        deep = {"residual_return": -0.006, "residual_rank_pct": 1/3, "trailing_residual_std": 0.01}
        self.assertTrue(screen.lag_condition(moderate, "moderate_lag"))
        self.assertFalse(screen.lag_condition(moderate, "deep_lag"))
        self.assertTrue(screen.lag_condition(deep, "deep_lag"))

    def test_cross_section_rank_uses_only_exact_pit_universe(self):
        ts = pd.Timestamp("2025-01-01T02:00Z")
        def row(residual):
            return pd.DataFrame([{"decision_ts": ts, "residual_return": residual, "trailing_residual_std": .01,
                                  "beta_clipped": 1.0, "beta_observations": 60, "atr_14d": 1.0}])
        cache = {"A": row(-.02), "B": row(.01), "C": row(-1.0)}
        with patch.object(screen, "pit_universe_symbols", return_value={"A", "B"}):
            ranked = screen.cross_section_at_timestamp(None, pd.DataFrame(), cache, ts, {}, {})
        self.assertEqual(set(ranked.symbol), {"A", "B"})
        self.assertEqual(float(ranked[ranked.symbol.eq("A")].residual_rank_pct.iloc[0]), 0.5)

    def test_parent_projection_nests_known_states_and_fails_closed_on_unknown(self):
        manifest = screen.frozen_manifest()
        raw = pd.DataFrame([
            {"lag_profile":"moderate_lag","raw_signal_address_hash":"up","symbol":"A","entry_ts":pd.Timestamp("2025-01-01",tz="UTC"),"parent_state":"both_up"},
            {"lag_profile":"moderate_lag","raw_signal_address_hash":"down","symbol":"B","entry_ts":pd.Timestamp("2025-01-01",tz="UTC"),"parent_state":"both_down"},
            {"lag_profile":"moderate_lag","raw_signal_address_hash":"unknown","symbol":"C","entry_ts":pd.Timestamp("2025-01-01",tz="UTC"),"parent_state":"unknown"},
        ])
        projected = screen.project_parent_policies(raw, manifest)
        policies = manifest[manifest.lag_profile.eq("moderate_lag")].drop_duplicates("selected_key_policy_hash")
        strict_hash = policies[policies.parent_policy.eq("both_up")].selected_key_policy_hash.iloc[0]
        broad_hash = policies[policies.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        self.assertEqual(set(projected[projected.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash), {"up"})
        self.assertEqual(set(projected[projected.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash), {"up","down"})

    @staticmethod
    def _candidate_rows():
        return pd.DataFrame([
            {"candidate_key":"a","symbol":"A","entry_ts":pd.Timestamp("2025-01-01",tz="UTC")},
            {"candidate_key":"b","symbol":"A","entry_ts":pd.Timestamp("2025-01-01T06:00Z")},
            {"candidate_key":"c","symbol":"A","entry_ts":pd.Timestamp("2025-01-01T13:00Z")},
        ])

    @staticmethod
    def _executor(key, exit_policy):
        hours={"fixed_4h":4,"fixed_8h":8,"fixed_12h":12}[exit_policy]
        return ({**key,"side":"long","decision_ts":key["entry_ts"],"entry_price":100.0,"initial_stop":90.0,"risk_denominator":10.0,
                 "exit_policy":exit_policy,"exit_ts":key["entry_ts"]+pd.Timedelta(hours=hours),"maximum_exit_ts":key["entry_ts"]+pd.Timedelta(hours=hours),
                 "exit_price":101.0,"exit_reason":"fixture","gross_R":.1},None)

    def test_nonoverlap_uses_actual_definition_exit_without_shared_state(self):
        short={"definition_id":"short","exit_policy":"fixed_4h","parameter_vector_hash":"p"}
        long={"definition_id":"long","exit_policy":"fixed_12h","parameter_vector_hash":"q"}
        a,askips,_=screen.simulate_definition(self._candidate_rows(),short,self._executor)
        b,bskips,_=screen.simulate_definition(self._candidate_rows(),long,self._executor)
        self.assertEqual(len(a),3); self.assertEqual(askips,[])
        self.assertEqual([x["candidate_key"] for x in b],["a","c"]); self.assertEqual([x["candidate_key"] for x in bskips],["b"])

    def test_entry_bar_stop_is_included_in_mae_mfe_path(self):
        ts=pd.date_range("2025-01-01",periods=50,freq="5min",tz="UTC")
        bars=pd.DataFrame({"ts":ts,"open":100.0,"high":101.0,"low":99.0,"close":100.0,"volume":1.0})
        bars.loc[0,"low"]=89.0
        frame=pd.DataFrame({"decision_ts":pd.date_range("2025-01-01T02:00Z",periods=2,freq="2h")})
        key={"candidate_key":"k","symbol":"A","side":"long","decision_ts":ts[0],"entry_ts":ts[0],"entry_price":100.0,
             "initial_stop":90.0,"risk_denominator":10.0,"evaluation_window_end":pd.Timestamp("2025-07-01",tz="UTC")}
        indexed,_=screen.execute_event_indexed(key,"fixed_4h",screen.indexed_execution_data(bars,frame))
        scalar,_=screen.execute_event_scalar(key,"fixed_4h",bars,frame)
        self.assertEqual(indexed["exit_ts"],ts[0]); self.assertEqual(indexed["exit_reason"],"completed_alt_2h_impulse_low_stop")
        self.assertAlmostEqual(indexed["mae_R"],-1.1); self.assertAlmostEqual(indexed["mae_R"],scalar["mae_R"])

    def test_control_addresses_are_unique_across_classes(self):
        controls=pd.DataFrame([
            {"definition_id":"d","candidate_key":"a","control_class":screen.CONTROL_CLASSES[0],"control_economic_address_hash":"same"},
            {"definition_id":"d","candidate_key":"b","control_class":screen.CONTROL_CLASSES[1],"control_economic_address_hash":"same"},
        ])
        retained,rejected=screen.deduplicate_control_addresses(controls)
        self.assertEqual((len(retained),len(rejected)),(1,1))

    def test_forbidden_signal_gates_and_nominal_preblock_are_absent(self):
        source=inspect.getsource(screen.enumerate_raw_signals)
        for forbidden in ("funding_gate_policy","open_interest","prior_high","compression","breakout_retest","blocked_until","next_allowed"):
            self.assertNotIn(forbidden,source)
        self.assertIn('"imputed_funding_gate_activated": False',source)


if __name__ == "__main__":
    unittest.main()
