import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from tools.qlmg_match_feature_builder import enrich_event_pool_with_match_features


class MatchFeatureBuilderTest(unittest.TestCase):
    def test_asof_features_are_pit_and_bucketed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ts = pd.date_range('2025-01-01', periods=400, freq='5min', tz='UTC')
            df = pd.DataFrame({
                'timestamp': ts,
                'open': np.linspace(100, 110, len(ts)),
                'high': np.linspace(101, 111, len(ts)),
                'low': np.linspace(99, 109, len(ts)),
                'close': np.linspace(100, 120, len(ts)),
                'volume': 1000.0,
                'turnover': 2_000_000.0,
                'open_interest': np.linspace(1000, 1200, len(ts)),
                'funding_rate': 0.0002,
            })
            df.to_parquet(root / 'BTCUSDT.parquet', index=False)
            pool = pd.DataFrame({
                'source_row_id':['r1'], 'source_window_id':['w1'], 'event_id':['e1'], 'candidate_id':['c'], 'candidate_key':['c'], 'family':['A3'], 'symbol':['BTCUSDT'],
                'decision_ts':[pd.Timestamp('2025-01-02T12:00:00Z')], 'entry_ts':[pd.Timestamp('2025-01-02T12:05:00Z')], 'exit_ts':[pd.Timestamp('2025-01-02T13:00:00Z')], 'source_net_R':[1.0], 'parent_regime':['up']
            })
            out, cov = enrich_event_pool_with_match_features(pool, bar_root=root)
            self.assertTrue(bool(out['match_feature_pit_ok'].iloc[0]))
            self.assertLessEqual(out['feature_source_ts'].iloc[0], out['decision_ts'].iloc[0])
            self.assertNotEqual(out['volatility_bucket'].iloc[0], 'vol_missing')
            self.assertNotEqual(out['liquidity_tier'].iloc[0], 'liq_missing')
            self.assertEqual(out['funding_bucket'].iloc[0], 'funding_positive')
            self.assertIn(out['oi_bucket'].iloc[0], {'oi_flat_pm5pct','oi_up_5_25pct','oi_surge_gte25pct'})
            self.assertEqual(len(cov), 1)

    def test_missing_symbol_remains_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            pool = pd.DataFrame({
                'source_row_id':['r1'], 'source_window_id':['w1'], 'event_id':['e1'], 'candidate_id':['c'], 'candidate_key':['c'], 'family':['A3'], 'symbol':['MISSINGUSDT'],
                'decision_ts':[pd.Timestamp('2025-01-02T12:00:00Z')], 'entry_ts':[pd.Timestamp('2025-01-02T12:05:00Z')], 'exit_ts':[pd.Timestamp('2025-01-02T13:00:00Z')], 'source_net_R':[1.0], 'parent_regime':['up']
            })
            out, _ = enrich_event_pool_with_match_features(pool, bar_root=Path(td))
            self.assertEqual(out['volatility_bucket'].iloc[0], 'vol_missing')
            self.assertEqual(out['liquidity_tier'].iloc[0], 'liq_missing')
            self.assertEqual(out['funding_bucket'].iloc[0], 'funding_missing')
            self.assertEqual(out['oi_bucket'].iloc[0], 'oi_missing')
            self.assertFalse(bool(out['match_feature_pit_ok'].iloc[0]))


if __name__ == '__main__':
    unittest.main()
