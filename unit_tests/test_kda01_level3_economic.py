from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from tools.qlmg_kda01_level3_economic import branch_side, cluster_bootstrap, equal_cluster_returns, gate_flags, positive_contribution_share, score_open_prices
from tools.run_kda01_level3_economic import FUNDING_ROOT


class KDA01Level3EconomicTests(unittest.TestCase):
    def test_long_short_and_cost_arithmetic(self):
        for actual, expected in zip(score_open_prices(100, 101, 1), (100, 86, 68)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(score_open_prices(100, 99, -1)[0], 100)

    def test_branch_side_frozen(self):
        self.assertEqual(branch_side("primary_positive_efficient_continuation"), 1)
        self.assertEqual(branch_side("robustness_negative_efficient_continuation"), -1)
        self.assertEqual(branch_side("primary_positive_completed_failure"), -1)
        self.assertEqual(branch_side("primary_negative_completed_failure"), 1)

    def test_invalid_price_rejected(self):
        for value in (0, -1, math.nan):
            with self.assertRaises(ValueError): score_open_prices(value, 1, 1)

    def test_equal_market_day_not_trade_weighted(self):
        trades=pd.DataFrame({"definition_id":["d"]*3,"event_id":[1,2,3],"day":["a","a","b"],"gross_bps":[0,100,0],"base_net_bps":[-14,86,-14],"stress_net_bps":[-32,68,-32]})
        result=equal_cluster_returns(trades,"day")
        self.assertEqual(result.trade_count.tolist(),[2,1]); self.assertEqual(result.base_net_bps.tolist(),[36,-14])

    def test_bootstrap_fixed_seed(self):
        a,lo,hi=cluster_bootstrap([1,2,3]); b,lo2,hi2=cluster_bootstrap([1,2,3])
        np.testing.assert_array_equal(a,b); self.assertEqual((lo,hi),(lo2,hi2))

    def test_positive_concentration_and_undefined(self):
        frame=pd.DataFrame({"g":["a","b","c"],"base_net_bps":[4,1,-9]})
        self.assertEqual(positive_contribution_share(frame,"g"),.8)
        self.assertTrue(math.isnan(positive_contribution_share(frame.assign(base_net_bps=-1),"g")))

    def test_exact_gate_boundaries_and_no_rounding(self):
        row={"accepted_count":100,"trades_2023":20,"trades_2024":20,"trades_2025":20,"equal_day_base_mean_bps":1e-12,"equal_day_base_median_bps":1e-12,"bootstrap_lower_bps":-5,"market_day_positive_share":.10,"symbol_positive_share":.25,"year_positive_share":.70,"equal_day_stress_mean_bps":-10}
        self.assertTrue(gate_flags(row)["all_gates_pass"])
        row["equal_day_base_mean_bps"]=-1e-12; self.assertFalse(gate_flags(row)["all_gates_pass"])

    def test_robustness_label_does_not_enter_calculations(self):
        with self.assertRaises(ValueError): branch_side("pooled_primary_robustness")

    def test_funding_authority_is_not_worktree_relative(self):
        self.assertEqual(str(FUNDING_ROOT), "/opt/testerdonch/results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")

    def test_schedule_writer_separates_accepted_and_rejected_records(self):
        source = Path("tools/run_kda01_level3_economic.py").read_text()
        self.assertIn('records.loc[records.accepted].to_parquet(', source)
        self.assertIn('records.loc[~records.accepted].to_parquet(', source)
        self.assertIn('"KDA01_LEVEL3_EXECUTION_REJECTIONS.parquet"', source)

if __name__ == "__main__": unittest.main()
