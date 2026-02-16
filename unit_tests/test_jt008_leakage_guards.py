import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ci_check_leakage_guards import (  # noqa: E402
    check_merge_asof_contract,
    check_regime_filtered_only,
    run_checks,
)


class TestJT008LeakageGuards(unittest.TestCase):
    def test_repo_guards_pass(self):
        root = Path(__file__).resolve().parents[1]
        report = run_checks(root)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(int(report["violation_count"]), 0)

    def test_detects_smoothed_probabilities(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "regime_detector_bad.py"
            p.write_text(
                """
import statsmodels.api as sm

def compute_markov_regime_4h():
    res = object()
    _ = res.smoothed_marginal_probabilities[0]
    return _

def compute_daily_combined_regime():
    res = object()
    _ = res.filtered_marginal_probabilities[0]
    return _
""",
                encoding="utf-8",
            )
            viol = check_regime_filtered_only(p)
            codes = {v.code for v in viol}
            self.assertIn("REGIME_SMOOTHED_FORBIDDEN", codes)
            self.assertIn("REGIME_FILTERED_REQUIRED", codes)

    def test_detects_bad_merge_direction(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad_asof.py"
            p.write_text(
                """
import pandas as pd

def x(a, b):
    return pd.merge_asof(a, b, on="ts", direction="nearest")
""",
                encoding="utf-8",
            )
            viol = check_merge_asof_contract(p, require_tolerance=False)
            codes = {v.code for v in viol}
            self.assertIn("MERGE_ASOF_DIRECTION", codes)
            self.assertIn("MERGE_ASOF_EXACT", codes)


if __name__ == "__main__":
    unittest.main()

