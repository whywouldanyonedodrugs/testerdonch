from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_kraken_readiness_repair as repair


class KrakenReadinessRepairTests(unittest.TestCase):
    def test_metadata_qc_false_positive_reclassification(self):
        row = {"dataset": "instruments", "status": "warn", "timestamp_columns": "openingDate;lastTradingTime"}
        out = repair.classify_k0_qc_row(row)
        self.assertEqual(out["classification"], "metadata_false_positive")
        self.assertEqual(out["effective_status"], "pass")

    def test_ticker_snapshot_non_monotone_handling(self):
        row = {"dataset": "tickers", "status": "warn", "timestamp_columns": "lastTime", "non_monotone_timestamps": True}
        out = repair.classify_k0_qc_row(row)
        self.assertEqual(out["classification"], "metadata_false_positive")
        self.assertEqual(out["repair_action"], "snapshot_cross_section_rules")

    def test_funding_post_holdout_fixed_maturity_exclusion(self):
        row = {"venue_symbol": "FF_ETHUSD_260925", "symbol": "FF_ETHUSD_260925", "type": "flexible_futures", "openingDate": "2026-02-20T08:00:00Z", "lastTradingTime": "2026-09-25T08:00:00Z"}
        self.assertEqual(repair.classify_funding_relevance(row, False), "post_holdout_lifecycle_only")

    def test_pre_holdout_perp_missing_funding_blocks(self):
        row = {"venue_symbol": "PF_XBTUSD", "symbol": "PF_XBTUSD", "type": "flexible_futures", "openingDate": "2022-03-22T13:15:36Z", "lastTradingTime": None}
        self.assertEqual(repair.classify_funding_relevance(row, False), "missing_pre_holdout_perpetual_funding")

    def test_high_priority_lane_counts_resolved_not_forced_tier1(self):
        lane, reason, _ = repair.lane_for_high_priority("PD06", analytics_available=False)
        self.assertEqual(lane, "needs_live_capture_substitute")
        self.assertIn(lane, repair.RESOLVED_LANES)
        self.assertIn("microstructure", reason)

    def test_pr08_can_be_tier1_with_cap(self):
        lane, _, _ = repair.lane_for_high_priority("PR08", analytics_available=False)
        self.assertEqual(lane, "compiled_tier1_with_analytics_cap")

    def test_missing_unknown_date_precision(self):
        self.assertEqual(repair.timestamp_precision("2024-01-10"), "date_only")
        self.assertEqual(repair.timestamp_precision("2024-01-10T12:00:00Z"), "exact_datetime")
        self.assertEqual(repair.timestamp_precision("unknown"), "unknown")

    def test_c2_markdown_source_trace(self):
        md = """# X\n\n## Main catalyst database\n\n| event_id | ticker | mechanism_family | mechanism_subtype | direction | event_state | first_public_ts_utc | effective_ts_utc | source |\n|---|---|---|---|---|---|---|---|---|\n| CAT0001 | XRP | legal_regulatory_repricing | ruling | long | confirmed | 2023-07-13 | 2023-07-13 | cite |\n\n### Normalization notes for the schema\n"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c2.md"
            p.write_text(md)
            events, trace = repair.parse_c2_markdown(p)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["timestamp_precision"], "date_only")
        self.assertEqual(len(trace), 1)
        self.assertIn("source_row_or_section_or_page", trace.columns)

    def test_tmux_wrapper_requires_launch(self):
        txt = Path("tools/run_kraken_readiness_repair_tmux.sh").read_text()
        self.assertIn("refusing to launch tmux without --launch-tmux", txt)
        self.assertIn("run_kraken_readiness_repair.py", txt)


if __name__ == "__main__":
    unittest.main()
