import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import run_kraken_k0_data_foundation as mod


class KrakenK0HelpersTest(unittest.TestCase):
    def test_symbol_parsing(self):
        out = mod.parse_kraken_symbol("PF_XBTUSD")
        self.assertEqual(out["venue_symbol"], "PF_XBTUSD")
        self.assertEqual(out["base_asset"], "XBT")
        self.assertEqual(out["quote_asset"], "USD")
        self.assertEqual(out["display_symbol"], "BTC/USD")

    def test_public_only_guard_blocks_private_and_order(self):
        self.assertTrue(mod.is_public_safe_url("https://futures.kraken.com/derivatives/api/v3/instruments"))
        self.assertFalse(mod.is_public_safe_url("https://futures.kraken.com/derivatives/api/v3/sendorder"))
        self.assertFalse(mod.is_public_safe_url("https://futures.kraken.com/derivatives/api/v3/private/accounts"))
        self.assertFalse(mod.is_public_safe_url("http://futures.kraken.com/derivatives/api/v3/instruments"))

    def test_timestamp_unit_normalization(self):
        recs = [{"time": 1700000000000}, {"time": 1700000060000}]
        earliest, latest, unit = mod.timestamp_range_from_records(recs)
        self.assertEqual(unit, "ms")
        self.assertIn("2023", earliest)
        self.assertIn("2023", latest)

    def test_schema_inference(self):
        schema = mod.infer_schema({"instruments": [{"symbol": "PF_XBTUSD", "tickSize": 0.5}]})
        self.assertIn("symbol", schema)
        self.assertIn("tickSize", schema)

    def test_download_cap_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            args = mod.parse_args(["--run-root", str(root), "--download-official-data", "--download-cap-gb", "0.0001", "--start", "2025-01-01", "--end", "2025-12-31", "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"))
            root.mkdir(parents=True, exist_ok=True)
            (root / "probes").mkdir()
            pd.DataFrame({"endpoint_family": ["candles_trade_1m"], "works": [True], "url": ["https://example.com"]}).to_csv(root / "probes/endpoint_capability_matrix.csv", index=False)
            with self.assertRaises(RuntimeError):
                mod.stage_storage(ctx)

    def test_no_vendor_classification_vocab(self):
        self.assertIn("kraken_progress_with_official_data", mod.NO_VENDOR_CLASSES)
        self.assertNotIn("waiting_for_vendor_data", mod.NO_VENDOR_CLASSES)

    def test_k1_eligibility_false_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            elig = mod.k1_eligibility(ctx)
            self.assertFalse(elig["k1_can_start"])

    def test_k1_eligibility_requires_oi_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram"])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            (root / "instrument_master").mkdir(parents=True)
            (root / "panels").mkdir()
            (root / "download").mkdir()
            (root / "qc").mkdir()
            (root / "universe").mkdir()
            pd.DataFrame({"venue_symbol": ["PF_XBTUSD"]}).to_parquet(root / "instrument_master/kraken_instrument_master.parquet", index=False)
            pd.DataFrame({"venue_symbol": ["PF_XBTUSD"], "dataset": ["candles"]}).to_parquet(root / "panels/kraken_k0_panel.parquet", index=False)
            pd.DataFrame({"status": ["downloaded"], "dataset": ["candles"]}).to_csv(root / "download/download_manifest.csv", index=False)
            pd.DataFrame({"status": ["pass"]}).to_csv(root / "qc/qc_summary.csv", index=False)
            pd.DataFrame({"venue_symbol": ["PF_XBTUSD", "PF_ETHUSD"], "tier": ["K-A", "K-B"]}).to_csv(root / "universe/kraken_universe_summary.csv", index=False)
            pd.DataFrame({
                "data_family": ["candles", "historical_funding", "analytics_open_interest"],
                "tier1_usability": [True, True, False],
            }).to_csv(root / "download/kraken_retention_depth_matrix.csv", index=False)
            self.assertFalse(mod.k1_eligibility(ctx)["k1_can_start"])
            ticker_dir = root / "downloaded_official_kraken/parquet/tickers"
            ticker_dir.mkdir(parents=True)
            pd.DataFrame({"symbol": ["PF_XBTUSD"], "openInterest": [123.0]}).to_parquet(ticker_dir / "tickers.parquet", index=False)
            self.assertFalse(mod.k1_eligibility(ctx)["k1_can_start"])
            pd.DataFrame({
                "dataset": ["historical_trade_candles_5m", "historical_trade_candles_5m"],
                "symbol": ["PF_XBTUSD", "PF_ETHUSD"],
                "status": ["downloaded", "downloaded"],
                "approx_coverage_days": [1200.0, 1200.0],
                "approx_rankable_pre_holdout_days": [1100.0, 1100.0],
            }).to_csv(root / "download/historical_bar_backfill_summary.csv", index=False)
            self.assertTrue(mod.k1_eligibility(ctx)["k1_can_start"])

    def test_backfill_window_allows_today_and_marks_pre_holdout_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            args = mod.parse_args(["--run-root", str(root), "--disable-telegram", "--historical-bar-backfill", "--backfill-start", "2023-01-01", "--backfill-end", "2026-06-30"])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            start, end = mod.backfill_window(ctx)
            self.assertEqual(str(start), "2023-01-01 00:00:00+00:00")
            self.assertEqual(str(end), "2026-06-30 00:00:00+00:00")
            self.assertGreater(mod.pre_holdout_seconds(start, end), 0)

    def test_backfill_symbol_override_and_chunk_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            args = mod.parse_args([
                "--run-root", str(root),
                "--disable-telegram",
                "--historical-bar-backfill",
                "--backfill-resolution", "5m",
                "--backfill-chunk-hours", "999",
                "--backfill-symbols", "PF_XBTUSD,PF_ETHUSD",
            ])
            ctx = mod.RunContext(args=args, run_root=root, notifier=mod.RunNotifier(root, disabled=True), start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2025-02-01", tz="UTC"))
            self.assertEqual(mod.backfill_symbols(ctx), ["PF_XBTUSD", "PF_ETHUSD"])
            self.assertLessEqual(mod.effective_backfill_chunk_hours(ctx), 158)

    def test_historical_chunk_key_and_disk_status(self):
        key = mod.historical_chunk_key(
            "historical_trade_candles_5m",
            "PF_XBTUSD",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
            "5m",
        )
        self.assertIn("PF_XBTUSD", key)
        self.assertIn("historical_trade_candles_5m", key)
        self.assertIn("disk_free_gb=", mod.disk_status_line(Path(".")))

    def test_tmux_wrapper_has_nice_ionice_and_launch_gate(self):
        text = Path("tools/run_kraken_k0_data_foundation_tmux.sh").read_text()
        self.assertIn("--launch-tmux", text)
        self.assertIn("ionice", text)
        self.assertIn("run_kraken_k0_data_foundation.py", text)


if __name__ == "__main__":
    unittest.main()
