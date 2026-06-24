from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import config as cfg
from shared_utils import resolve_intrabar_1m_path


class ResolveIntrabar1mPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_primary = cfg.PARQUET_1M_DIR
        self._old_fallback = cfg.PARQUET_1M_FALLBACK_DIR
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.primary = root / "hot"
        self.fallback = root / "cold"
        self.primary.mkdir(parents=True, exist_ok=True)
        self.fallback.mkdir(parents=True, exist_ok=True)
        cfg.PARQUET_1M_DIR = self.primary
        cfg.PARQUET_1M_FALLBACK_DIR = self.fallback

    def tearDown(self) -> None:
        cfg.PARQUET_1M_DIR = self._old_primary
        cfg.PARQUET_1M_FALLBACK_DIR = self._old_fallback
        self._tmp.cleanup()

    def test_prefers_hot_store_when_both_exist(self) -> None:
        hot = self.primary / "BTCUSDT.parquet"
        cold = self.fallback / "BTCUSDT.parquet"
        hot.touch()
        cold.touch()
        self.assertEqual(resolve_intrabar_1m_path("BTCUSDT"), hot)

    def test_falls_back_to_cold_store_when_hot_missing(self) -> None:
        cold = self.fallback / "ETHUSDT.parquet"
        cold.touch()
        self.assertEqual(resolve_intrabar_1m_path("ETHUSDT"), cold)

    def test_returns_none_when_both_missing(self) -> None:
        self.assertIsNone(resolve_intrabar_1m_path("SOLUSDT"))


if __name__ == "__main__":
    unittest.main()
