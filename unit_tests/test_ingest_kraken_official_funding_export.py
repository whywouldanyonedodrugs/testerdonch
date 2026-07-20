import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.ingest_kraken_official_funding_export import (
    decimal_rate, parse_hour, safe_member_kind,
)


class FundingExportIngestionTests(unittest.TestCase):
    def test_exact_utc_hour_and_decimal_contract(self):
        self.assertEqual(parse_hour(b"2025-12-31 23:00:00").hour, 23)
        with self.assertRaises(ValueError):
            parse_hour(b"2025-12-31 23:01:00")
        self.assertEqual(str(decimal_rate(b"-0.0000000100")), "-1.00E-8")
        with self.assertRaises(ValueError):
            decimal_rate(b"NaN")

    def test_zip_security_rejects_traversal_symlink_and_encryption(self):
        traversal = zipfile.ZipInfo("../evil.csv")
        with self.assertRaises(RuntimeError):
            safe_member_kind(traversal)
        symlink = zipfile.ZipInfo("exports/PF_TESTUSD.csv")
        symlink.external_attr = 0o120777 << 16
        with self.assertRaises(RuntimeError):
            safe_member_kind(symlink)
        encrypted = zipfile.ZipInfo("exports/PF_TESTUSD.csv")
        encrypted.flag_bits = 1
        with self.assertRaises(RuntimeError):
            safe_member_kind(encrypted)

    def test_expected_payload_and_appledouble_classification(self):
        self.assertEqual(safe_member_kind(zipfile.ZipInfo("exports/PF_TESTUSD.csv")), "funding_csv")
        self.assertEqual(safe_member_kind(zipfile.ZipInfo("__MACOSX/exports/._PF_TESTUSD.csv")), "appledouble_metadata_excluded")
        self.assertEqual(safe_member_kind(zipfile.ZipInfo("exports/.DS_Store")), "finder_metadata_excluded")
        with self.assertRaises(RuntimeError):
            safe_member_kind(zipfile.ZipInfo("exports/readme.txt"))


if __name__ == "__main__":
    unittest.main()
