import unittest

from tools.finalize_kraken_futures_analytics_phase_a import pages_for_rows


class FinalizeAnalyticsPhaseATests(unittest.TestCase):
    def test_page_projection_accounts_for_inclusive_duplicate(self):
        self.assertEqual(pages_for_rows(0), 1)
        self.assertEqual(pages_for_rows(2000), 1)
        self.assertEqual(pages_for_rows(2001), 2)
        self.assertEqual(pages_for_rows(10080), 6)


if __name__ == "__main__":
    unittest.main()
