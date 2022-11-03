import sys
import unittest


class SkipTestCase(unittest.TestCase):
    @unittest.skip("demonstrating skipping")
    def test_nothing(self):
        self.fail("shouldn't happen")

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_windows_support(self):
        pass

    def test_maybe_skipped(self):
        self.skipTest("external resource not available")

    @unittest.expectedFailure
    def test_fail(self):
        self.assertEqual(1, 1, "broken")
