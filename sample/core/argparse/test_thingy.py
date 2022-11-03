"""
python -m unittest -v test_thingy.py
"""
import sys
import unittest

sys.argv = ["thingy.py", "-a", "123", "-b", "abc"]
from core.argparse.thingy import Thingy, parse_args_f


class ThingyTestCase(unittest.TestCase):
    def test_parser_args_f(self):
        parser = parse_args_f()
        self.assertEqual(parser.arg1, "123")

    def test_parser_args(self):
        parser = Thingy.parse_args(["-a", "123", "-b", "abc"])
        self.assertEqual(parser.arg2, "abc")


if __name__ == "__main__":
    unittest.main()
