import os
import sys
import unittest

from tests.widget import test_widget


def load_tests():
    return unittest.TestSuite(
        [
            test_widget.load_tests(),
        ]
    )


if __name__ == "__main__":
    aaa = os.environ.get("PYTHONPATH")
    print(f"{aaa}")
    result = unittest.TextTestRunner(verbosity=2).run(load_tests())
    if not result.wasSuccessful():
        sys.exit(1)
