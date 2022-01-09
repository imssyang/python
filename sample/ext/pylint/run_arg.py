import pylint
import sys

sys.argv = ["pylint", "test.py"]
pylint.run_pylint()
