from pylint.reporters.text import TextReporter
from pylint.lint import Run

with open("report.out", "w") as f:
  reporter = TextReporter(f)
  Run(["test.py"], reporter=reporter, do_exit=False)
