import pylint.lint
pylint_opts = ['--disable=line-too-long', 'test.py']
pylint.lint.Run(pylint_opts)
