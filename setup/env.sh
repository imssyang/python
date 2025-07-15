#!/bin/bash

# pyenv
export PYENV_ROOT=/opt/python/pyenv
if [[ -f "${PYENV_ROOT:-"$HOMEBREW_PREFIX"}/bin/pyenv" ]]; then
  eval "$(pyenv init - zsh)"
fi

package_paths=(
  /opt/python/sample
  $PYTHONPATH
)
IFS=:$IFS; export PYTHONPATH="${package_paths[*]}"; IFS=${IFS#?}
export PYTHONWARNINGS="ignore::DeprecationWarning"
export PYTHONNOUSERSITE=1

# virtualenvwrapper
export WORKON_HOME=/opt/python/envs
if [[ -f ${WORKON_HOME}/virtualenvwrapper.sh ]]; then
  source ${WORKON_HOME}/virtualenvwrapper.sh
fi
#export VIRTUALENVWRAPPER_PYTHON=/opt/python3/bin/python3

# pipx
#export PIPX_BIN_DIR=/opt/python3/bin
#export PIPX_HOME=/opt/python3/pipx

# twine
#export TWINE_USERNAME=imssyang
#export TWINE_PASSWORD=1992@pypi.com

# virtualenv
#export VIRTUALENV_CONFIG_FILE=/opt/python3/conf/virtualenv.ini

# pytest
export PYTEST_ADDOPTS="-vv --disable-pytest-warnings --durations=0 -s -o log_cli=true --log-level=INFO"

# alias
#alias uwsgitop="uwsgitop localhost:2031"

