#!/bin/bash

# python3
export PYTHON3_DISABLE_ENV=yes
package_paths=(
  /usr/local/lib/python3.8/site-packages
  /opt/python3/lib/python3.8/site-packages
  /opt/python3/sample
  $PYTHONPATH
)
IFS=:$IFS; export PYTHONPATH="${package_paths[*]}"; IFS=${IFS#?}
export PYTHONWARNINGS="ignore::DeprecationWarning"
export PYTHONNOUSERSITE=1
eval "optbin -s /opt/python3/bin"
eval "optman -s /opt/python3/share/man"
eval "optpkg -s /opt/python3/lib/pkgconfig"
eval "optlib -s /opt/python3/lib"
eval "optlib -s /opt/python3/lib/python3.8/config-3.8-x86_64-linux-gnu"
eval "optlib -s /opt/python3/lib/python3.8/lib-dynload"
eval "optlib -s /opt/python3/lib/python3.8/site-packages"

# pip
export PIP_CONFIG_FILE=/opt/python3/conf/pip.conf

# pipx
export PIPX_BIN_DIR=/opt/python3/bin
export PIPX_HOME=/opt/python3/pipx

# twine
export TWINE_USERNAME=imssyang
export TWINE_PASSWORD=1992@pypi.com

# virtualenv
export VIRTUALENV_CONFIG_FILE=/opt/python3/conf/virtualenv.ini

# virtualenvwrapper
export VIRTUALENVWRAPPER_PYTHON=/opt/python3/bin/python3
export WORKON_HOME=/opt/python3/envs
if [[ -f /opt/python3/bin/virtualenvwrapper.sh ]]; then
  source /opt/python3/bin/virtualenvwrapper.sh
fi

# flask
export FLASK_DEBUG=1

# jupyter
export JUPYTER_CONFIG_DIR=/opt/python3/conf/jupyter
export JUPYTER_DATA_DIR=/opt/python3/data/jupyter
export JUPYTER_RUNTIME_DIR=/opt/python3/data/jupyter/runtime

