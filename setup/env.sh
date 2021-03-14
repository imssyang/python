#!/bin/bash

# python3
export PYTHON3_DISABLE_ENV=yes
#export PYTHONHOME=/opt/python3  # when enable, gdb cannot run.
#export PYTHONPATH=/usr/local/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1
eval "optbin -s /opt/python3/bin"
eval "optman -s /opt/python3/share/man"
eval "optpkg -s /opt/python3/lib/pkgconfig"
eval "optlib -s /opt/python3/lib"
eval "optlib -s /opt/python3/lib/python3.8/config-3.8-x86_64-linux-gnu"
eval "optlib -s /opt/python3/lib/python3.8/lib-dynload"
eval "optlib -s /opt/python3/lib/python3.8/site-packages"

# pipx
export PIPX_BIN_DIR=/opt/python3/bin
export PIPX_HOME=/opt/python3/pipx

# twine
export TWINE_USERNAME=imssyang
export TWINE_PASSWORD=1992@pypi.com

# virtualenv
export VIRTUALENV_CONFIG_FILE=/opt/python3/conf/virtualenv.ini

# virtualenvwrapper
export WORKON_HOME=/opt/python3/envs
if [[ -f /opt/python3/bin/virtualenvwrapper.sh ]]; then
  source /opt/python3/bin/virtualenvwrapper.sh
fi


