#!/bin/bash

#export PYTHONHOME=/opt/python3
#export PYTHONPATH=
#export TWINE_USERNAME=imssyang
#export TWINE_PASSWORD=1992@pypi.com
export VIRTUALENV_CONFIG_FILE=/opt/python3/conf/virtualenv.ini

export PYTHON3_DISABLE_ENV=yes
eval "optbin -s /opt/python3/bin"
eval "optman -s /opt/python3/share/man"
eval "optpkg -s /opt/python3/lib/pkgconfig"
eval "optlib -s /opt/python3/lib"
eval "optlib -s /opt/python3/lib/python3.8/config-3.8-x86_64-linux-gnu"
eval "optlib -s /opt/python3/lib/python3.8/lib-dynload"
eval "optlib -s /opt/python3/lib/python3.8/site-packages"

