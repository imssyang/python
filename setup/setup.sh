#!/bin/bash

APP=python3
HOME=/opt/$APP

init() {
  if [[ ! -d $HOME/envs ]]; then
    mkdir $HOME/envs
    chmod 755 $HOME/envs
    echo "mkdir $HOME/envs"
  fi

  $HOME/bin/pip3 install --no-cache-dir -r $HOME/setup/requirements.txt
}

case "$1" in
  init) init ;;
  *) SCRIPTNAME="${0##*/}"
    echo "Usage: $SCRIPTNAME {init}"
    exit 3
    ;;
esac

exit 0

# vim: syntax=sh ts=4 sw=4 sts=4 sr noet
