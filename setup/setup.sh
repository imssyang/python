#!/bin/bash

APP=python3
HOME=/opt/$APP

_mkdir() {
  name=$1
  if [[ ! -d $name ]]; then
    mkdir -p $name
  fi
}

_rmdir() {
  name=$1
  if [[ -d $name ]]; then
    rm -rf $name
  fi
}

_create_symlink() {
  src=$1
  dst=$2
  if [[ ! -d $dst ]] && [[ ! -s $dst ]]; then
    ln -s $src $dst
    echo "($APP) create symlink: $src -> $dst"
  fi
}

_delete_symlink() {
  dst=$1
  if [[ -d $dst ]] || [[ -s $dst ]]; then
    rm -rf $dst
    echo "($APP) delete symlink: $dst"
  fi
}

init() {
  _mkdir $HOME/envs

  _create_symlink $HOME/bin/pip3     $HOME/bin/pip
  _create_symlink $HOME/bin/python3  $HOME/bin/python
  _create_symlink $HOME/bin/pydoc3   $HOME/bin/pydoc

  chown -R root:root $HOME
  chmod 755 $HOME

  $HOME/bin/python3 -m pip install --upgrade pip setuptools wheel
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
