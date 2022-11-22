#!/bin/bash

APP=python3
HOME=/opt/$APP
SYSD=/etc/systemd/system

if [[ $2 == tsinghua ]]; then
  INDEX_URL="-i https://pypi.tuna.tsinghua.edu.cn/simple"
fi

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

_enable_service() {
  name=$1
  _create_symlink $HOME/setup/$name $SYSD/$name
  systemctl enable $name
  systemctl daemon-reload
}

_disable_service() {
  name=$1
  systemctl disable $name
  systemctl daemon-reload
  _delete_symlink $SYSD/$name
}

_start_service() {
  name=$1
  systemctl start $name
  systemctl status $name
}

_stop_service() {
  name=$1
  systemctl stop $name
  systemctl status $name
}

init() {
  _mkdir $HOME/data
  _mkdir $HOME/envs
  _mkdir $HOME/share/run

  _create_symlink $HOME/bin/pip3     $HOME/bin/pip
  _create_symlink $HOME/bin/python3  $HOME/bin/python
  _create_symlink $HOME/bin/pydoc3   $HOME/bin/pydoc

  chown -R root:root $HOME
  chmod 755 $HOME

  $HOME/bin/python3 -m pip install --upgrade pip setuptools wheel $INDEX_URL
  $HOME/bin/pip3 install --no-cache-dir -r $HOME/setup/requirements.txt $INDEX_URL
}

deinit() {
  _rmdir $HOME/data
  _rmdir $HOME/envs
  _rmdir $HOME/share/run

  _delete_symlink $HOME/bin/pip
  _delete_symlink $HOME/bin/python
  _delete_symlink $HOME/bin/pydoc

  $HOME/bin/pip3 uninstall --no-cache-dir -r $HOME/setup/requirements.txt $INDEX_URL

  _disable_service jupyter-lab.service
  _disable_service flask.service
}

docker() {
  $HOME/bin/pip3 install --no-cache-dir -r $HOME/setup/requirements-docker.txt $INDEX_URL
}

host() {
  $HOME/bin/pip3 install --no-cache-dir -r $HOME/setup/requirements-host.txt $INDEX_URL
}

_labextensions=(
  @mflevine/jupyterlab_html
  # https://github.com/jupyterlab/jupyterlab-latex
  # https://github.com/mflevine/jupyterlab_html
  # https://github.com/QuantStack/jupyterlab-drawio
  # https://github.com/jupyterlab/jupyter-renderers
  # https://github.com/bokeh/jupyter_bokeh
  # https://github.com/lckr/jupyterlab-variableInspector
  # https://github.com/matplotlib/ipympl
  # https://github.com/microsoft/gather
)

jupyter() {
  $HOME/bin/pip3 install --no-cache-dir -r $HOME/setup/requirements-jupyter.txt $INDEX_URL

  set -x
  for extname in "${_labextensions[@]}"; do
    $HOME/bin/jupyter labextension install $extname --dev-build=False --minimize=False
  done
  set +x

  python3 -m bash_kernel.install

  _enable_service jupyter-lab.service
}

flask() {
  _enable_service flask.service
}

start() {
  _start_service jupyter-lab.service
}

stop() {
  _stop_service jupyter-lab.service
}

show() {
  systemctl status jupyter-lab.service
}

case "$1" in
  init) shift; init $@ ;;
  deinit) shift; deinit $@ ;;
  docker) shift; docker $@ ;;
  host) shift; host $@ ;;
  jupyter) shift; jupyter $@ ;;
  flask) shift; flask $@ ;;
  start) shift; start $@ ;;
  stop) shift; stop $@ ;;
  show) shift; show $@ ;;
  *) SCRIPTNAME="${0##*/}"
    echo "Usage: $SCRIPTNAME {init|deinit|docker|host|jupyter|flask|start|stop|show}"
    exit 3
    ;;
esac

exit 0

# vim: syntax=sh ts=4 sw=4 sts=4 sr noet
