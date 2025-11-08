#!/bin/bash

if [[ $USER == "ec2-user" ]]; then
  APP_DIR=/opt/ziea/ziea-ai
else
  APP_DIR=~/.aws/starpower/ziea/ziea-ai
fi

debug() {
  set -x
  "$@"
  { set +x; } 2>/dev/null
}

fastapi_run() {
  debug fastapi run ${APP_DIR}/source/api/main.py \
    --host 0.0.0.0 \
    --port 9090
}

gunicorn_run() {
  debug gunicorn --bind 0.0.0.0:9090 \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --threads 1 \
    --timeout 30 \
    --keep-alive 2 \
    --env LANG=C.UTF-8 \
    --env LC_ALL=C.UTF-8 \
    --env LC_LANG=C.UTF-8] \
    --chdir ${APP_DIR} \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    --pythonpath ${APP_DIR}/source \
    api.main:app
}

chatui_install() {
  charui=${APP_DIR}/chatui/build
  debug rm -rf ${APP_DIR}/source/api/static/*
  debug rm -rf ${APP_DIR}/source/api/templates/*
  debug cp -r ${charui}/static/* ${APP_DIR}/source/api/static
  debug cp ${charui}/favicon.ico ${APP_DIR}/source/api/static
  debug cp ${charui}/asset-manifest.json ${APP_DIR}/source/api/static/manifest.json
  debug cp ${charui}/index.html ${APP_DIR}/source/api/templates
}

case "$1" in
  fastapi)
    shift
    fastapi_run $@
    ;;
  gunicorn)
    shift
    gunicorn_run $@
    ;;
  chatui)
    shift
    chatui_install $@
    ;;
  *)
    echo "Usage: ${0##*/} {flask|gunicorn|chatui}"
    ;;
esac

