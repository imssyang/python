#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function run() {
  cd $PROJECT_ROOT
  (
    set -x
    PYTHONPATH=$PROJECT_ROOT/source:$PYTHONPATH \
      python3 source/app/main.py
  )
  set +x
  cd -
}

run $@
