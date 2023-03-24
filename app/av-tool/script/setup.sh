#!/bin/bash

PROJECT_DIR=/opt/python3/app/av-tool
flask --no-debug --app ${PROJECT_DIR}/source/api run --host 0.0.0.0
