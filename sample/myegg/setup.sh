#!/bin/bash -

help() {
  python setup.py --help
  python setup.py --help-commands
}

clean() {
  sdist_clean
  bdist_clean
  test_clean
}

sdist() {
  python setup.py sdist
}

sdist_clean() {
  rm -rfv dist
  rm -rfv MANIFEST
}

bdist_egg() {
  python setup.py bdist_egg
}

bdist_wheel() {
  python setup.py bdist_wheel
}

bdist_clean() {
  rm -rfv build/ dist/ *.egg-info/
}

test() {
  python setup.py test
}

test_egg() {
  PYTHONPATH=dist/myegg-1.0.0-py3.8.egg python test/test_run.py
}

test_wheel() {
  PYTHONPATH=dist/myegg-1.0.0-py3-none-any.whl python test/test_run.py
}

test_clean() {
  rm -rf *.egg-info/ my_egg/__pycache__/ test/__pycache__/
}

case "$1" in
  help) help ;;
  clean) clean ;;
  sdist) sdist ;;
  sdist_clean) sdist_clean ;;
  bdist_egg) bdist_egg ;;
  bdist_wheel) bdist_wheel ;;
  bdist_clean) bdist_clean ;;
  test) test ;;
  test_egg) test_egg ;;
  test_wheel) test_wheel ;;
  test_clean) test_clean ;;
  *)
    SCRIPTNAME="${0##*/}"
    echo "Usage: $SCRIPTNAME {help|clean|sdist|sdist_clean|bdist_egg|bdist_wheel|bdist_clean|test|test_egg|test_wheel|test_clean}"
    exit 3
    ;;
esac

exit 0


