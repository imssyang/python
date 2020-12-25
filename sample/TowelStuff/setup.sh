#!/bin/bash -

help() {
  python setup.py --help
  python setup.py --help-commands
}

register() {
  #
  # @deprecated
  #
  # [Log]
  #   Username: imssyang
  #   Password:
  #   Registering TowelStuff to https://upload.pypi.org/legacy/
  #   Server response (410): Project pre-registration is no longer required or supported, upload your files instead.
  #
  python setup.py register
}

sdist() {
  python setup.py sdist
}

sdist_clean() {
  rm -rfv dist
  rm -rfv MANIFEST
}

sdist_upload() {
  #
  # [distutils/setuptools]
  # Create a file $HOME/.pypirc:
  #   [distutils]
  #   index-servers =
  #       pypi
  #
  #   [pypi]
  #   username: <username>
  #   password: <password>
  #
  # Create a file ~/.pypirc (added pytest optionally):
  #   [distutils]
  #   index-servers =
  #     pypi
  #     pypitest
  #
  #   [pypi]
  #   repository=https://pypi.python.org/pypi
  #   username=your_username
  #   password=your_password
  #
  #   [pypitest]
  #   repository=https://testpypi.python.org/pypi
  #   username=your_username
  #   password=your_password
  #
  # [Log]
  #   reading manifest template 'MANIFEST.in'
  #   writing manifest file 'MANIFEST'
  #   creating sampleTowelStuff-0.1dev
  #   creating sampleTowelStuff-0.1dev/towelstuff
  #   making hard links in sampleTowelStuff-0.1dev...
  #   hard linking LICENSE.txt -> sampleTowelStuff-0.1dev
  #   hard linking README.txt -> sampleTowelStuff-0.1dev
  #   hard linking setup.py -> sampleTowelStuff-0.1dev
  #   hard linking towelstuff/__init__.py -> sampleTowelStuff-0.1dev/towelstuff
  #   creating dist
  #   Creating tar archive
  #   removing 'sampleTowelStuff-0.1dev' (and everything under it)
  #   running upload
  #   Submitting dist/sampleTowelStuff-0.1dev.tar.gz to https://upload.pypi.org/legacy/
  #   Server response (200): OK
  #
  python setup.py sdist upload
}

twine_upload() {
  #
  # [twine]
  #   pip install twine
  #   python setup.py clean sdist
  #   TWINE_USERNAME=me TWINE_PASSWORD=passwd twine upload dist/*
  #   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
  #
  twine upload dist/*
}

bdist_wininst() {
  python setup.py bdist_wininst
}

bdist_clean() {
  rm -rfv build/ dist/
}

case "$1" in
  help) help ;;
  register) register ;;
  sdist) sdist ;;
  sdist_clean) sdist_clean ;;
  sdist_upload) sdist_upload ;;
  twine_upload) twine_upload ;;
  bdist_wininst) bdist_wininst ;;
  bdist_clean) bdist_clean ;;
  *)
    SCRIPTNAME="${0##*/}"
    echo "Usage: $SCRIPTNAME {help|register|sdist|sdist_clean|sdist_upload|twine_upload|bdist_wininst|bdist_clean}"
    exit 3
    ;;
esac

exit 0


