#!/bin/bash
set -e
BASE=$(pwd)
XETLA_PROJ=$BASE/ipex-extension-example
if [ ! -d "$XETLA_PROJ" ]; then
  cd $BASE
  git clone https://github.com/ESI-SYD/ipex-extension-example.git -b dev/triton-xetla
fi
cd $XETLA_PROJ
git submodule update --init --recursive
python setup.py install
