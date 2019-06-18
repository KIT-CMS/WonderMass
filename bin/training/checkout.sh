#!/bin/bash

virtualenv py2_venv
source py2_venv/bin/activate
which pip
pip install --upgrade pip
pip install tensorflow-gpu
pip install keras
pip install uproot
pip install numpy
pip install matplotlib
pip install root_pandas
pip install sklearn

thisdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $thisdir/setup_env.sh
