#!/bin/bash

source py2_venv/bin/activate

export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

export KERAS_BACKEND=tensorflow
export CUDA_VISIBLE_DEVICES='1'
