#!/bin/bash

#rm -rf build-cuda
#mkdir build-cuda
cd build-cuda

make clean

cmake .. -DGGML_CUDA=ON  -DGGML_BLAS=ON 

cmake --build . --config Release -j "$(nproc)"
