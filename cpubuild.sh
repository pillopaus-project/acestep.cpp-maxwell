#!/bin/bash

#rm -rf build-cpu
#mkdir build-cpu
cd build-cpu

make clean 
 
cmake .. -DGGML_BLAS=ON -DBUILD_SHARED_LIBS=ON

cmake --build . --config Release -j "$(nproc)"
