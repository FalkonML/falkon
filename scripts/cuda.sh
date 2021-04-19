#!/bin/bash

if [ "${IDX}" = "cpu" ]; then
  export FORCE_ONLY_CPU=1
else
  export FORCE_CUDA=1
fi

if [ "${IDX}" = "cu92" ]; then
  export TOOLKIT_PATH="cuda9.2"
fi
if [ "${IDX}" = "cu101" ]; then
  export TOOLKIT_PATH="cuda10.1"
fi
if [ "${IDX}" = "cu102" ]; then
  export TOOLKIT_PATH="cuda10.2"
  # Fix cublas on CUDA 10.2:
  if [ -d "${TOOLKIT_PATH}/targets/x86_64-linux/include" ]; then
    cp -r "${TOOLKIT_PATH}/x86_64-linux/include/*" "${TOOLKIT_PATH}/include/"
  fi
  if [ -d "${TOOLKIT_PATH}/targets/x86_64-linux/lib" ]; then
    cp -r "${TOOLKIT_PATH}/targets/x86_64-linux/lib/*" "${TOOLKIT_PATH}/lib/"
  fi
fi
if [ "${IDX}" = "cu110" ]; then
  export TOOLKIT_PATH="cuda11.0"
fi
if [ "${IDX}" = "cu111" ]; then
  export TOOLKIT_PATH="cuda111"
fi

# TODO Handle CPU
export CUDA_HOME="${TOOLKIT_PATH}"
export PATH="${TOOLKIT_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TOOLKIT_PATH}/lib64/"
nvcc --version
