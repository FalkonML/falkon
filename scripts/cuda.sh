#!/bin/bash
set -euxo pipefail

if [ "${CUDA_VERSION}" = "92" ]; then
  TOOLKIT_PATH="/opt/cuda9.2"
fi
if [ "${CUDA_VERSION}" = "101" ]; then
  TOOLKIT_PATH="/opt/cuda10.1"
fi
if [ "${CUDA_VERSION}" = "102" ]; then
  TOOLKIT_PATH="/opt/cuda10.2"
  # Fix cublas on CUDA 10.2:
  if [ -d "${TOOLKIT_PATH}/targets/x86_64-linux/include" ]; then
    cp -r "${TOOLKIT_PATH}/x86_64-linux/include/*" "${TOOLKIT_PATH}/include/"
  fi
  if [ -d "${TOOLKIT_PATH}/targets/x86_64-linux/lib" ]; then
    cp -r "${TOOLKIT_PATH}/targets/x86_64-linux/lib/*" "${TOOLKIT_PATH}/lib/"
  fi
fi
if [ "${CUDA_VERSION}" = "110" ]; then
  TOOLKIT_PATH="/opt/cuda11.0"
fi
if [ "${CUDA_VERSION}" = "111" ]; then
  TOOLKIT_PATH="/opt/cuda11.1"
fi

echo "${TOOLKIT_PATH}"
#
#if [ "${CUDA_VERSION}" != "cpu" ]; then
#  export CUDA_HOME="${TOOLKIT_PATH}"
#  export PATH="${TOOLKIT_PATH}/bin:${PATH}"
#  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TOOLKIT_PATH}/lib64/"
#  nvcc --version
#fi
#
#echo "CUDA_HOME"
#echo $CUDA_HOME
#echo "PATH"
#echo $PATH
#echo "LD_LIBRARY_PATH"
#echo $LD_LIBRARY_PATH
