#!/bin/bash
set -euxo pipefail

if [ "${CUDA_VERSION}" = "92" ]; then
  TOOLKIT_PATH="/usr/local/cuda-9.2"
fi
if [ "${CUDA_VERSION}" = "101" ]; then
  TOOLKIT_PATH="/usr/local/cuda-10.1"
fi
if [ "${CUDA_VERSION}" = "102" ]; then
  TOOLKIT_PATH="/usr/local/cuda-10.2"
fi
if [ "${CUDA_VERSION}" = "110" ]; then
  TOOLKIT_PATH="/usr/local/cuda-11.0"
fi
if [ "${CUDA_VERSION}" = "111" ]; then
  TOOLKIT_PATH="/usr/local/cuda-11.1"
fi
if [ "${CUDA_VERSION}" = "cpu" ]; then
  TOOLKIT_PATH=""
fi

echo "${TOOLKIT_PATH}"

