#!/bin/bash
set -euxo pipefail

if [ "${CUDA_VERSION}" = "92" ]; then
  TOOLKIT_PATH="/usr/local/cuda9.2"
fi
if [ "${CUDA_VERSION}" = "101" ]; then
  TOOLKIT_PATH="/usr/local/cuda10.1"
fi
if [ "${CUDA_VERSION}" = "102" ]; then
  TOOLKIT_PATH="/usr/local/cuda10.2"
fi
if [ "${CUDA_VERSION}" = "110" ]; then
  TOOLKIT_PATH="/usr/local/cuda11.0"
fi
if [ "${CUDA_VERSION}" = "111" ]; then
  TOOLKIT_PATH="/usr/local/cuda11.1"
fi

echo "${TOOLKIT_PATH}"

