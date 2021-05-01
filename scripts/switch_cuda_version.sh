#!/bin/bash
set -exo pipefail

CUDA_DIR="/usr/local/cuda-$1"

if ! ls "$CUDA_DIR"
then
    echo "folder $CUDA_DIR not found to switch"
fi

echo "Switching symlink to $CUDA_DIR"
mkdir -p /usr/local
rm -rf /usr/local/cuda
ln -s "$CUDA_DIR" /usr/local/cuda

# Using nvcc instead of deducing from cudart version since it's unreliable (was 110 for cuda11.1 and 11.2)
CUDA_VERSION=$(nvcc --version | sed -n 4p | cut -f5 -d" " | cut -f1 -d",")
CUDNN_VERSION=$(find /usr/local/cuda/lib64/libcudnn.so.* | sort | tac | head -1 | rev | cut -d"." -f -3 | rev)
export CUDNN_VERSION

ls -lah /usr/local/cuda

echo "CUDA_VERSION=$CUDA_VERSION"
echo "CUDNN_VERSION=$CUDNN_VERSION"
