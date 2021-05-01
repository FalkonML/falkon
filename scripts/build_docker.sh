#!/bin/bash

set -xeou pipefail

TOPDIR=$(git rev-parse --show-toplevel)

CUDA_VERSION=${CUDA_VERSION:-10.2}
case ${CUDA_VERSION} in
  cpu)
    BASE_TARGET=base
    DOCKER_TAG=cpu
    ;;
  all)
    BASE_TARGET=all_cuda
    DOCKER_TAG=latest
    ;;
  *)
    BASE_TARGET=cuda${CUDA_VERSION}
    DOCKER_TAG=cuda${CUDA_VERSION}
    ;;
esac

docker build \
  --target final \
  --build-arg "BASE_TARGET=${BASE_TARGET}" \
  --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
  --tag "falkon/build:${DOCKER_TAG}" \
  -f "${TOPDIR}/Dockerfile" \
  ${TOPDIR}
