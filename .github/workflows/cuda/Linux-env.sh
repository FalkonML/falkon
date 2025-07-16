#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu128)
    export CUDA_HOME=/usr/local/cuda-12.8
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="5.0+PTX;6.0;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu126)
    export CUDA_HOME=/usr/local/cuda-12.6
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="5.0+PTX;6.0;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu124)
    export CUDA_HOME=/usr/local/cuda-12.4
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="5.0+PTX;6.0;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu121)
    export CUDA_HOME=/usr/local/cuda-12.1
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="5.0+PTX;6.0;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu118)
    export CUDA_HOME=/usr/local/cuda-11.8
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu117)
    export CUDA_HOME=/usr/local/cuda-11.7
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu116)
    export CUDA_HOME=/usr/local/cuda-11.6
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu115)
    export CUDA_HOME=/usr/local/cuda-11.5
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu113)
    export CUDA_HOME=/usr/local/cuda-11.3
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu102)
    export CUDA_HOME=/usr/local/cuda-10.2
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
    ;;
  *)
    ;;
esac
