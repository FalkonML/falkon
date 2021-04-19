#!/bin/bash

if [ "$CUDA_VERSION" = "cpu" ]; then
  export TOOLKIT=cpuonly
fi

if [ "$CUDA_VERSION" = "cu92" ]; then
  export CUDA_SHORT=9.2
  export CUDA=9.2.148-1
  export UBUNTU_VERSION=ubuntu1604
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "$CUDA_VERSION" = "cu101" ]; then
  export CUDA_SHORT=10.1
  export CUDA=10.1.243-1
  export UBUNTU_VERSION=ubuntu1804
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi


if [ "$CUDA_VERSION" = "cu102" ]; then
  export CUDA_SHORT=10.2
  export CUDA=10.2.89-1
  export UBUNTU_VERSION=ubuntu1804
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "$CUDA_VERSION" = "cu110" ]; then
  export CUDA=11.0.3-450.51.06-1
  export CUDA_MED=11.0.3
  export CUDA_SHORT=11.0
  export UBUNTU_VERSION=ubuntu1804
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "$CUDA_VERSION" = "cu111" ]; then
  export CUDA=11.1.1-455.32.00-1
  export CUDA_MED=11.1.1
  export CUDA_SHORT=11.1
  export UBUNTU_VERSION=ubuntu1804
  export TOOLKIT="cudatoolkit=${CUDA_SHORT}"
fi

if [ "${IDX}" = "cpu" ]; then
  export FORCE_ONLY_CPU=1
else
  export FORCE_CUDA=1
fi

if [ "${TRAVIS_OS_NAME}" = "linux" ] && { [ "${CUDA_VERSION}" = "cu92" ] || [ "${CUDA_VERSION}" = "cu101" ] || [ "${CUDA_VERSION}" = "cu102" ]; }; then
  INSTALLER="cuda-repo-${UBUNTU_VERSION}_${CUDA}_amd64.deb"
  wget -nv "http://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/${INSTALLER}"
  sudo dpkg -i "${INSTALLER}"
  wget -nv "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/7fa2af80.pub"
  sudo apt-key add 7fa2af80.pub
  sudo apt update -qq
  sudo apt install "cuda-core-${CUDA_SHORT/./-}" "cuda-nvcc-${CUDA_SHORT/./-}" "cuda-libraries-dev-${CUDA_SHORT/./-}"
  sudo apt clean
  CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
  LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
  PATH=${CUDA_HOME}/bin:${PATH}
  nvcc --version

  # Fix cublas on CUDA 10.2:
  if [ -d "/usr/local/cuda-10.2/targets/x86_64-linux/include" ]; then
    sudo cp -r /usr/local/cuda-10.2/targets/x86_64-linux/include/* "${CUDA_HOME}/include/"
  fi
  if [ -d "/usr/local/cuda-10.2/targets/x86_64-linux/lib" ]; then
    sudo cp -r /usr/local/cuda-10.2/targets/x86_64-linux/lib/* "${CUDA_HOME}/lib/"
  fi
fi

if [ "${CUDA_VERSION}" = "cu110" ] || [ "${CUDA_VERSION}" = "cu111"]; then
  wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/cuda-${UBUNTU_VERSION}.pin
  sudo mv cuda-${UBUNTU_VERSION}.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget -nv https://developer.download.nvidia.com/compute/cuda/${CUDA_MED}/local_installers/cuda-repo-${UBUNTU_VERSION}-${CUDA_SHORT/./-}-local_${CUDA}_amd64.deb
  sudo dpkg -i cuda-repo-${UBUNTU_VERSION}-${CUDA_SHORT/./-}-local_${CUDA}_amd64.deb
  sudo apt-key add /var/cuda-repo-${UBUNTU_VERSION}-${CUDA_SHORT/./-}-local/7fa2af80.pub
  sudo apt update -qq
  sudo apt install cuda-nvcc-${CUDA_SHORT/./-} cuda-libraries-dev-${CUDA_SHORT/./-}
  sudo apt clean
  CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
  LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
  PATH=${CUDA_HOME}/bin:${PATH}
  nvcc --version
fi
