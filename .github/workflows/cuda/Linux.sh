#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

if [ "${1}" = "cu124" ] || [ "${1}" = "cu126" ] || [ "${1}" = "cu128" ]; then
    OS=ubuntu2204
else
    OS=ubuntu2004
fi

case ${1} in
  cu128)
    CUDA=12.8
    APT_KEY=${OS}-${CUDA/./-}-local
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    FILENAME="cuda-repo-${APT_KEY}_${CUDA}.0-570.86.10-1_amd64.deb"
    PACKAGES="cuda-nvcc-12-8 cuda-libraries-dev-12-8"
    ;;
  cu126)
    CUDA=12.6
    APT_KEY=${OS}-${CUDA/./-}-local
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    FILENAME="cuda-repo-${APT_KEY}_${CUDA}.0-560.28.03-1_amd64.deb"
    PACKAGES="cuda-nvcc-12-6 cuda-libraries-dev-12-6"
    ;;
  cu124)
    CUDA=12.4
    APT_KEY=${OS}-${CUDA/./-}-local
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.1/local_installers
    FILENAME="cuda-repo-${APT_KEY}_${CUDA}.1-550.54.15-1_amd64.deb"
    PACKAGES="cuda-nvcc-12-4 cuda-libraries-dev-12-4"
    ;;
  cu121)
    CUDA=12.1
    APT_KEY=${OS}-${CUDA/./-}-local
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.1/local_installers
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.1-530.30.02-1_amd64.deb
    ;;
  cu118)
    CUDA=11.8
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.0-520.61.05-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    ;;
  cu117)
    CUDA=11.7
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.1-515.65.01-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.1/local_installers
    ;;
  cu116)
    CUDA=11.6
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.2-510.47.03-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.2/local_installers
    ;;
  cu115)
    CUDA=11.5
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.2-495.29.05-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.2/local_installers
    ;;
  cu113)
    CUDA=11.3
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.0-465.19.01-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    ;;
  cu102)
    CUDA=10.2
    APT_KEY=${CUDA/./-}-local-${CUDA}.89-440.33.01
    FILENAME=cuda-repo-${OS}-${APT_KEY}_1.0-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}/Prod/local_installers
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv ${URL}/${FILENAME}
sudo dpkg -i ${FILENAME}

if [ "${1}" = "cu117" ] || [ "${1}" = "cu118" ] || [ "${1}" = "cu121" ] || [ "${1}" = "cu124" ] || [ "${1}" = "cu126" ] || [ "${1}" = "cu128" ]; then
  sudo cp /var/cuda-repo-${APT_KEY}/cuda-*-keyring.gpg /usr/share/keyrings/
else
  sudo apt-key add /var/cuda-repo-${APT_KEY}/7fa2af80.pub
fi

sudo apt-get -qq update
sudo apt -y install cuda
sudo apt clean

rm -f ${FILENAME}
