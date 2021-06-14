#!/bin/bash

set -exu

echo $PATH
which cmake
cmake --version

# Env variables that should be set
# CUDA_VERSION, PYTHON_VERSION, PYTORCH_VERSION, WHEEL_FOLDER, HOME_DIR
if [[ -d "$HOME_DIR" ]]; then
  falkon_rootdir="$HOME_DIR"
else
  echo "ERROR: Falkon root-directory not found at ${HOME_DIR}."
  exit 1
fi

if [[ ${BUILD_DOCS:-} == "TRUE" ]]; then
  do_build_docs=1
else
  do_build_docs=
fi

if [[ ${UPLOAD_CODECOV:-} == "TRUE" ]]; then
  do_codecov=1
else
  do_codecov=
fi

if [[ "${CUDA_VERSION}" == 'cpu' ]]; then
  is_cpu_only=1
else
  is_cpu_only=
fi

cd "$falkon_rootdir"

if [ -n "${is_cpu_only}" ]; then
  export USE_CUDA=0
  export CUDA_VERSION="0.0"
  export CUDNN_VERSION="0.0"
  cuda_toolkit='cpuonly'
  cuda_name="cpu"
else
  echo "Switching to CUDA version ${CUDA_VERSION}"
  # will export CUDA_VERSION, CUDNN_VERSION
  . scripts/switch_cuda_version.sh "${CUDA_VERSION}"
  cuda_toolkit="cudatoolkit=${CUDA_VERSION}"
  cuda_name="cuda${CUDA_VERSION}"
fi

# Create new conda env
conda_env="${cuda_name}-${PYTHON_VERSION}-${PYTORCH_VERSION}"
conda create --quiet --yes -n "${conda_env}" python="${PYTHON_VERSION}"
source activate "${conda_env}"

echo "$(date) || Installing dependencies..."
SETUPTOOLS_VERSION="=46.0.0"
NUMPY_VERSION="=1.19"
CYTHON_VERSION="=0.29"
MKL_VERSION="==2020.1"
# numpy (no mkl), setuptools, cython
time conda install --quiet --yes -n ${conda_env} \
                  "numpy${NUMPY_VERSION}" nomkl \
                  "setuptools${SETUPTOOLS_VERSION}" \
                  "cython${CYTHON_VERSION}"
# mkl
time conda install --quiet --yes -n ${conda_env} \
                  "mkl-include${MKL_VERSION}" "mkl-static${MKL_VERSION}" -c intel
# pytorch (via pip)
if [ -n "${is_cpu_only}" ]; then
    TORCH_CUDA_VERSION="cpu"
else
    TORCH_CUDA_VERSION="cu$(echo ${CUDA_VERSION} | tr -d '.')"
fi
TORCH_PYTHON_VERSION="$(echo $PYTHON_VERSION | tr -d '.')"
if [ "$PYTHON_VERSION" == "3.6" || "$PYTHON_VERSION" == "3.7" ]; then
    TORCH_PYTHON_VERSION_2="${TORCH_PYTHON_VERSION}m"
else
    TORCH_PYTHON_VERSION_2="$TORCH_PYTHON_VERSION"
fi
TORCH_WHEEL_NAME="${TORCH_CUDA_VERSION}/torch-${PYTORCH_VERSION}+${TORCH_CUDA_VERSION}-cp${TORCH_PYTHON_VERSION}-cp${TORCH_PYTHON_VERSION_2}-linux_x86_64.whl"
TORCH_WHEEL_ALT_NAME="${TORCH_CUDA_VERSION}/torch-${PYTORCH_VERSION}-cp${TORCH_PYTHON_VERSION}-cp${TORCH_PYTHON_VERSION_2}-linux_x86_64.whl"
time pip install --quiet --no-cache-dir "https://download.pytorch.org/whl/${TORCH_WHEEL_NAME}" || \
     pip install --quiet --no-cache-dir "https://download.pytorch.org/whl/${TORCH_WHEEL_ALT_NAME}"
# keops (via pip)
time pip install --no-cache-dir --editable ./keops
echo "$(date) || Dependencies installed."

# Install Falkon
echo "$(date) || Installing Falkon..."
install_modifiers="[test"
if [ -n "${do_build_docs}" ]; then
  install_modifiers="${install_modifiers},doc"
fi
install_modifiers="${install_modifiers}]"
export CC=g++
time pip install --quiet --editable ".${install_modifiers}"
echo "$(date) || Falkon installed."

# DEBUG
ldconfig -p | grep cusolver

# Test Falkon
echo "$(date) || Testing Falkon..."
#flake8 --count falkon
pytest --cov-report=term-missing --cov-report=xml:coverage.xml --cov=falkon --cov-config setup.cfg
if [ -n "${do_codecov}" ]; then
  echo "$(date) || Uploading test-data to codecov..."
  curl -s https://codecov.io/bash | bash -s -- -c -f coverage.xml -t $CODECOV_TOKEN
  echo "$(date) || Data uploaded."
fi
echo "$(date) || Falkon tested."

# Build wheel
echo "$(date) || Building wheel..."
dist_name="torch-${PYTORCH_VERSION}+${cuda_name}"
current_build_folder="${WHEEL_FOLDER}/${dist_name}"
mkdir -p current_build_folder
export CC=g++
time python setup.py bdist_wheel --dist-dir="${current_build_folder}"
ls -lah "${current_build_folder}"
echo "$(date) || Wheel built."

# Build documentation
if [ -n "${do_build_docs}" ]; then
    echo "$(date) || Building docs..."
    pushd "./doc"
    /sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset
    (
      set +x
      git remote set-url origin "https://Giodiro:${GIT_TOKEN}@github.com/FalkonML/falkon.git"
    )
    make clean && make html && make install
    popd
    echo "$(date) || Docs built."
fi

# Clean-up
echo "Cleaning up..."
conda deactivate
conda env remove --quiet -yn "$conda_env"
rm -rf "build/"
