#!/bin/bash

set -ex

# Env variables that should be set
# CUDA_VERSION, PYTHON_VERSION, PYTORCH_VERSION
# WHEEL_FOLDER
if [[ -d '/falkon' ]]; then
  falkon_rootdir='/falkon'
else
  echo "ERROR: Falkon root-directory not found at /falkon."
  exit 1
fi

if [[ ${BUILD_DOCS:-} == "TRUE" ]]
  do_build_docs=1
else
  do_build_docs=
fi

if [[ ${UPLOAD_CODECOV:-} == "TRUE" ]]
  do_codecov=1
else
  do_codecov=
fi


if [[ "${CUDA_VERSION}" == 'cpu' ]]; then
  export USE_CUDA=0
  export CUDA_VERSION="0.0"
  export CUDNN_VERSION="0.0"
else
  echo "Switching to CUDA version ${CUDA_VERSION}"
  . ./switch_cuda_version.sh "${CUDA_VERSION}"
fi

cd "$falkon_rootdir"

conda_env="${CUDA_VERSION}-${PYTHON_VERSION}-${PYTORCH_VERSION}"
conda create --yes -n "${conda_env}" python="${PY_VERSION}"
source activate "${conda_env}"

# Install Prerequisites
echo "Installing PyTorch version ${PYTORCH_VERSION}..."
conda install -q pytorch=${TORCH_VERSION} -c pytorch --yes -n ${conda_env}

echo "Installing KeOps..."
pip install --no-cache-dir --editable ./keops

# Install Falkon
echo "$(date) || Installing Falkon..."
install_modifiers="[test"
if [ -n "${do_build_docs}" ]; then
  install_modifiers="${install_modifiers},doc"
fi
install_modifiers="${install_modifiers}]"
time pip install --editable ".${install_modifiers}"
echo "$(date) || Falkon installed."

# Test Falkon
echo "$(date) || Testing Falkon..."
pytest 'falkon/tests/test_kernels.py::TestLaplacianKernel::test_mmv[No KeOps-gpu]'
pytest --cov-report=term-missing --cov-report=xml:coverage.xml --cov=falkon --cov-config setup.cfg
flake8 --count falkon
if [ -n "${do_codecov}" ]; then
  echo "$(date) || Uploading test-data to codecov..."
  curl -s https://codecov.io/bash | bash -s -- -c -f coverage.xml -t $CODECOV_TOKEN
  echo "$(date) || Data uploaded."
fi
echo "$(date) || Falkon tested."

# Build wheel
echo "$(date) || Building wheel..."
dist_name="torch-${TORCH_VERSION}+${CUDA_VERSION}"
current_build_folder="${WHEEL_FOLDER}/${dist_name}"
mkdir -p current_build_folder
python setup.py bdist_wheel --dist-dir="${current_build_folder}"
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
}

# Clean-up
echo "Cleaning up..."
source deactivate
conda env remove -yn "$conda_env"
rm -rf "build/"
