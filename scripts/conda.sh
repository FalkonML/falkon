#!/bin/bash

conda create --yes -n "${CONDA_ENV}" python="${PYTHON_VERSION}"
source activate test
