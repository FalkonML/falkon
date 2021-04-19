#!/bin/bash

conda update --yes conda
conda create --yes -n test python="${PYTHON_VERSION}"
source activate test
