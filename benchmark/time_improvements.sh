#!/bin/bash

# Prepare log file
if [ ! -d logs ]; then
    mkdir logs
fi

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Common variables
DSET="higgs"
PY_LAUNCHER="time_improvements.py"
LOG_FILE="logs/time_improvements_${DSET}.log"

conda activate torch

export CUDA_VISIBLE_DEVICES="0"
PYTHONPATH='..' python $PY_LAUNCHER --exp-num 1 --dataset $DSET 2>&1 | tee -a $LOG_FILE
export CUDA_VISIBLE_DEVICES="0"
PYTHONPATH='..' python $PY_LAUNCHER --exp-num 2 --dataset $DSET 2>&1 | tee -a $LOG_FILE
export CUDA_VISIBLE_DEVICES="0"
PYTHONPATH='..' python $PY_LAUNCHER --exp-num 3 --dataset $DSET 2>&1 | tee -a $LOG_FILE
export CUDA_VISIBLE_DEVICES="0,1"
PYTHONPATH='..' python $PY_LAUNCHER --exp-num 4 --dataset $DSET 2>&1 | tee -a $LOG_FILE
export CUDA_VISIBLE_DEVICES="0,1"
PYTHONPATH='..' python $PY_LAUNCHER --exp-num 5 --dataset $DSET 2>&1 | tee -a $LOG_FILE

conda deactivate