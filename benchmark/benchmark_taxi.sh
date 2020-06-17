#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

DSET="taxi"
PY_LAUNCHER="benchmark_runner.py"
TRAINING_POINTS=1000000000

export CUDA_VISIBLE_DEVICES="0,1"

# Falkon 64
if [ true = false ]; then
    ALGO="falkon"
    M=80000
    TYPE="float64"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 1 --penalty 1e-7 --kernel laplacian 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

# Falkon 32
if [ true = true ]; then
    ALGO="falkon"
    M=100000
    TYPE="float32"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE \
	    --sigma 1.0 --penalty 2e-7 --kernel gaussian --seed 12 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE \
	    --sigma 1.0 --penalty 2e-7 --kernel gaussian --seed 13 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE \
	    --sigma 1.0 --penalty 2e-7 --kernel gaussian --seed 14 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE \
	    --sigma 1.0 --penalty 2e-7 --kernel gaussian --seed 15 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE \
	    --sigma 1.0 --penalty 2e-7 --kernel gaussian --seed 16 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

# GPytorch
if [ false = true ]; then
    ALGO="gpytorch-reg"
    M=500
    VAR="diag"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
    BATCH_SIZE=16000
    LR=0.001
    EPOCHS=5
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
	    --batch-size $BATCH_SIZE --var-dist $VAR --lr $LR --sigma 1 \
	    --epochs $EPOCHS --learn-hyperparams --seed 12 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
	    --batch-size $BATCH_SIZE --var-dist $VAR --lr $LR --sigma 1 \
	    --epochs $EPOCHS --learn-hyperparams --seed 13 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
	    --batch-size $BATCH_SIZE --var-dist $VAR --lr $LR --sigma 1 \
	    --epochs $EPOCHS --learn-hyperparams --seed 14 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

# GPFlow
if [ false = true ]; then
    ALGO="gpflow-reg"
    M=1000
    VAR=diag
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
    BATCH_SIZE=32000
    EPOCHS=$(( $TRAINING_POINTS / $BATCH_SIZE * 10 ))
    ERROR_EVERY=30000  # This is one epoch
    conda activate gpflow
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
      --var-dist ${VAR} --sigma 1 --batch-size $BATCH_SIZE \
      --lr 0.003 --natgrad-lr 0.0000 --epochs $EPOCHS --error-every $ERROR_EVERY \
      --learn-hyperparams --seed 16 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

