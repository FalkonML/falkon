#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

DSET="yelp"
PY_LAUNCHER="benchmark_runner.py"
export CUDA_VISIBLE_DEVICES="0,1"

# Falkon 64
if [ true = false ]; then
    ALGO="falkon"
    M=50000
    TYPE="float64"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET \
                                        -t $TYPE --sigma 31.4 --kernel linear \
                                        --penalty 1e-7 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

# Falkon 32
if [ true = true ]; then
    ALGO="falkon"
    M=50000
    TYPE="float32"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 12 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 13 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 14 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 15 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 16 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

# GPytorch
if [ true = false ]; then
    ALGO="gpytorch-reg"
    M=1000
    VAR="diag"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size 4096 \
          --var-dist $VAR --lr 0.01 --sigma 6 -e 100 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

# GPFlow
if [ true = false ]; then
        ALGO="gpflow-reg"
        M=100
        VAR="diag"
        OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
        conda activate gpflow
        echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
                                --var-dist ${VAR} --sigma 5 --batch-size 1024 --learn-hyperparams \
				--lr 0.005 --natgrad-lr 0.000 --epochs 2000 --error-every 10 \
				--seed 12 2>&1 | tee -a $OUTFILE
        conda deactivate
        echo "${ALGO} on ${DSET} data complete..."
fi

