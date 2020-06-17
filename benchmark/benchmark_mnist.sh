#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

# Prepare GPU
export CUDA_VISIBLE_DEVICES="0,1"

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Common variables
DSET="mnist"
PY_LAUNCHER="benchmark_runner.py"

# Falkon (32)
if [ true = true ]; then
	ALGO="falkon"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 4.4 --penalty 1e-8 \
					    -M $M -t $TYPE --kernel gaussian --seed 12 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPytorch
if [ true = false ]; then
	ALGO="gpytorch-cls"
	M=1000
	VAR="diag"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr 0.001 --batch-size 4096 --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e 15 --seed 12 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPFlow
if [ true = false ]; then
	ALGO="gpflow-cls"
	M=500
	VAR="full"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate gpflow
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 5 --batch-size 4096 --learn-hyperparams \
			        --lr 0.005 --natgrad-lr 0.0001 --epochs 10000 \
			        --seed 12 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi

# EigenPro
if [ true = true ]; then
	ALGO="eigenpro"
	OUTFILE="logs/${DSET}_${ALGO}.txt"
	conda activate epro2
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 5 -e 5 --seed 12 --data-subsample 1000000 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi
