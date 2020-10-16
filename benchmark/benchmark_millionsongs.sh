#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

export CUDA_VISIBLE_DEVICES="1"

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

DSET="millionsongs"
PY_LAUNCHER="benchmark_runner.py"

# Falkon 64 / 32
if [ true = false ]; then
	ALGO="falkon"
	M=50000
	TYPE="float64"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 6 \
    --penalty 1e-6 --seed 12 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

if [ true = false ]; then
	ALGO="falkon"
	M=50000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 7 \
	  --penalty 2e-6 --seed 12 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 7 \
	  --penalty 2e-6 --seed 13 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 7 \
	  --penalty 2e-6 --seed 14 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 7 \
	  --penalty 2e-6 --seed 15 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 7 \
	  --penalty 2e-6 --seed 16 --kernel gaussian 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPytorch
if [ true = true ]; then
	ALGO="gpytorch-reg"
	M=3000
	VAR="full"
	BATCH_SIZE=12000
	LR=0.01
    NATGRAD_LR=0.00
	EPOCHS=20
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 7 -e $EPOCHS --seed 12 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 7 -e $EPOCHS --seed 13 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 7 -e $EPOCHS --seed 14 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 7 -e $EPOCHS --seed 15 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 7 -e $EPOCHS --seed 16 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPFlow
if [ false = true ]; then
	ALGO="gpflow-reg"
	M=3000
	VAR="diag"
	BATCH_SIZE=16000
	EPOCHS=3000
	ERROR_EVERY=32
	LR=0.008
    TYPE="float64"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate gpflow
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 5 -t $TYPE --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 12 2>&1 | tee -a $OUTFILE
    exit 1
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 5 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 5 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 5 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 5 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi

# EigenPro
if [ false = true ]; then
	ALGO="eigenpro"
	OUTFILE="logs/${DSET}_${ALGO}.txt"
	conda activate epro2
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 6 -e 10 --n-subsample 12000 \
	      --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 6 -e 10 --n-subsample 12000 \
	      --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 6 -e 10 --n-subsample 12000 \
	      --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 6 -e 10 --n-subsample 12000 \
	      --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 6 -e 10 --n-subsample 12000 \
	      --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
fi
