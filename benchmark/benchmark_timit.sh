#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

export CUDA_VISIBLE_DEVICES=0

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

DSET="timit"
TRAIN_DATAPOINTS=1200000
PY_LAUNCHER="benchmark_runner.py"

##### TIMIT Dataset
# Falkon 64
if [ false = true ]; then
	ALGO="falkon"
	M=100000
	TYPE="float64"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 20 \
					    --sigma 15 --penalty 1e-9 --kernel gaussian \
					    2>&1 | tee -a $OUTFILE
	conda deactivate
fi

if [ false = true ]; then
	# Falkon 32
	ALGO="falkon"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 20 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 12 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 20 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 13 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 20 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 14 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 20 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 15 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 20 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 16 \
					    2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPytorch -- This fails, programmer error most likely
if [ false = true ]; then
	ALGO="gpytorch-cls"
	M=100
	VAR="diag"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --var-dist $VAR \
		--sigma 15 --lr 0.01 --batch-size 2048 -e 30 --seed 12 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPFlow
if [ true = true ]; then
	ALGO="gpflow-cls"
	M=2000
	VAR="diag"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	BATCH_SIZE=2048
	EPOCHS=$(( $TRAIN_DATAPOINTS / $BATCH_SIZE * 13 ))
	ERROR_EVERY=500
	LR=0.01
	conda activate gpflow
	echo "Running ${ALGO} on ${DSET} data for ${EPOCHS} iterations, log will be saved in ${OUTFILE}"
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 15 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS \
				--error-every $ERROR_EVERY --seed 12 2>&1 | tee -a $OUTFILE
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 15 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS \
				--error-every $ERROR_EVERY --seed 13 2>&1 | tee -a $OUTFILE
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 15 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS \
				--error-every $ERROR_EVERY --seed 14 2>&1 | tee -a $OUTFILE
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 15 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS \
				--error-every $ERROR_EVERY --seed 15 2>&1 | tee -a $OUTFILE
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 15 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS \
				--error-every $ERROR_EVERY --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi


# EigenPro
if [ true = false ]; then
	ALGO="eigenpro"
	OUTFILE="logs/${DSET}_${ALGO}.txt"
	conda activate epro2
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 5 --sigma 14.5 --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 5 --sigma 14.5 --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 5 --sigma 14.5 --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 5 --sigma 14.5 --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 5 --sigma 14.5 --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi
