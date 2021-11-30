#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

export CUDA_VISIBLE_DEVICES="1"

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

DSET="flights"
TRAIN_DATAPOINTS=5930000
PY_LAUNCHER="benchmark_runner.py"

# Falkon 32
if [ false = true ]; then
	ALGO="falkon"
	M=75000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 12 --kernel gaussian 2>&1 | tee -a $OUTFILE
	exit 1;
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 13 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 14 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 15 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 16 --kernel gaussian 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPytorch
if [ true = false ]; then
	ALGO="gpytorch-reg"
	M=2000
	VAR="natgrad"
	BATCH_SIZE=16000
	LR=0.005
    NATGRAD_LR=0.005
	EPOCHS=30
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 0.9 -e $EPOCHS --seed 12 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 0.9 -e $EPOCHS --seed 13 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 0.9 -e $EPOCHS --seed 14 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 0.9 -e $EPOCHS --seed 15 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --natgrad-lr $NATGRAD_LR --sigma 0.9 -e $EPOCHS --seed 16 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPFlow
if [ true = true ]; then
	ALGO="gpflow-reg"
	M=2000
	VAR="diag"
	BATCH_SIZE=16000
	LR=0.005
	EPOCHS=$(( $TRAIN_DATAPOINTS / $BATCH_SIZE * 25 ))
	ERROR_EVERY=$(( $TRAIN_DATAPOINTS / $BATCH_SIZE ))
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate gpflow
	echo "Running ${ALGO} on ${DSET} data for ${EPOCHS} iterations, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 1 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 1 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 1 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 1 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 1 --batch-size $BATCH_SIZE --learn-hyperparams \
				--lr $LR --natgrad-lr 0.0000 --epochs $EPOCHS --error-every $ERROR_EVERY \
				--seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi

# EigenPro
if [ true = false ]; then
	ALGO="eigenpro"
	OUTFILE="logs/${DSET}_${ALGO}.txt"
	conda activate epro2
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 1 -e 10 --n-subsample 12000 \
	      --data-subsample 1000000 --seed 12 --eta-divisor 10 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 1 -e 10 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 13 --eta-divisor 10 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 1 -e 10 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 14 --eta-divisor 10 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 1 -e 10 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 15 --eta-divisor 10 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 1 -e 10 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 16 --eta-divisor 10 2>&1 | tee -a $OUTFILE
	conda deactivate
fi
