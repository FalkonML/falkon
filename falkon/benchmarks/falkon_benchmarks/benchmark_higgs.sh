#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

# Prepare GPU
export CUDA_VISIBLE_DEVICES="1"

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Common variables
DSET="higgs"
PY_LAUNCHER="benchmark_runner.py"


# Falkon Logistic
if [ true = false ]; then
	conda activate torch
	ALGO="falkon-cls"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 5 -e 0 \
		--penalty-list 1e-3 1e-6 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9  \
		--iter-list 3 3 3 8 8 8 8 8 8 --seed 13 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 5 -e 0 \
		--penalty-list 1e-3 1e-6 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9  \
		--iter-list 3 3 3 8 8 8 8 8 8 --seed 14 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

if [ true = false ]; then
	conda activate torch
	ALGO="falkon"
	M=120000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPyTorch - SGPR
if [ true = false ]; then
  ALGO="gpytorch-sgpr"
  M=50
  LR=0.001
  EPOCHS=15
  OUTFILE="logs/${DSET}_${ALGO}_${M}_learned_centers.txt"
  conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --learn-hyperparams -e $EPOCHS \
				--seed 12 2>&1 | tee -a $OUTFILE
  conda deactivate
fi


# GPytorch
if [ true = false ]; then
	ALGO="gpytorch-cls"
	M=1000
	VAR="tril_natgrad"
	LR=0.02
    NATGRAD_LR=0.02
	BATCH_SIZE=16000
	EPOCHS=15
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --natgrad-lr $NATGRAD_LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e $EPOCHS --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --natgrad-lr $NATGRAD_LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e $EPOCHS --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --natgrad-lr $NATGRAD_LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e $EPOCHS --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --natgrad-lr $NATGRAD_LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e $EPOCHS --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --natgrad-lr $NATGRAD_LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e $EPOCHS --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPFlow
if [ true = true ]; then
	ALGO="gpflow-cls"
	M=2000
	VAR="diag"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	BATCH_SIZE=16000
	EPOCHS=33000 # Around 60 epochs
	ERROR_EVERY=550
	LR=0.02
	conda activate gpflow
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 5 --batch-size $BATCH_SIZE --learn-hyperparams \
			        --lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
			        --seed 12 2>&1 | tee -a $OUTFILE
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
if [ true = false ]; then
	# NOTE: EigenPro might take forever on this dataset
	ALGO="eigenpro"
	OUTFILE="logs/${DSET}_${ALGO}.txt"
	conda activate epro2
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 5 -e 10 --seed 12 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi
