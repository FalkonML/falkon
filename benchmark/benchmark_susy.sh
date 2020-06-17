#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

# Prepare GPU
export CUDA_VISIBLE_DEVICES="0"

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Common variables
DSET="susy"
TRAIN_DATAPOINTS=5000000
PY_LAUNCHER="benchmark_runner.py"


# Falkon Logistic
if [ true = false ]; then
	conda activate torch
	ALGO="falkon-cls"
	M=20000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 3 -e 0 \
		--penalty-list 1e-4 1e-7 1e-8 1e-8 1e-8 1e-8 1e-8 1e-8 \
		--iter-list 5 5 5 8 8 8 8 8 --seed 12 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 3 -e 0 \
		--penalty-list 1e-4 1e-6 1e-8 1e-8 1e-8 1e-8 1e-8 1e-8 \
		--iter-list 5 5 5 8 8 8 8 8 --seed 13 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 3 -e 0 \
		--penalty-list 1e-4 1e-6 1e-8 1e-8 1e-8 1e-8 1e-8 1e-8 \
		--iter-list 5 5 5 8 8 8 8 8 --seed 14 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 3 -e 0 \
		--penalty-list 1e-4 1e-6 1e-8 1e-8 1e-8 1e-8 1e-8 1e-8 \
		--iter-list 5 5 5 8 8 8 8 8 --seed 15 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 3 -e 0 \
		--penalty-list 1e-4 1e-6 1e-8 1e-8 1e-8 1e-8 1e-8 1e-8 \
		--iter-list 5 5 5 8 8 8 8 8 --seed 16 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

if [ true = false ]; then
	conda activate torch
	ALGO="falkon"
	M=30000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 20 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPytorch
if [ true = true ]; then
	ALGO="gpytorch-cls"
	M=1000
	VAR="diag"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	BATCH_SIZE=8000
	LR=0.002
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e 10 --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e 10 --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e 10 --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e 10 --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--lr $LR --batch-size $BATCH_SIZE --learn-hyperparams \
				--var-dist $VAR --sigma 5 -e 10 --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPFlow
if [ true = false ]; then
	ALGO="gpflow-cls"
	M=2000
	VAR="diag"
	BATCH_SIZE=16000
	LR=0.003
	ERROR_EVERY=$(( $TRAIN_DATAPOINTS / $BATCH_SIZE ))
	EPOCHS=$(( $TRAIN_DATAPOINTS / $BATCH_SIZE * 20 ))
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate gpflow
	echo "Running ${ALGO} on ${DSET} data for ${EPOCHS} epochs, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 4 --batch-size $BATCH_SIZE --learn-hyperparams \
			        --lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
			        --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 4 --batch-size $BATCH_SIZE --learn-hyperparams \
			        --lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
			        --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 4 --batch-size $BATCH_SIZE --learn-hyperparams \
			        --lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
			        --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 4 --batch-size $BATCH_SIZE --learn-hyperparams \
			        --lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
			        --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M \
				--var-dist ${VAR} --sigma 4 --batch-size $BATCH_SIZE --learn-hyperparams \
			        --lr $LR --natgrad-lr 0.000 --epochs $EPOCHS --error-every $ERROR_EVERY \
			        --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi

# EigenPro
if [ true = false ]; then
	ALGO="eigenpro"
	OUTFILE="logs/${DSET}_${ALGO}.txt"
	conda activate epro2
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 4 -e 2 \
			        --data-subsample 600000 --seed 12 --epro-q 850 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 4 -e 2 \
			        --data-subsample 600000 --seed 13 --epro-q 850 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 4 -e 2 \
			        --data-subsample 600000 --seed 14 --epro-q 850 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 4 -e 2 \
			        --data-subsample 600000 --seed 15 --epro-q 850 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 4 -e 2 \
			        --data-subsample 600000 --seed 16 --epro-q 850 2>&1 | tee -a $OUTFILE
	conda deactivate
	echo "${ALGO} on ${DSET} data complete..."
fi
