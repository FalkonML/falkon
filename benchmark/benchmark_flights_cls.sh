#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

export CUDA_VISIBLE_DEVICES="0,1"

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

DSET="flights-cls"
PY_LAUNCHER="benchmark_runner.py"

if [ true = false ]; then
        conda activate torch
        ALGO="falkon-cls"
        M=100000
        TYPE="float32"
        OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 0 \
                --penalty-list 1e-3 1e-6 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 \
                --iter-list 3 3 3 8 8 8 8 8 8 \
                -M $M -t $TYPE --seed 12 2>&1 | tee -a $OUTFILE
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 0 \
                --penalty-list 1e-3 1e-6 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 \
                --iter-list 3 3 3 8 8 8 8 8 8 \
                -M $M -t $TYPE --seed 13 2>&1 | tee -a $OUTFILE
        PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 0 \
                --penalty-list 1e-3 1e-6 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 \
                --iter-list 3 3 3 8 8 8 8 8 8 \
                -M $M -t $TYPE --seed 14 2>&1 | tee -a $OUTFILE
        conda deactivate
fi

# Falkon 32
if [ true = true ]; then
	ALGO="falkon"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 12 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 13 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 14 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 15 --kernel gaussian 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 16 --kernel gaussian 2>&1 | tee -a $OUTFILE
	conda deactivate
fi


# GPytorch
if [ false = true ]; then
	ALGO="gpytorch-cls"
	M=1000
	VAR="diag"
	BATCH_SIZE=8000
	EPOCHS=15
	LR=0.005
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate torch
	#PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
        #  --var-dist $VAR --lr $LR --sigma 0.9 -e $EPOCHS --seed 12 \
        #  --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --sigma 0.9 -e $EPOCHS --seed 13 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --sigma 0.9 -e $EPOCHS --seed 14 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --sigma 0.9 -e $EPOCHS --seed 15 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M --batch-size $BATCH_SIZE \
          --var-dist $VAR --lr $LR --sigma 0.9 -e $EPOCHS --seed 16 \
          --learn-hyperparams 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# GPFlow
if [ true = false ]; then
	ALGO="gpflow-cls"
	M=2000
	VAR="diag"
	BATCH_SIZE=16000
	EPOCHS=3700
	ERROR_EVERY=370
	LR=0.005
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${VAR}.txt"
	conda activate gpflow
	echo "Running ${ALGO} on ${DSET} data, log will be saved in ${OUTFILE}"
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
if [ false = true ]; then
	ALGO="eigenpro"
	OUTFILE="logs/${DSET}_${ALGO}.txt"
	ETA_DIVISOR=12
	conda activate epro2
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 12 --n-subsample 12000 \
	      --data-subsample 1000000 --seed 12 --eta-divisor $ETA_DIVISOR 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 12 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 13 --eta-divisor $ETA_DIVISOR 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 12 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 14 --eta-divisor $ETA_DIVISOR 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 12 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 15 --eta-divisor $ETA_DIVISOR 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 0.9 -e 12 --n-subsample 12000 \
	       --data-subsample 1000000 --seed 16 --eta-divisor $ETA_DIVISOR 2>&1 | tee -a $OUTFILE
	conda deactivate
fi
