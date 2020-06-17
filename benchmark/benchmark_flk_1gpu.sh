#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

export CUDA_VISIBLE_DEVICES="0"
PY_LAUNCHER="benchmark_runner.py"

# Prepare conda
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# FLIGHTS-CLS
DSET="flights-cls"
if [ false = true ]; then
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
if [ true = false ]; then
	ALGO="falkon"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	#PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 0.9 \
	#  --penalty 1e-8 --seed 12 --kernel gaussian 2>&1 | tee -a $OUTFILE
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

# FLIGHTS
DSET="flights"
if [ false = true ]; then
	ALGO="falkon"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 20 -d $DSET -t $TYPE --sigma 0.9 \
	  --penalty 1e-8 --seed 12 --kernel gaussian 2>&1 | tee -a $OUTFILE
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

# HIGGS
DSET="higgs"
if [ true = true ]; then
	conda activate torch
	ALGO="falkon-cls"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET --sigma 5 -e 0 \
		--penalty-list 1e-3 1e-6 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9 1e-9  \
		--iter-list 3 3 3 8 8 8 8 8 8 --seed 12 \
		-M $M -t $TYPE 2>&1 | tee -a $OUTFILE
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
if [ false = true ]; then
	conda activate torch
	ALGO="falkon"
	M=120000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 12 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 12 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 12 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 12 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 12 --sigma 3.8 --penalty 1e-7 \
					    -M $M -t $TYPE --kernel gaussian --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# MILLIONSONGS
DSET="millionsongs"
if [ false = true ]; then
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

# SUSY
DSET="susy"
if [ false = true ]; then
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
if [ false = true ]; then
	conda activate torch
	ALGO="falkon"
	M=30000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 10 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 12 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 10 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 13 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 10 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 14 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 10 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 15 2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -e 10 --sigma 3.0 --penalty 1e-6 \
					    -M $M -t $TYPE --kernel gaussian --seed 16 2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# TAXI
DSET="taxi"
if [ false = true ]; then
    ALGO="falkon"
    M=100000
    TYPE="float32"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 10 -d $DSET -t $TYPE \
	    --sigma 0.9 --penalty 2e-7 --kernel gaussian --seed 12 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 7 -d $DSET -t $TYPE \
	    --sigma 0.9 --penalty 2e-7 --kernel gaussian --seed 13 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 7 -d $DSET -t $TYPE \
	    --sigma 0.9 --penalty 2e-7 --kernel gaussian --seed 14 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 7 -d $DSET -t $TYPE \
	    --sigma 0.9 --penalty 2e-7 --kernel gaussian --seed 15 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 7 -d $DSET -t $TYPE \
	    --sigma 0.9 --penalty 2e-7 --kernel gaussian --seed 16 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

# TIMIT
DSET="timit"
if [ false = true ]; then
	ALGO="falkon"
	M=100000
	TYPE="float32"
	OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
	conda activate torch
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 10 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 12 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 10 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 13 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 10 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 14 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 10 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 15 \
					    2>&1 | tee -a $OUTFILE
	PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -d $DSET -M $M -t $TYPE -e 10 \
					    --sigma 14.5 --penalty 5e-9 --kernel gaussian --seed 16 \
					    2>&1 | tee -a $OUTFILE
	conda deactivate
fi

# YELP
DSET="yelp"
if [ false = true ]; then
    ALGO="falkon"
    M=50000
    TYPE="float32"
    OUTFILE="logs/${DSET}_${ALGO}_${M}_${TYPE}.txt"
    conda activate torch
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 15 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 12 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 15 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 13 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 15 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 14 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 15 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 15 2>&1 | tee -a $OUTFILE
    PYTHONPATH='..' python $PY_LAUNCHER -a $ALGO -M $M -e 15 -d $DSET \
                                        -t $TYPE --sigma 20.0 --kernel gaussian \
                                        --penalty 1e-6 --seed 16 2>&1 | tee -a $OUTFILE
    conda deactivate
fi

