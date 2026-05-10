#!/bin/bash
#SBATCH --account=IscrB_ProAmmo
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=1         # tasks out of 32
#SBATCH --gres=gpu:1                # gpus per node out of 4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --qos=boost_qos_dbg
#SBATCH -p boost_usr_prod


if [ ! -d logs ]; then
    mkdir logs
fi

activate_falkon

# Common variables
PY_LAUNCHER="benchmark_runner.py"
HIGGS_PATH=".."
DSET="higgs"
M=120000
TYPE="float32"

KEOPS_OUTFILE="logs/${DSET}_${M}_${TYPE}_KEOPS.txt"
NOKEOPS_OUTFILE="logs/${DSET}_${M}_${TYPE}_NOKEOPS.txt"

PYTHONPATH='..' python $PY_LAUNCHER -d $DSET --data-path $HIGGS_PATH -e 20 --sigma 3.8 --penalty 1e-7 \
                    -M $M -t $TYPE --kernel gaussian --seed 12 2>&1 | tee -a $KEOPS_OUTFILE
PYTHONPATH='..' python $PY_LAUNCHER -d $DSET --data-path $HIGGS_PATH -e 20 --sigma 3.8 --penalty 1e-7 \
                    -M $M -t $TYPE --kernel gaussian --seed 12 --use-keops 2>&1 | tee -a $NOKEOPS_OUTFILE