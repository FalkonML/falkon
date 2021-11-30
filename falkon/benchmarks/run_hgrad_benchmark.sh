
function run_exp () {
   local SIG_INIT=$1
   local PEN_INIT=$2
   local LR=$3
   local M=$4
   local DATASET=$5
   local ENAME=$6
   local VAL_PCT=$7
   local MODEL=$8
   PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
        --seed 12319 \
        --cg-tol 1e-3 \
        --val-pct $VAL_PCT \
        --sigma-type single \
        --sigma-init $SIG_INIT \
        --penalty-init $PEN_INIT \
        --lr $LR \
        --epochs $NUM_EPOCHS \
        --op \
        --os \
        --num-centers $M \
        --dataset $DATASET \
        --model $MODEL \
        --cuda \
        --name "${DATASET}_hopt_${MODEL}_${ENAME}"
}

function run_exp_all_models () {
     run_exp "$1" "$2" "$3" "$4" "$5" "$6" "$7" "loocv"
     run_exp "$1" "$2" "$3" "$4" "$5" "$6" "$7" "sgpr"
     run_exp "$1" "$2" "$3" "$4" "$5" "$6" "$7" "gcv"
#     run_exp "$1" "$2" "$3" "$4" "$5" "$6" "$7" "hgrad-ift"
     run_exp "$1" "$2" "$3" "$4" "$5" "$6" "$7" "hgrad-closed"
    run_exp "$1" "$2" "$3" "$4" "$5" "$6" "$7" "creg-nopenfit"
     run_exp "$1" "$2" "$3" "$4" "$5" "$6" "$7" "creg-penfit"
}


M=20
DATASET=boston
LR=0.02
NUM_EPOCHS=200
VAL_PCT=0.2

PEN_INIT=1e-4
SIG_INIT=15
ENAME="test_hopt_rmsprop_m${M}_lr${LR}_pinit${PEN_INIT}sinit${SIG_INIT}_meanrem_val${VAL_PCT}"
run_exp_all_models "$SIG_INIT" "$PEN_INIT" "$LR" "$M" "$DATASET" "$ENAME" "$VAL_PCT"

PEN_INIT=1
SIG_INIT=15
ENAME="test_hopt_rmsprop_m${M}_lr${LR}_pinit${PEN_INIT}sinit${SIG_INIT}_meanrem_val${VAL_PCT}"
run_exp_all_models "$SIG_INIT" "$PEN_INIT" "$LR" "$M" "$DATASET" "$ENAME" "$VAL_PCT"


#  VAL_PCT=0.2
#  ENAME="test_hopt_m${M}_lr${LR}_pinit${PEN_INIT}sinit${SIG_INIT}_meanrem_val${VAL_PCT}"
#  run_exp_all_models "$SIG_INIT" "$PEN_INIT" "$LR" "$M" "$DATASET" "$ENAME" "$VAL_PCT"
#
#  VAL_PCT=0.4
#  ENAME="test_hopt_m${M}_lr${LR}_pinit${PEN_INIT}sinit${SIG_INIT}_meanrem_val${VAL_PCT}"
#  run_exp_all_models "$SIG_INIT" "$PEN_INIT" "$LR" "$M" "$DATASET" "$ENAME" "$VAL_PCT"
#
#  VAL_PCT=0.6
#  ENAME="test_hopt_m${M}_lr${LR}_pinit${PEN_INIT}sinit${SIG_INIT}_meanrem_val${VAL_PCT}"
#  run_exp_all_models "$SIG_INIT" "$PEN_INIT" "$LR" "$M" "$DATASET" "$ENAME" "$VAL_PCT"
#
#  VAL_PCT=0.8
#  ENAME="test_hopt_m${M}_lr${LR}_pinit${PEN_INIT}sinit${SIG_INIT}_meanrem_val${VAL_PCT}"
#  run_exp_all_models "$SIG_INIT" "$PEN_INIT" "$LR" "$M" "$DATASET" "$ENAME" "$VAL_PCT"
