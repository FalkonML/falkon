## benchmark_hgrad.py

First hypergradient runner function, supports several models:
 - nkrr (probably not working anymore)
 - gpflow (3 possibilities: SGPR, GPR, SVGP)
   
    ```bash
    # Useful additional options: --optimize-centers 
    python benchmark_hgrad.py --seed 1234 --lr 0.1 --steps 100 --sigma-type single --sigma-init 1.0 --penalty-init 1.0 --gpflow --M 20 --dataset ho-higgs --name boston_test_sgpr
    ```

## simple_hopt.py

Latest hyper-parameter tuning runner. Supports 2 different experiment types, both with several models.

The supported experiments are:
 1. Hyperparameter optimization via gradient descent
 2. Hyperparameter grid-search

The available models are:
 1. LOOCV (`--model loocv`)
 2. GCV (`--model gcv`)
 3. SGPR (`--model sgpr`)
 4. GPR (`--model gpr`)
 5. HyperGrad (closed-form) (`--model hgrad-closed`)
 6. HyperGrad (via implicit function theorem) (`--model hgrad-ift`)
 7. Complexity reg with penalized data-fit term (`--model creg-penfit`)
 8. Complexity reg with plain data-fit term (`--model creg-nopenfit`)

An example for running SGPR in grid-search mode:
```bash
GS=grid_specs/tuned_boston.csv
python gen_grid_spec.py --out-file "$GS"
ENAME=test_tuned_m20
M=20
PYTHONPATH=.. python simple_hopt.py \
    --seed 12319 \
    --dataset boston \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
    --sigma-type single \
    --sigma-init 1.0 \
    --penalty-init 1e-4 \
    --num-centers $M \
    --dataset boston \
    --model sgpr \
    --grid-spec "$GS" \
    --name "boston_gs_sgpr_${ENAME}"
```

Or for running it in optimization mode:
```bash
ENAME=test_hopt_m20
M=20
MODEL=sgpr

PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --dataset boston \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
    --sigma-type single \
    --sigma-init 1 \
    --penalty-init 1e-4 \
    --lr 0.05 \
    --epochs 100 \
    --op \
    --os \
    --num-centers $M \
    --dataset protein \
    --model sgpr \
    --name "protein_hopt_sgpr_${ENAME}"
 PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --dataset boston \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
    --sigma-type single \
    --sigma-init 15 \
    --penalty-init 1e-4 \
    --lr 0.05 \
    --epochs 100 \
    --op \
    --os \
    --num-centers $M \
    --dataset protein \
    --model sgpr \
    --name "protein_hopt_sgpr_${ENAME}"
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --dataset boston \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
    --sigma-type single \
    --sigma-init 1 \
    --penalty-init 1 \
    --lr 0.05 \
    --epochs 100 \
    --op \
    --os \
    --num-centers $M \
    --dataset protein \
    --model sgpr \
    --name "protein_hopt_sgpr_${ENAME}"
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --dataset boston \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
    --sigma-type single \
    --sigma-init 15 \
    --penalty-init 1 \
    --lr 0.05 \
    --epochs 100 \
    --op \
    --os \
    --num-centers $M \
    --dataset protein \
    --model sgpr \
    --name "protein_hopt_${MODEL}_${ENAME}"
```


```
function run_exp () {
   
    PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
        --seed 12319 \
        --cg-tol 1e-3 \
        --val-pct 0.2 \
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


M=20
DATASET=protein
LR=0.05
NUM_EPOCHS=200
PEN_INIT=1
SIG_INIT=1
ENAME="test_hopt_m${M}_lr${LR}_pinit${PEN_INIT}sinit${SIG_INIT}_meanrem"

MODEL=loocv
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
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
MODEL=sgpr
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
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
MODEL=gcv
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
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
MODEL=hgrad-ift
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
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
MODEL=hgrad-closed
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
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
MODEL=creg-nopenfit
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
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
MODEL=creg-penfit
PYTHONPATH=.. python hgrad_benchmarks/simple_hopt.py \
    --seed 12319 \
    --cg-tol 1e-3 \
    --val-pct 0.2 \
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
