#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

echo "Running with 1 GPU"
export CUDA_VISIBLE_DEVICES="0"
python mmv_timings.py --num-gpus 1 2>&1 | tee -a "logs/mmv_timings_1GPU.txt"
exit 1;

echo "Running with 2 GPUs"
export CUDA_VISIBLE_DEVICES="0,1"
python mmv_timings.py --num-gpus 2 2>&1 | tee -a "logs/mmv_timings_2GPU.txt"

