#!/bin/bash
if [ ! -d logs ]; then
    mkdir logs
fi

echo "Running with 1 GPU"
export CUDA_VISIBLE_DEVICES="0"
python lauum_timings.py 2>&1 | tee -a "logs/lauum_timings_1GPU.txt"

echo "Running with 2 GPUs"
export CUDA_VISIBLE_DEVICES="0,1"
python lauum_timings.py 2>&1 | tee -a "logs/lauum_timings_2GPU.txt"
exit 1;


echo "Running with 3 GPUs"
export CUDA_VISIBLE_DEVICES="0,1,2"
python lauum_timings.py 2>&1 | tee -a "logs/lauum_timings_3GPU.txt"

echo "Running with 4 GPUs"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python lauum_timings.py 2>&1 | tee -a "logs/lauum_timings_4GPU.txt"

echo "Running with 5 GPUs"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
python lauum_timings.py 2>&1 | tee -a "logs/lauum_timings_5GPU.txt"
