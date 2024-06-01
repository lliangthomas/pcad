#!/bin/bash

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 2-0:00
#SBATCH -o /data/tliang/log/job-%x-%A.out
#SBATCH -e /data/tliang/log/job-%x-%A.err

echo "Slurm Visible Cuda Devices:"
echo $CUDA_VISIBLE_DEVICES
gpus="--gpus \"device=${CUDA_VISIBLE_DEVICES}\""
echo $gpus

cd /data/tliang/tmdt-benchmark/anomalib
# docker build . -t test-anomalib

# Train NeRF
docker run $gpus --shm-size 96G \
	-v /data/tliang/tmdt-benchmark/anomalib:/workspace/anomalib \
	-v /data/tliang/tmdt-benchmark/data:/workspace/data \
    --rm \
	test-anomalib \
    /bin/bash -c "cd /workspace/anomalib && python benchmark.py"