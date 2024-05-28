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

docker pull dromni/nerfstudio:1.1.0
docker run $gpus --shm-size 96G \
	-v /data/tliang/tmdt-benchmark/pad/git-pad:/workspace/pad \
	-v /data/tliang/tmdt-benchmark/data:/workspace/data \
    -v /data/tliang/tmdt-benchmark/colmap-data:/workspace/colmap-data \
    --rm \
	dromni/nerfstudio:1.1.0 \ 
    ns-process-data images --data /workspace/data/class-01/train/good --output-dir /workspace/colmap-data