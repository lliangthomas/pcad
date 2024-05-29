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

cd nerf-pytorch-docker
docker build . -t nerf-pytorch
cd ..
docker run $gpus --shm-size 96G \
	-v /data/tliang/tmdt-benchmark/nerf-pytorch-docker/pad:/workspace/nerf-pytorch-docker \
	-v /data/tliang/tmdt-benchmark/data:/workspace/data \
    --rm \
	nerf-pytorch \
    /bin/bash -c "bash download_example_data.sh && python run_nerf.py --config configs/lego.txt"