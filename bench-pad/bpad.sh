#!/bin/bash

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 2-0:00
#SBATCH -o /data/tliang/log/job-%x-%A.out
#SBATCH -e /data/tliang/log/job-%x-%A.err

# echo "Slurm Visible Cuda Devices:"
# echo $CUDA_VISIBLE_DEVICES
# gpus="--gpus \"device=${CUDA_VISIBLE_DEVICES}\""
# echo $gpus

cd /home/thomasl/tmdt-benchmark/bench-pad/nerf-pytorch-docker
docker build . -t nerf-pytorch

# Train NeRF
docker run --gpus all \
	-v /home/thomasl/tmdt-benchmark/synthetic-corrected:/workspace/data \
	-v /home/thomasl/tmdt-benchmark/bench-pad/nerf-pytorch-docker:/workspace/nerf-pytorch-docker \
    --rm \
	-it \
	nerf-pytorch \
    /bin/bash -c "export CONDA_PREFIX=/opt/conda && echo $CONDA_PREFIX && cd nerf-pytorch-docker && python run_nerf.py --config configs/mguard-synthetic.txt --i_testset 100 --i_video 100"