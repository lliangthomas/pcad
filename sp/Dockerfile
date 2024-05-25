FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
RUN apt update && apt install -y tzdata
RUN apt install -y git g++ ffmpeg libsm6 libxext6
RUN apt clean && apt install wget && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
ADD splatpose /workspace/splatpose

# NOTE: Does NOT work to install any CUDA related packages in the Dockerfile since CUDA is not enabled during Docker build

# pip install submodules/diff-gaussian-rasterization submodules/simple-knn
# pip install -r requirements.txt