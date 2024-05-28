FROM mambaorg/micromamba:1.5-jammy-cuda-12.1.1

WORKDIR /workspace
ADD pad /workspace/pad
ADD requirements.txt /workspace

USER root
RUN apt-get update \
    && apt-get install -y wget

SHELL ["/bin/bash", "-c"]

RUN micromamba config append channels conda-forge \ 
    && micromamba create -n benchmark python=3.10 -y

SHELL ["micromamba", "run", "-n", "benchmark", "/bin/bash", "-c"]

# PAD
RUN pip install -r requirements.txt