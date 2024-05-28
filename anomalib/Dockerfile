FROM mambaorg/micromamba:1.5-jammy-cuda-12.1.1

WORKDIR /workspace
ADD pad /workspace/pad
ADD src /workspace

USER root
RUN apt-get update \
    && apt-get install -y wget

SHELL ["/bin/bash", "-c"]

RUN micromamba config append channels conda-forge \ 
    && micromamba create -n benchmark python=3.10 -y

SHELL ["micromamba", "run", "-n", "benchmark", "/bin/bash", "-c"]

# Anomalib
RUN pip install anomalib \
    && anomalib install \
    # pkg_resources deprecated from setuptools >= 70
    && pip install setuptools==69.5.1

# ENTRYPOINT ["/bin/bash"]