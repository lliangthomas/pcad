FROM mambaorg/micromamba:1.5-jammy-cuda-12.1.1

WORKDIR /workspace
# ADD dataset /workspace/dataset ## Better to download the dataset in the container instead?
ADD pad /workspace/pad
ADD splatpose /workspace/splatpose
ADD src /workspace

RUN micromamba create -n benchmark python=3.10 -y \
    && micromamba activate benchmark

# Anomalib
RUN pip install anomalib -y \
    && anomalib install \
    # pkg_resources deprecated from setuptools >= 70
    && pip install setuptools==69.5.1

# PAD
RUN pip install -r requirements.txt

# SplatPose
RUN cd splatpose \
    && micromamba env create --file environment.yml \
    && micromamba activate gaussian_splatting \
    && pip install submodules\diff-gaussian-rasterization \
    && pip install submodules\simple-knn

ENTRYPOINT ["/bin/bash"]