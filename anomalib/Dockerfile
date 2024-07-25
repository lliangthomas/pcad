FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

USER root
RUN apt-get update \
    && apt-get install -y wget ffmpeg libsm6 libxext6

# Anomalib
RUN git clone https://github.com/openvinotoolkit/anomalib.git anomalib-git \
    && cd anomalib-git \
    && git checkout c3e0b38 \
    && pip install . \
    && anomalib install \
    # pkg_resources deprecated from setuptools >= 70
    && pip install setuptools==69.5.1

# ENTRYPOINT ["/bin/bash"]