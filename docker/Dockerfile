# Conservative reference environment. Validate actual GPU kernels on the host.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}
RUN python -m pip install --no-cache-dir torchvision==0.20.1 numpy==2.1.3 Pillow==11.0.0 scikit-image==0.24.0
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /code
# Code and private data are mounted separately; no credentials in the image.
