# Waifu2x Dockerfile running in CPU Only mode
# Not recommended for training.

FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

RUN apt-get update && apt-get upgrade -y && apt-get install -y python3.10 python3-venv python3-pip git python3-distutils vim zip unzip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
ENV TORCH_CUDA_ARCH_LIST "compute capability"
RUN git clone https://github.com/NVIDIA/apex && cd apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" ./

RUN mkdir /Waifu2x
WORKDIR /Waifu2x