# Build argumnets
ARG CUDA_VER=12.1.1
ARG UBUNTU_VER=22.04
# Download the base image
FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu${UBUNTU_VER}
# you can check for all available images at https://hub.docker.com/r/nvidia/cuda/tags
# Install as root
USER root
# Shell
SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]
# Python version
ARG PYTHON_VER=3.10
# Install dependencies
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -y --no-install-recommends \
bash \
bash-completion \
ca-certificates \
curl \
git \
htop \
nano \
openssh-client \
python${PYTHON_VER} python${PYTHON_VER}-dev python3-pip python-is-python3 \
sudo \
tmux \
unzip \
vim \
wget \ 
gcc \
zip && \
apt-get autoremove -y && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


WORKDIR /home

RUN pip install --upgrade --no-cache-dir pip setuptools wheel && \
pip install --upgrade --no-cache-dir torch torchvision torchaudio torchtext torchserve lightning && \
pip install triton && \
pip install --upgrade --no-cache-dir \
ipywidgets \
jupyterlab \
matplotlib \
nltk \
notebook \
numpy \
pandas \
Pillow \
plotly \
PyYAML \
scipy \
scikit-image \
scikit-learn \
sympy \
seaborn \
albumentations \
tqdm && \
pip cache purge && \
# Set path of python packages
echo "# Set path of python packages" >>/home/.bashrc && \
echo 'export PATH=$HOME/.local/bin:$PATH' >>/home/.bashrc
EXPOSE 8888
USER root
# ENTRYPOINT ["jupyter", "lab", "--no-browser", "--port", "8888", "--ServerApp.token=''", "--ip='*'", "--allow-root"]