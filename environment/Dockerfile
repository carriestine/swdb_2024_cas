# hash:sha256:f6c4c7a927542259fae7895dac8c3905037fb2f1c7bffb6331c2039747646842
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    brain-observatory-utilities==0.4.0 \
    caveclient \
    cloud-volume \
    ipywidgets \
    k3d \
    matplotlib \
    nglui \
    numpy \
    pandas==1.5.3 \
    scikit-learn==1.5.1 \
    seaborn==0.13.2 \
    skeleton-plot
