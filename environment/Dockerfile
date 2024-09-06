# hash:sha256:4643ef6bcdbf63afc669b572561302d41705768ac5094d52b9e248ed079bbc1d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    brain-observatory-utilities==0.4.0 \
    matplotlib \
    numpy \
    pandas==1.5.3 \
    scikit-learn==1.5.1 \
    seaborn==0.13.2
