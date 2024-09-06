# hash:sha256:1a9593e47659675c00273af0fec70da25250367d97e9e448be30143ddd0de614
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    brain-observatory-utilities==0.4.0 \
    caveclient==5.25.2 \
    cloud-volume==10.4.0 \
    ipywidgets==8.1.5 \
    k3d==2.16.1 \
    matplotlib==3.9.2 \
    nglui==3.4.0 \
    numpy==1.23.5 \
    pandas==1.5.3 \
    scikit-learn==1.5.1 \
    seaborn==0.13.2 \
    skeleton-plot==0.0.10
