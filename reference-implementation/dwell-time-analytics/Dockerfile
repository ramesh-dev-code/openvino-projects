FROM intel/dlstreamer:devel as openvino_dlstreamer
USER root
RUN apt-get update && \
    apt-get install -y \
        vim \
        git \
        wget \
        build-essential       
RUN pip install munkres
ENV PROJECT_DIR="/home/dlstreamer/"
RUN mkdir -p $PROJECT_DIR 
WORKDIR $PROJECT_DIR
