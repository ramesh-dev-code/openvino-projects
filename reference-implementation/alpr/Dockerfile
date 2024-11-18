FROM intel/dlstreamer:devel as openvino_dlstreamer
USER root
RUN apt-get update && \
    apt-get install -y \
        vim \
        git \
        wget \
        build-essential       
RUN pip install 'paddlepaddle==2.6.2' 'pyclipper==1.3.0.post6' 'shapely==2.0.6'
ENV PROJECT_DIR="/home/openvino/"
RUN mkdir -p $PROJECT_DIR 
WORKDIR $PROJECT_DIR
