#!/bin/bash
docker run -t --rm \
    --name intel_download_model \
    -v $PWD:/home/dlstreamer \
    ov_dlstreamer:latest \
    /bin/bash -c "omz_downloader --name person-detection-retail-0013 -o /home/dlstreamer/models --precision FP16; \
    omz_downloader --name person-reidentification-retail-0277 -o /home/dlstreamer/models --precision FP16; \
    omz_converter --name person-reidentification-retail-0277 -d /home/dlstreamer/models -o /home/dlstreamer/models --precision FP16; \
    chown -R `id -u`:`id -g` /home/dlstreamer/models"
