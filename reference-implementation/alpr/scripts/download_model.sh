#!/bin/bash
docker run -t --rm \
    --name intel_download_model \
    -v $PWD:/home/openvino \
    ov_dlstreamer:latest \
    /bin/bash -c "omz_downloader --name horizontal-text-detection-0001 -o /home/openvino/models --precision FP16-INT8; \
    omz_downloader --name vehicle-license-plate-detection-barrier-0123 -o /home/openvino/models --precision FP16; \
    omz_converter --name vehicle-license-plate-detection-barrier-0123 -d /home/openvino/models -o /home/openvino/models --precision FP16; \
    rm -rf /home/openvino/models/public/vehicle-license-plate-detection-barrier-0123/model; chown -R `id -u`:`id -g` /home/openvino/models"
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar -P models/
tar -xvf models/en_PP-OCRv3_rec_infer.tar -C models/ 
rm models/en_PP-OCRv3_rec_infer.tar
