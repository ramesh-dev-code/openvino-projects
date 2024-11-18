#!/bin/bash
GST_PIPELINE_CMD="gst-launch-1.0 filesrc location=assets/test_input.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=BGRx ! queue ! gvadetect model-proc=models/model-proc/vehicle-license-plate-detection-barrier-0123.json model=models/public/vehicle-license-plate-detection-barrier-0123/FP16/vehicle-license-plate-detection-barrier-0123.xml threshold=0.5 inference-interval=5 device=CPU ! queue ! gvametaconvert ! gvapython module=assets/lp_txt_det_pp_ocr.py ! gvametapublish file-path=./out.txt ! videoconvert ! fpsdisplaysink video-sink=xvimagesink sync=True"
xhost local:root
docker run -it --rm -u root \
	-d --name alpr-pp-ocr \
	-e DISPLAY=$DISPLAY \
	--device /dev/dri:/dev/dri \
	--group-add="$(stat -c "%g" /dev/dri/render*)" \
	-v /dev/bus/usb:/dev/bus/usb \
	-v ~/.Xauthority:/home/dlstreamer/.Xauthority \
	-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
       	-v $PWD:/home/openvino \
	ov_dlstreamer:latest \
	bash -c "$GST_PIPELINE_CMD"

