#!/bin/bash
xhost local:root
docker run -it --rm -u root \
	-d --name dwell-time-analytics-reid \
	-e DISPLAY=$DISPLAY \
	--device /dev/dri:/dev/dri \
	--group-add="$(stat -c "%g" /dev/dri/render*)" \
	-v /dev/bus/usb:/dev/bus/usb \
	-v ~/.Xauthority:/home/dlstreamer/.Xauthority \
	-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
       	-v $PWD:/home/dlstreamer \
	ov_dlstreamer:latest \
	bash -c "python3 assets/person_track_dwell_reid.py assets/test_input.mp4 5"
