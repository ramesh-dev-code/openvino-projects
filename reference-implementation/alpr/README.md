# Automatic License Plate Recognition

## Software   
OS: Ubuntu 20.04   
Docker Image: [intel/dlstreamer:devel](https://hub.docker.com/layers/intel/dlstreamer/devel/images/sha256-3d211ab50bcdd3d9c4a71d18893c826e9a3717d3301f4a1b5b7aa68563ce78d5?context=explore)   
Intel DL Streamer 2023.0.0   
Intel OpenVINO Toolkit 2023.0.0   
PaddlePaddle 2.6.2

## Models
Vehicle license plate detection - [vehicle-license-plate-detection-barrier-0123](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/vehicle-license-plate-detection-barrier-0123)   
Text detection - [horizontal-text-detection-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/horizontal-text-detection-0001)   
Text recognition - [en_PP-OCRv3_rec](https://github.com/PaddlePaddle/PaddleOCR)   

## Installation Steps   
1. Clone this repository and rename the folder as alpr
```
cd alpr   
chmod -R 777 ./   
```
3. Build the docker image with Intel DL Streamer and application dependencies   
```
./scripts/build.sh
```
2. Download the required models   
```
./scripts/download_model.sh
```
3. Copy the input video file (named as test_input.mp4) to the assets directory and run the ALPR pipeline    
```
./run_alpr_pp_ocr.sh   
```
**Note**: On successful execution of the above script, the inference results are saved in out.txt file in the current working directory   
