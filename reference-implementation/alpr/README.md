# Automatic License Plate Recognition

## Software   
OS: Ubuntu 20.04/22.04   
Docker   
Intel DL Streamer 2022.3.0   
Intel OpenVINO Toolkit 2022.3.0   

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
3. Run the ALPR pipeline   
```
./run_alpr_pp_ocr.sh   
```
**Note**: On successful execution of the above script, the inference results are saved in out.txt file in the current working directory   
