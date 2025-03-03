# Improved Dwell Time Aanalytics with Person Re-Identification   

## Device under Test   
Model: NUC11TNHi3   
Processor: Intel Core i3-1115G4   
iGPU: Intel UHD Graphics    
RAM: 16GB   
OS: Ubuntu 20.04   
Docker Image: [intel/dlstreamer:devel](https://hub.docker.com/layers/intel/dlstreamer/devel/images/sha256-3d211ab50bcdd3d9c4a71d18893c826e9a3717d3301f4a1b5b7aa68563ce78d5?context=explore)   
Intel DL Streamer 2023.0.0   
Intel OpenVINO Toolkit 2023.0.0   

## Models
Person Detection - [person-detection-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-detection-retail-0013/README.md)   
Person Re-Identification- [person-reidentification-retail-0277](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0277/README.md)   

## Installation Steps   
1. Clone this repository and rename the folder as alpr
```
cd dwell-time-analytics   
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
./run_dwell_time_analytics.sh   
```
### Sample Output   
![image](https://github.com/user-attachments/assets/8b0d9d28-ce34-4dbf-9603-97ff0cb3aeb4)
