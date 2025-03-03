# Deploying YOLOv11n-based Polyp Detection in Colonoscopy Video on AI PC Model using OpenVINO   

## Device Under Test   
Processor: Intel Core Ultra 7 165U   
iGPU: Intel Graphics (Driver: 32.0.101.5763)   
NPU: Intel AI Boost (Driver: 32.0.100.3104)   
RAM: 32GB   
OS: Widows 11 Enterprise (23H2)   
Python: 3.10.11   
OpenVINO: 2025.0.0   
Ultralytics: 8.3.59   

## Prerequisites   
Install [GPU driver for Windows](https://www.intel.com/content/www/us/en/download/785597/846697/intel-arc-iris-xe-graphics-windows.html)    
Install [NPU driver for Windows](https://www.intel.com/content/www/us/en/download/794734/838895/intel-npu-driver-windows.html) using these [steps](https://downloadmirror.intel.com/838895/NPU_Win_Release_Notes_v3104.pdf)   

## Installation   
1. Create a Python virtual environment
   ``` 
   python -m venv polyp_det_venv
   .\polyp_det_venv\Scripts\activate
   python -m pip install pip --upgrade     
   ```   
2. Install the dependencies
   ```
   pip install openvino==2025.0.0 ultralytics==8.3.59   
   ```  
  
## Model Optimization   
```
python optimize_yolov11n.py
```

## Run Polyp Detection   
Download the [input video](https://github.com/dashishi/LDPolypVideo-Benchmark?tab=readme-ov-file#download) and provide the valid path to the input video and the optimized model (.xml) in yolov11n_polyp_detection.py
```
python yolov11n_polyp_detection.py CPU
python yolov11n_polyp_detection.py GPU
python yolov11n_polyp_detection.py NPU
```

### Detection Results   
CPU   
![image](https://github.com/user-attachments/assets/a083d8fb-2777-4602-aecd-37c45e369231)

iGPU   
![image](https://github.com/user-attachments/assets/8ff22b2d-161f-4d53-8687-47a493accdc2)
NPU   
![image](https://github.com/user-attachments/assets/a28e69f4-acd5-4a56-b337-d31318d10a04)

