# Polyp Detection in Input Video with Optimized YOLOv11n Model using OpenVINO   

## Device Under Test   
Processor: Intel Core Ultra 7 165U   
RAM: 32GB   
OS: Widows 11 Enterprise (23H2)   
Python: 3.10.11   
OpenVINO: 2025.0.0   
Ultralytics: 8.3.59   

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
```
python yolov11n_polyp_detection.py CPU
python yolov11n_polyp_detection.py GPU
python yolov11n_polyp_detection.py NPU
```

### Detection Result
