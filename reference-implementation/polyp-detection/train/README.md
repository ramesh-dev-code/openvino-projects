## Training YOLOv11n on Colonoscopy Dataset using Intel Arc A770 GPU
### [Colonoscopy Dataset](https://github.com/dashishi/LDPolypVideo-Benchmark?tab=readme-ov-file)
Normalize the annoation coordinates to YOLO format using preprocess.py   
### Model Training   
#### Training Platform   
Host: Intel Core Ultra 7 155H Processor   
Discrete GPU: Intel Arc A770 GPU (Driver: 24.39.31294)   
OS: Ubuntu 22.04 (Kernel: 6.8.0-52-generic)   
Python: 3.10.12    
torch: 2.5.1+xpu      
torchvision: 0.20.1+xpu   
torchaudio: 2.5.1+xpu   
Ultralytics: 8.3.72       

#### Installation    
1. Install dependencies & customize Ultralytics repo for training YOLOv11n on Intel GPU
```
pip install ultralytics==8.3.72
pip install “torch==2.5.1+xpu” “torchvision==0.20.1+xpu” “torchaudio==2.5.1+xpu” --index-url https://download.pytorch.org/whl/test/xpu
git clone https://github.com/ultralytics/ultralytics.git   
```
2. Comment out the select_device function body in [torch_utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py) and change the return statement as below    
```
def select_device(device="", batch=0, newline=False, verbose=True):
   return torch.device("xpu")   
```
#### Prepare the dataset   
```
#coco_polyp.yaml
path: /home/edge-ai/Ramesh/Onyx/ultralytics/ultralytics/datasets/coco_polyp
train: images/train
val: images/val
# Classes
names:
  0: polyp
```
#### Train YOLOv11n on A770 GPU    
```
python train.py
```
Use the following command to verify the utilization of A770 GPU   
```
sudo intel_gpu_top   
```
