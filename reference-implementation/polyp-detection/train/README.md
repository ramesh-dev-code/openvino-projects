## Training YOLOv11n on Colonoscopy Dataset using Intel Arc A770 GPU
### Dataset
Download the [Colonoscopy dataset](https://github.com/dashishi/LDPolypVideo-Benchmark?tab=readme-ov-file#download)   
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

#### Prerequisite
Install [GPU/OpenCL driver for Ubuntu 22.04](https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop-22-04-lts)      
Install [Intel-Optimized PyTorch 2.5](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html) for training the model on Intel GPUs

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
Create the datasets/coco_polyp folder with the polyp dataset in the ultralytics/ultralytics subfolder as below   
![image](https://github.com/user-attachments/assets/c896ed85-2fc1-4c70-9a63-4a8fd8c83a80)

#### Create the dataset configuration file to train the model
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
![image](https://github.com/user-attachments/assets/1ad9a5ef-0e04-4110-bc17-0a6dd2e7ed55)

