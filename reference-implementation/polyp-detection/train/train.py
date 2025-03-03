from ultralytics import YOLO
import torch
model = YOLO("yolo11n.pt")

train_results = model.train(
    data="coco_polyp.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device=torch.device('xpu'), # device to run on
    amp=False	
)
metrics = model.val()
