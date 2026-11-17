import argparse
import requests
import torch
from ultralytics import YOLO
from pathlib import Path
import openvino as ov
import cv2 as cv

parser = argparse.ArgumentParser(description="Export a YOLOv8 detection or segmentation model to OpenVINO.")
parser.add_argument("--model", default="yolov8m-seg", help="Model name without the .pt extension")
parser.add_argument("--output-dir", type=Path, help="OpenVINO export directory (default: yolov8_models/<model>_ov)")
parser.add_argument("--device", type=str.upper, default="NPU", help="OpenVINO device (default: NPU)")
parser.add_argument("--imgsz", type=int, default=640, help="Square input image size for export and inference (default: 640)")
args = parser.parse_args()
if args.imgsz < 32 or args.imgsz % 32:
    parser.error("imgsz must be a positive multiple of 32")

models_dir = Path("./yolov8_models")
models_dir.mkdir(exist_ok=True)
MODEL_NAME = args.model
output_dir = args.output_dir or models_dir / f"{MODEL_NAME}_ov"
det_model = YOLO(models_dir / f"{MODEL_NAME}.pt")
if det_model.task not in ("detect", "segment"):
    parser.error(f"Unsupported model task: {det_model.task}; expected detect or segment")
label_map = det_model.model.names

image_path = Path("./data/coco_bike.jpg")
image_path.parent.mkdir(exist_ok=True)
if not image_path.exists():
    r = requests.get(url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg")
    with open(image_path, "wb") as f:
        f.write(r.content)
res = det_model.predict(
    source=image_path,
    imgsz=args.imgsz,
    save=False   
)
output_path = Path("./output") / det_model.task / f"{MODEL_NAME}.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)
res[0].save(filename=str(output_path))
print("Detection results before optimization:", res)

# Optimize the model using OpenVINO
det_model_path = output_dir / f"{MODEL_NAME}.xml"
if not det_model_path.exists():
    det_model.export(
        format="openvino",
        dynamic=False,
        imgsz=args.imgsz,
        half=True,
        project=str(output_dir.parent),
        name=output_dir.name,
    )

# Run inference with the optimized model
core = ov.Core()
det_ov_model = core.read_model(det_model_path)
ov_config = {}
device = args.device
'''
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, args.imgsz, args.imgsz]})
'''
if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
det_compiled_model = core.compile_model(det_ov_model, device, ov_config)

def infer(*args):
    result = det_compiled_model(args)
    if det_model.task == "segment":
        return torch.from_numpy(result[0]), torch.from_numpy(result[1])
    return torch.from_numpy(result[0])

det_model.predictor.inference = infer
det_model.predictor.model.pt = False

# Preprocess input image
image = cv.imread(str(image_path))
image = cv.resize(image, (args.imgsz, args.imgsz))

res = det_model.predict(
    source=image,
    imgsz=args.imgsz,
    save=False   
)
output_path = Path("./output") / det_model.task / f"{MODEL_NAME}_ov.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)
res[0].save(filename=str(output_path))
print("Detection results after optimization:", res)
