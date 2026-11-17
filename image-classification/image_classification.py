import argparse
import cv2 as cv
import numpy as np
import openvino as ov
from pathlib import Path
import requests 
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser(description="Run image classification with an OpenVINO model.")
parser.add_argument("--model", default="resnet50", help="Model name; loads <model>_ov/model.xml")
parser.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU)")
args = parser.parse_args()

# Load OpenVINO model
core = ov.Core()
model_path = Path(f"{args.model}_ov") / "model.xml"
ov_model = core.compile_model(model=model_path, device_name=args.device)
input_shape = ov_model.input(0).shape
height, width = input_shape[2], input_shape[3]
output_layer = ov_model.output(0)

# Prepare the input image
image_filename = Path("data/coco_bike.jpg")
if not image_filename.exists():
    response = requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg")
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(image_filename)

image = cv.cvtColor(cv.imread(filename=str(image_filename)), code=cv.COLOR_BGR2RGB)
# Resize to model input shape
input_image = cv.resize(src=image, dsize=(width, height))
input_image = input_image.transpose(2, 0, 1)
# Reshape to model input shape
input_image = np.expand_dims(input_image, 0)
print("Input image shape: ", input_image.shape)

# Run inference
result_infer = ov_model([input_image])[output_layer]
result_index = np.argmax(result_infer)

# Get the predicted class from the ImageNet 2012 label file
imagenet_filename = Path("data/imagenet_2012.txt")
if not imagenet_filename.exists():
    response = requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt")
    imagenet_filename.write_text(response.text)

imagenet_classes = imagenet_filename.read_text().splitlines()
print('Inference Result: ', imagenet_classes[result_index])