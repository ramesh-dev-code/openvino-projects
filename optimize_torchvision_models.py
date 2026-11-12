import argparse
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torchvision.models as models


def main():
    parser = argparse.ArgumentParser(description="Convert a torchvision model to OpenVINO IR.")
    parser.add_argument(
        "--model",
        choices=("mobilenet_v3_small", "efficientnet_b0", "efficientnet_b4", "efficientnet_b5", "resnet50", "convnext_small"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device (default: CPU)")
    args = parser.parse_args()

    # Load the pre-trained model from torchvision
    weights = models.get_model_weights(args.model).DEFAULT
    model = models.get_model(args.model, weights=weights).eval()
    transforms = weights.transforms()
    crop_size = transforms.crop_size
    height, width = (crop_size[0], crop_size[0]) if len(crop_size) == 1 else crop_size
    print("Model Input Shape: ", (1, 3, height, width))
    input_tensor = transforms(torch.rand(1, 3, height, width))

    with torch.inference_mode():
        pytorch_output = model(input_tensor).numpy()

    if not args.output_dir.exists():
        converted_model = ov.convert_model(model, example_input=input_tensor, input=[1, 3, height, width])
        output_dir = args.output_dir or Path(f"{args.model}_ov")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.xml"
        ov.save_model(converted_model, model_path)

    device = args.device

    compiled_model = ov.Core().compile_model(args.output_dir / "model.xml", device)
    openvino_output = compiled_model(input_tensor.numpy())[compiled_model.output(0)]

    print(f"Saved OpenVINO IR: {model_path} and {model_path.with_suffix('.bin')}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"OpenVINO output shape: {openvino_output.shape}")
    print(f"Maximum absolute difference: {np.max(np.abs(pytorch_output - openvino_output)):.6f}")


if __name__ == "__main__":
    main()