import argparse
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torchvision.models as models
from torchvision.transforms import Normalize


def main():
    parser = argparse.ArgumentParser(description="Convert a torchvision model to OpenVINO IR.")
    parser.add_argument(
        "--model",
        choices=("mobilenet_v3_small", "efficientnet_b0", "efficientnet_b4", "efficientnet_b5", "resnet50", "convnext_small"),
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: <model>_ov)")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device (default: CPU)")
    parser.add_argument("--height", type=int, help="Input height (defaults to the model's recommended crop height)")
    parser.add_argument("--width", type=int, help="Input width (defaults to the model's recommended crop width)")
    args = parser.parse_args()

    # Load the pre-trained model from torchvision
    weights = models.get_model_weights(args.model).DEFAULT
    model = models.get_model(args.model, weights=weights).eval()
    transforms = weights.transforms()
    crop_size = transforms.crop_size
    default_height, default_width = (crop_size[0], crop_size[0]) if len(crop_size) == 1 else crop_size
    height = args.height if args.height is not None else default_height
    width = args.width if args.width is not None else default_width
    if height <= 0 or width <= 0:
        parser.error("height and width must be positive integers")

    print("Model Input Shape: ", (1, 3, height, width))
    input_tensor = Normalize(mean=transforms.mean, std=transforms.std)(torch.rand(1, 3, height, width))

    with torch.inference_mode():
        pytorch_output = model(input_tensor).numpy()

    output_dir = args.output_dir if args.output_dir is not None else Path(f"{args.model}_ov")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.xml"
    if not model_path.exists():
        converted_model = ov.convert_model(model, example_input=input_tensor, input=[1, 3, height, width])
        # Embedding model-specific preprocessing into the OpenVINO model
        prep = ov.preprocess.PrePostProcessor(converted_model)
        prep.input("x").tensor().set_layout(ov.Layout("nchw"))
        prep.input("x").preprocess().mean([123.675,116.28,103.53])
        prep.input("x").preprocess().scale([58.395,57.12,57.375])
        converted_model = prep.build()
        ov.save_model(converted_model, model_path)

    core = ov.Core()
    ov_model = core.read_model(model_path)    
    compiled_model = core.compile_model(ov_model, args.device)
    openvino_output = compiled_model(input_tensor.numpy())[compiled_model.output(0)]

    print(f"Saved OpenVINO IR: {model_path} and {model_path.with_suffix('.bin')}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"OpenVINO output shape: {openvino_output.shape}")
    print(f"Maximum absolute difference: {np.max(np.abs(pytorch_output - openvino_output)):.6f}")


if __name__ == "__main__":
    main()
