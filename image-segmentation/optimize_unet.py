import argparse
from pathlib import Path

import numpy as np
import openvino as ov
import segmentation_models_pytorch as smp
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert a segmentation_models_pytorch U-Net to OpenVINO IR.")
    parser.add_argument("--encoder", default="resnet34", help="U-Net encoder name")
    parser.add_argument("--encoder-weights", choices=("imagenet",), default="imagenet")
    parser.add_argument("--checkpoint", type=Path, help="Path to a U-Net PyTorch state_dict")
    parser.add_argument("--classes", type=int, default=2, help="Number of output channels")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("unet_ov"))
    parser.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU)")
    args = parser.parse_args()

    if args.classes < 1 or args.height < 32 or args.width < 32 or args.height % 32 or args.width % 32:
        parser.error("classes must be positive and height and width must be positive multiples of 32")

    model = smp.Unet(encoder_name=args.encoder, encoder_weights=args.encoder_weights, in_channels=3, classes=args.classes)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    else:
        print("No checkpoint supplied")
        tmp_model = smp.from_pretrained('jq/resnet34-unet-malaria')
        model.load_state_dict(tmp_model.state_dict())
        #torch.save(model.state_dict(), 'unet_weights.pt')
    model.eval()

    input_tensor = torch.rand(1, 3, args.height, args.width)
    with torch.inference_mode():
        pytorch_output = model(input_tensor).numpy()

    converted_model = ov.convert_model(model, example_input=input_tensor, input=[1, 3, args.height, args.width])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.xml"
    ov.save_model(converted_model, model_path)

    compiled_model = ov.Core().compile_model(model_path, args.device)
    openvino_output = compiled_model(input_tensor.numpy())[compiled_model.output(0)]

    print(f"Saved OpenVINO IR: {model_path} and {model_path.with_suffix('.bin')}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"OpenVINO output shape: {openvino_output.shape}")
    print(f"Maximum absolute difference: {np.max(np.abs(pytorch_output - openvino_output)):.6f}")


if __name__ == "__main__":
    main()