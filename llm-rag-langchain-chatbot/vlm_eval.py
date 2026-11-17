from pathlib import Path
from PIL import Image
from time import perf_counter
import openvino_genai as ov_genai
import openvino as ov
from io import BytesIO
import requests
import numpy as np

image_path = "data/coco_bike_1024.jpg"
if not Path(image_path).exists():
    source_path = Path("data/coco_bike.jpg")
    resized_path = Path("data/coco_bike_1024.jpg")

    Image.open(source_path).convert("RGB").resize(
        (1024, 1024), Image.Resampling.BICUBIC
    ).save(resized_path)

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)

image_tensor = load_image(image_path)
model_dir = Path("Qwen2.5-VL-7B-Instruct-fp16-ov")
model_cache = str(Path(model_dir) / "model_cache")
ov_properties = {"CACHE_DIR": model_cache, "PERFORMANCE_HINT": "LATENCY"}
pipe = ov_genai.VLMPipeline(str(model_dir), "GPU", **ov_properties)
prompt = "Describe the image."
config = ov_genai.GenerationConfig()
config.max_new_tokens = 64
config.do_sample = False
pipe.generate(prompt, image_tensor, config)

times = []
for _ in range(5):
    start = perf_counter()
    result = pipe.generate(prompt, image_tensor, config)
    times.append(perf_counter() - start)

print(f"Mean generation time: {sum(times) / len(times):.3f}s")
print(result.perf_metrics)