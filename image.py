from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import os

# =========================
# Config
# =========================
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
INPUT_IMAGE_PATH = "custom_dataset/base_image.png"  # sua imagem base
OUTPUT_DIR = "outputs_variations"

PROMPT = "A futuristic version of the product with neon accents and a dark background"

GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
STRENGTH = 0.7  # controla o quanto a imagem muda (0.3 = pouco, 0.8 = muito)
NUM_IMAGES = 3

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load model
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Set scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.to(device)

print(f"Using device: {device}")

# =========================
# Load base image
# =========================
init_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
init_image = init_image.resize((768, 768))  # tamanho padrão para SD 2.1

# =========================
# Generate variations
# =========================
for i in range(NUM_IMAGES):

    image = pipe(
        prompt=PROMPT,
        image=init_image,
        strength=STRENGTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS
    ).images[0]

    output_path = os.path.join(OUTPUT_DIR, f"variation_{i+1}.png")
    image.save(output_path)

    print(f"Saved: {output_path}")
