from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import os

# =========================
# Config
# =========================
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
INPUT_IMAGE_PATH = "custom_dataset/sneaker1.jpg"
OUTPUT_DIR = "outputs_variations"

# 2 prompts
PROMPTS = [
    "A futuristic version of the sneakers with neon accents and a dark background",
    "A luxury product photo of sneakers in a studio with soft lighting, ultra realistic"
]

NUM_IMAGES_PER_PROMPT = 5  # 2 prompts × 5 = 10 imagens ✅

GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
STRENGTH = 0.7

# Create folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load model
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# =========================
# Load base image
# =========================
init_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
init_image = init_image.resize((768, 768))

print(f"Using device: {device}")

# =========================
# Generate images
# =========================
image_counter = 1

for prompt in PROMPTS:
    for i in range(NUM_IMAGES_PER_PROMPT):

        image = pipe(
            prompt=prompt,
            image=init_image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS
        ).images[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"img_{image_counter}.png"
        )

        image.save(output_path)
        print(f"Saved: {output_path}")

        image_counter += 1

