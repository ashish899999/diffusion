# Install required libraries
!pip install diffusers transformers accelerate scipy safetensors

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Stable Diffusion pipeline
# Using smaller model variant for faster generation
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use fp16 for faster inference
    variant="fp16",
    use_safetensors=True
).to(device)

# Optimize for performance
pipe.enable_attention_slicing()  # Reduces memory usage with minimal performance impact
# pipe.enable_xformers_memory_efficient_attention()  # Uncomment if xformers is installed

def generate_image(prompt, negative_prompt=None, num_images=1):
    """Generate images from text prompt"""
    with torch.autocast(device):  # Automatic mixed precision for faster inference
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=25,  # Balance between quality and speed
            guidance_scale=7.5,       # Controls how closely the image follows the prompt
            width=512,                # Standard size for good results
            height=512
        ).images

    # Display images
    for img in images:
        display(img)

    return images

# Example usage
prompt = "a realistic photo of a astronaut riding a horse on mars, 4k, high resolution"
negative_prompt = "blurry, low quality, cartoon, anime, deformed"

generated_images = generate_image(prompt, negative_prompt)

# Optional: Save images
for i, img in enumerate(generated_images):
    img.save(f"generated_image_{i}.png")
