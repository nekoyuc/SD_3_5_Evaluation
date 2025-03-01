import torch
from huggingface_hub import login
from diffusers import StableDiffusion3Pipeline

#login()

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

with open("prompts.txt", "r") as file:
    prompts = file.readlines()

i = 0

for prompt in prompts:
    prompt = prompt.strip()
    if prompt.endswith(","):
        prompt = prompt[:-1]
    prompt = prompt[1:-1]
    image1 = None
    image2 = None
    image1 = pipeline(
        prompt=prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    image2 = pipeline(
        prompt=prompt,
        num_inference_steps=40,
        guidance_scale=2.5,
    ).images[0]
    image1.save(f"results/{prompt}_1.png")
    image2.save(f"results/{prompt}_2.png")
    i += 1
    print(f"Finished {i} prompts")

'''
prompt = "A whimsical illustration of a small village in the forest, with a river running through it."

image = None
image = pipeline(
        prompt=prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
image.save("whimsical.png")
'''
