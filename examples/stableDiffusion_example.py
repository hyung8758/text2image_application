from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch

# gpu: float 16, cpu: float 32
float_type = torch.float32

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=float_type)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=float_type,
).to("cpu")

# # lora weights
# pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl")
# pipeline.fuse_lora(lora_scale=0.7)

# # to unfuse the LoRA weights
# pipeline.unfuse_lora()
# pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora")
# pipeline.fuse_lora(lora_scale=0.7)

prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image

