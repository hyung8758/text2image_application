"""
etri koala: text2image generative AI 

Hyungwon Yang
"""
from datetime import datetime
import torch
from diffusers import StableDiffusionXLPipeline

# You can replace the checkpoint id with several koala models as below:
# "etri-vilab/koala-700m"
# "etri-vilab/koala-1b"
# "etri-vilab/koala-1b-llava-cap"
# num_threads=1
# torch.set_num_threads(num_threads)

# pipe = StableDiffusionXLPipeline.from_pretrained("etri-vilab/koala-700m-llava-cap", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = StableDiffusionXLPipeline.from_pretrained("etri-vilab/koala-700m-llava-cap", use_safetensors=True, variant="fp16")
pipe = pipe.to('cpu')

start_time = datetime.now()
prompt = "Albert Einstein in a surrealist Cyberpunk 2077 world, hyperrealistic"
image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("./Einstein_example.png")

# If you use negative prompt, you could get more stable and accurate generated images.

negative_prompt = '(deformed iris, deformed pupils, deformed nose, deformed mouse), worst  quality, low quality, ugly, duplicate, morbid,  mutilated, extra fingers, mutated hands, poorly drawn hands, poorly  drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad  proportions, extra limbs, cloned face, disfigured, gross proportions,  malformed limbs, missing arms, missing legs'

pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("./Einstein_example_with_neg_prompt.png")

end_time = datetime.now()
print(str(end_time-start_time))
# torch.set_num_threads(torch.get_num_threads())