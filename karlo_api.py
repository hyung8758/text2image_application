"""
kakaobrain karlo: text2image generative AI 

Hyungwon Yang
"""
import urllib
from PIL import Image
from src.modules import karlo

# prompt = "넓은 들판에 사과 나무 한그루가 서있다. 그 주변을 아이두명이 해맑은 표정으로 뛰어다니고 있다. 사과나무에는 사과가 풍성하게 열렸다."
# negative_prompt = "저녁"
prompt = """
Many boats are floating on the lake. 
People on those boats are looking around them and enjoying the sightseeing.
it looks peaceful and calm. it is spring so a lot of flowers are bloomed already.
flowers should be located by a lake side.
"""
negative_prompt = ""

width = 512
height = 512
image_format = "png"
samples = 1 # 생성 이미지 파일 수.

nsfw_checker = False

# [내 애플리케이션] > [앱 키] 에서 확인한 REST API 키 값 입력
REST_API_KEY = '9a6fcb6420c4ccfe09f7176f1f199ac3'

response = karlo.text2image(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, image_format=image_format, samples=samples)
print("response: {}".format(response))
result = Image.open(urllib.request.urlopen(response.get("images")[0].get("image")))
result.show()


## 직접 모델 다운받아 사용해보기
from diffusers import UnCLIPPipeline
import torch

pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
pipe = pipe.to('cuda')

prompt = "a high-resolution photograph of a big red frog on a green leaf."
image = pipe(prompt).images[0]
image.save("./frog.png")