
import requests
import json

class Karlo:
    
    def __init__(self, REST_API_KEY):
        self.REST_API_KEY = REST_API_KEY
    
    # 이미지 생성
    def text2image(self, prompt:str, negative_prompt:str, width:int=512, height:int=512, image_format:str="png", samples:int=1) -> json:
        r = requests.post(
            'https://api.kakaobrain.com/v2/inference/karlo/t2i',
            json = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width' : width,
                'height' : height,
                'image_format' : image_format,
                'samples' : samples
                
            },
            headers = {
                'Authorization': f'KakaoAK {self.REST_API_KEY}',
                'Content-Type': 'application/json'
            }
        )
        # 응답 JSON 형식으로 변환
        response = json.loads(r.content)
        return response