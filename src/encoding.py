"""
encoding script
from kakao brain

edited by Hyungwon Yang
"""

import io
import base64
from PIL import Image

# Base64 인코딩 함수
def img2str(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img

# Base64 디코딩 및 이미지 변환 함수
def str2img(base64_string, mode='RGBA'):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata)).convert(mode)
    return img