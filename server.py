"""
text2image server.
# run in 
    - uvicorn: 
        - console: $ python server.py
        - nohup background: $ nohup python server.py > text2image.log &
    - gunicorn: 
        - num_jb=4
        - gunicorn -w $num_jb -k uvicorn.workers.uvicornWorker server:app -b 0.0.0.0:33010

Hyungwon Yang
24.03.19
"""

import os
import io
import yaml
import logging
import base64
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from src.handlers.ImageHandler import ImageHandler
from src.handlers.TextHandler import TextHandler

# Create FastAPI app instance
app = FastAPI()

# Initialize handlers
text_handler = TextHandler()
image_handler = ImageHandler()
text_handler.initialize(model="ke-t5", use_cuda=False)
image_handler.initialize(model="karlo", cuda_device=1)
# Assign the handlers to the app's state
app.state.image_handler = image_handler
app.state.text_handler = text_handler

def compress_image(image, quality=85):
    """
    Compresses the given PIL image using JPEG compression.
    
    Inputs:
        image (PIL.Image): The input image to compress.
        quality (int): The compression quality (0-100, default=85).
    
    outputs:
        PIL.Image: The compressed image.
    """
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='JPEG', quality=quality)
    img_byte_array.seek(0)
    return Image.open(img_byte_array)

@app.post("/text2image")
async def post(request: Request):
    logging.info("start post request.")
    response = {"message": None, "success": True,  "image": None}
    # Retrieve the handlers using dependency injection
    image_handler = request.app.state.image_handler
    text_handler = request.app.state.text_handler
    json_data = await request.json()
    logging.info(json_data)
    input_text = json_data['input_text']
    save_image = json_data['save_image']
    save_name = json_data['save_name']
    logging.info("INPUT TEXT: {}".format(input_text))
    if text_handler:
        input_text = text_handler.run(input_value=input_text)
        logging.info("text_handler output: {}".format(input_text))
    if image_handler:
        output_image = image_handler.run(input_value=input_text, save_image=save_image, save_name=save_name)
    logging.info("image_handler output: {}".format(output_image))
    
    # Compress the image using JPEG compression
    compressed_image = compress_image(output_image)
    
    # Convert compressed image to base64-encoded string
    img_byte_array = io.BytesIO()
    compressed_image.save(img_byte_array, format='JPEG')
    img_byte_array = img_byte_array.getvalue()
    base64_encoded_img = base64.b64encode(img_byte_array).decode('utf-8')
    response['message'] = "sending a result image."
    response['image'] = base64_encoded_img
    logging.info("done post request.")
    return response

# Entry point for running with Gunicorn
if __name__ == "__main__":
    import uvicorn
    import yaml
    
    cur_dir = os.getcwd()
    server_conf_path = os.path.join(cur_dir, "conf", "server.yaml")
    with open(server_conf_path) as f:
        server_conf = yaml.load(f, Loader=yaml.FullLoader)
    uvicorn.run(app, host=server_conf['server_host'], port=server_conf['server_port'])