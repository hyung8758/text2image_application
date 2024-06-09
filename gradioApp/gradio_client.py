""" instruction (gpu version)
아래 명령어는 text2image_application/ 디렉토리를 기준으로 실행할 것. 

1. Docker compose 활용하여 triton server 실행.
$ cd 

2. container 내부로 들어온 뒤 아래 라이브러리 설치 진행.
$ pip install torch torchvision torchaudio
$ pip install transformers ftfy scipy accelerate
$ pip install diffusers
$ pip install transformers[onnxruntime]

3. triton 서버 실행.
$ export PYTHONIOENCODING=UTF-8
$ tritonserver --model-repository=/models

4. 기존 triton 서버 terminal은 놔두고, terminal을 새롭게 열어 client 파일 실행.
$ cd gradioApp
$ python3 text2karloImageCpuClient.py

"""
import argparse

import gradio as gr
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--triton_url", default="localhost:8001")
# parser.add_argument("--iter", default=20, type=int)
args = parser.parse_args()

client = grpcclient.InferenceServerClient(url=f"{args.triton_url}")

def generate(prompt):
    text_obj = np.array([prompt], dtype="object").reshape((-1, 1))
    input_text = grpcclient.InferInput(
        "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
    )
    input_text.set_data_from_numpy(text_obj)

    output_img = grpcclient.InferRequestedOutput("generated_image")
    response = client.infer(
        model_name="pipeline", inputs=[input_text], outputs=[output_img]
    )
    resp_img = response.as_numpy("generated_image")
    print(resp_img.shape)
    im = Image.fromarray(np.squeeze(resp_img.astype(np.uint8)))
    return im

with gr.Blocks() as app:
    prompt = gr.Textbox(label="Prompt")
    submit_btn = gr.Button("Generate")
    img_output = gr.Image()
    submit_btn.click(fn=generate, inputs=prompt, outputs=img_output)

app.launch(height=1024,server_name="0.0.0.0")
