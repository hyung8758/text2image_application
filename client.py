"""
client.py: test client.

Hyungwon Yang
24.03.19
MediaZen
"""

# restful clinet 
import numpy as np
import io
import json
import time
import base64
import argparse
from PIL import Image
from typing import Any
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

class postRequestClient:
    def __init__(self, server_addr: str="localhost", server_port: int=8001) -> None:
        self.conn = grpcclient.InferenceServerClient(url=f"{server_addr}:{server_port}")
    
    def save_image(self, encoded_image: str, save_path: str) -> None:
        # Decode base64 string to bytes
        decoded_img_bytes = base64.b64decode(encoded_image)
        
        # Create a BytesIO object to read the decoded image bytes
        img_byte_stream = io.BytesIO(decoded_img_bytes)
        
        # Open the image using PIL
        image = Image.open(img_byte_stream)
        
        # Save the image to the specified file path
        image.save(save_path)
        
    def __call__(self, json_data: json, *args: Any, **kwargs: Any) -> Any:
        try:
            print("config yaml: {}".format(json_data))
            prompt = json_data['input_text']
            text_obj = np.array(prompt, dtype="object").reshape((-1, 1))
            input_text = grpcclient.InferInput(
                            "input_text", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
                        )
            input_text.set_data_from_numpy(text_obj)
            output_img = grpcclient.InferRequestedOutput("generated_image")
            # print(f"output_img:{output_img}")
            response = self.conn.infer(
                            model_name="ensemble_model", inputs=[input_text], outputs=[output_img]
                        )
            resp_img = response.as_numpy("generated_image")
            if json_data["save_image"]:
                im = Image.fromarray(np.squeeze(resp_img.astype(np.uint8)))
                im.save(json_data["save_name"])
        except Exception as e:
            print("Error: {}".format(e))
        finally:
            self.conn.close()
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_host",
                        type=str,
                        default="127.0.0.1",
                        help="client api server port.")
    parser.add_argument("--server_port",
                        type=int,
                        default=8001,
                        help="client api server port.")
    
    args = parser.parse_args()
    
    client = postRequestClient(server_addr=args.server_host, server_port=args.server_port)
    json_request = dict(
        input_text="초록색의 개구리 한 마리가 나뭇잎 위에 앉았다.",
        save_image="true",
        save_name="green_frog1.png"
    )
    
    # client action.
    start_time = time.time()
    client(json_request)
    end_time = time.time()
    print("[clinet server] rseponse time: {} seconds".format(round(end_time-start_time,4)))
    