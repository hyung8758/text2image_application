"""
client.py: test client.

Hyungwon Yang
24.03.19
MediaZen
"""

# restful clinet 
import io
import json
import time
import base64
import argparse
import http.client
from PIL import Image
from typing import Any

class postRequestClient:
    def __init__(self, server_addr: str="localhost", server_port: int=33010) -> None:
        self.conn = http.client.HTTPConnection(server_addr, server_port)
    
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
            headers = {"Content-type": "application/json"}
            self.conn.request("POST", "/text2image", body=json.dumps(json_data), headers=headers)
            
            # get a response
            response = self.conn.getresponse()
            json_result = json.loads(response.read().decode('utf-8'))
            # print("json result: {}".format(json_result))
            if json_result['success']:
                encoded_image = json_result['image']
                save_path = "result_image.jpg"
                self.save_image(encoded_image, save_path)
        except Exception as e:
            print("error: {}".format(e))
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
                        default=33010,
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
    