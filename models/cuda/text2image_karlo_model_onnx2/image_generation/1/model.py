"""
triton server text2image model. 

Hyungwon Yang
24.05.22
MeidaZen
"""

import json
import time
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from tqdm.auto import tqdm
# from torch.utils.dlpack import from_dlpack, to_dlpack
from diffusers import UnCLIPPipeline


class TritonPythonModel:
    def initialize(self, args):
        # translator parameters.
        self.max_token_length = 64
        self.num_beams = 5
        self.repetition_penalty = 1.3
        self.no_repeat_ngram_size = 3
        self.num_return_sequences = 1
        # image generation parameters.
        self.prior_num_inference_steps = 25 # 25
        self.decoder_num_inference_steps = 25 # 25
        self.super_res_num_inference_steps = 7 # 7
        
        model_config = json.loads(args["model_config"])
        cuda_device = json.loads(args["model_instance_device_id"])
        print("cuda device: {}".format(cuda_device))
        print("in image_generation model config: {}".format(model_config))
        # Get output text configuration
        output_generated_image_config = pb_utils.get_output_config_by_name(
            model_config, "output_generated_image"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_generated_image_config["data_type"]
        )
        
        # karlo image
        print("Loading image generator model.")
        image_model_name = "kakaobrain/karlo-v1-alpha"
        self.image_model = UnCLIPPipeline.from_pretrained(image_model_name, 
                                                          torch_dtype=torch.float16
                                                          ).to(f"cuda:{cuda_device}")
        print("Done.")
    
    def execute(self, requests):
        responses = []
        input_text = []
        for request in requests:
            input_prompt = pb_utils.get_input_tensor_by_name(request, "input_image_text")
            # print("[image] each request number: {}".format(len(input_prompt.as_numpy())))
            input_text.append(input_prompt.as_numpy()[0][0].decode('utf-8'))
        # input_text = [pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0][0].decode('utf-8') for request in requests]
        print("[{}] given input: {}".format(len(input_text), input_text))
        i_start_time = time.time()
        decoded_image = self.image_model(input_text,
                                        prior_num_inference_steps=self.prior_num_inference_steps,
                                        decoder_num_inference_steps=self.decoder_num_inference_steps,
                                        super_res_num_inference_steps=self.super_res_num_inference_steps,
                                        ).images
        # return results
        print("images: {}".format(decoded_image))
        for each_decoded_image in decoded_image:
            print("each image: {}".format(each_decoded_image))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "output_generated_image",
                        np.array(each_decoded_image, dtype=self.output_dtype),
                    )
                ]
            )
            responses.append(inference_response)
        i_end_time = time.time()
        print('image generation duration: {} seconds'.format(round(i_end_time-i_start_time,2)))
        return responses
    