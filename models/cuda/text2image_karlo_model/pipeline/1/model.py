"""
triton server base model: text2image 

Hyungwon Yang
24.03.29
MeidaZen
"""

import json
import time
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from tqdm.auto import tqdm
from torch.utils.dlpack import from_dlpack, to_dlpack
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        self.prior_num_inference_steps = 25
        self.decoder_num_inference_steps = 10
        self.super_res_num_inference_steps = 7
        
        print("init trtion server")
        start_time = time.time()
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "generated_image"
            )["data_type"]
        )
        # translator
        translator_model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
        self.tokenizer = AutoTokenizer.from_pretrained(translator_model_name)
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_name, 
                                                                      torch_dtype=torch.float16
                                                                      ).to("cuda")
        print("loading translator model DONE")
        # karlo image
        image_model_name = "kakaobrain/karlo-v1-alpha"
        self.image_model = UnCLIPPipeline.from_pretrained(image_model_name, 
                                                          torch_dtype=torch.float16
                                                          ).to("cuda")
        end_time = time.time()
        print("loading image generator model DONE: {} seconds".format(round(end_time-start_time,2)))
    
    def execute(self, requests):
        responses = []
        print("start execution")
        for request in requests:
            input_prompt = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_text = input_prompt.as_numpy()[0][0].decode('utf-8')
            print("given input: {}".format(input_text))
            t_start_time = time.time()
            ## 1st. translation
            # tokenizing
            tokenized_text = self.tokenizer(
                [input_text],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_text = tokenized_text.to("cuda")
            # print("tokenized text: {}".format(tokenized_text))
            # translate english text to korean.
            translated = self.translator_model.generate(
                **tokenized_text,
                max_length=self.max_token_length,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                num_return_sequences=self.num_return_sequences,
            )
            t_end_time = time.time()
            print('translation duration: {} seconds'.format(round(t_end_time-t_start_time,2)))
            output_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            print("output_text: {}".format(output_text))
            ## 2nd. text to image generation.
            i_start_time = time.time()
            deocded_image = self.image_model(output_text,
                                            prior_num_inference_steps=self.prior_num_inference_steps,
                                            decoder_num_inference_steps=self.decoder_num_inference_steps,
                                            super_res_num_inference_steps=self.super_res_num_inference_steps,
                                            ).images[0]
            # return results
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(deocded_image, dtype=self.output_dtype),
                    )
                ]
            )
            i_end_time = time.time()
            print('image generation duration: {} seconds'.format(round(i_end_time-i_start_time,2)))
            responses.append(inference_response)
            print("response: {}".format(responses))
        return responses