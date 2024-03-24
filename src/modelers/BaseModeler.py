"""
control pretrained models from diffusers library

Hyungwon Yang
24.01.16
"""

import logging
import torch
import argparse
from typing import Any
from PIL.Image import Image
from abc import ABC
from abc import abstractmethod

class BaseModeler(ABC):
    
    def __init__(self) -> None:
        self.pipe = None
        self.tokenizer = None
        self.image_dict = dict()
    
    def __call__(self, prompt : str) -> Any:
        parser = self.get_parser()
        args = parser.parse_args()
        logging.info("args in modelers: {}".format(args))
        return self.run(prompt=prompt, args=args)
        
    def get_parser(self) -> argparse.ArgumentParser: 

        parser = argparse.ArgumentParser()
        
        group = parser.add_argument_group("Basic configuration")
        group.add_argument(
            "--use_cuda",
            type=bool,
            default=True,
            help="use GPU for inference. If it is set to False then cpu will be utilized.",
        )
        group.add_argument("--cuda_device", type=int, default=0, help="cuda device number.")
        
        # add more configuration if needed.
        self.add_task_arguments(parser)
        return parser
    
    @abstractmethod
    def add_task_arguments(self, parser: argparse.ArgumentParser):
        raise NotImplementedError

    def store_image(self, image_name: str, image: Image) -> None:
        if image_name in self.image_dict.keys():
            raise FileExistsError("{} is already stored.".format(image_name))
        self.image_dict[image_name] = image
        
    def get_image(self, image_name: str) -> Image:
        return self.image_dict[image_name]

    # download or retrieved already donwloaded pretrained model.
    def load_model(self, model: str, use_cuda: bool = True, cuda_device: int = 1):
        logging.info("Loading {} model...".format(model))
        if model == "karlo":
            from diffusers import UnCLIPPipeline
            model_name = "kakaobrain/karlo-v1-alpha"
            self.pipe = UnCLIPPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        elif model == "koala":
            from diffusers import StableDiffusionXLPipeline
            model_name = "etri-vilab/koala-700m-llava-cap"
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        elif model == "ke-t5":
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pipe = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError("Unknown model name: {}".format(model))
        logging.info("Model is loaded successfully.")
        if use_cuda:
             self.pipe = self.pipe.to('cuda:{}'.format(cuda_device))
             logging.info("cuda device set to {}".format(cuda_device))
        else:
            self.pipe = self.pipe.to("cpu")
            logging.info("cpu device is set.")
            
    def save_model(self, model: str, save_path: str):
        pass
    
    @abstractmethod
    def run(self, prompt: str) -> Any:
        raise NotImplementedError