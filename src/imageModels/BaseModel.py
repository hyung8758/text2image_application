"""
control pretrained models from diffusers library

Hyungwon Yang
24.01.16
"""

import logging
import torch
import argparse
from PIL.Image import Image
from abc import ABC
from abc import abstractmethod

class BaseDiffuser(ABC):
    
    def __init__(self) -> None:
        self.pipe = None
        self.image_dict = dict()
    
    def __call__(self):
        parser = self.get_parser()
        args = parser.parse_args()
        
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
        # group.add_argument("--pretrain_model", type=str, default="kakaobrain/karlo-v1-alpha", help="pretrained model name")
        group.add_argument("--pretrain_model", type=str, default="kalro", choices=['kalro', 'koala'], help="pretrained model name.")
        group.add_argument("--prompt", type=str, help="English sentence for generating an image.", required=True)
        
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
    def load_model(self, pretrain_model: str, use_cuda: bool = True, cuda_device: int = 0):
        logging.info("Loading {} model...".format(pretrain_model))
        if pretrain_model == "kalro":
            from diffusers import UnCLIPPipeline
            self.pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
        elif pretrain_model == "koala":
            from diffusers import StableDiffusionXLPipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained("etri-vilab/koala-700m-llava-cap", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        else:
            raise ValueError("Unknown pretained model name: {}".format(pretrain_model))
        logging.info("Model is loaded successfully.")
        if use_cuda:
             self.pipe = self.pipe.to('cuda:{}'.format(cuda_device))
             logging.info("cuda device set to {}".format(cuda_device))
        else:
            self.pipe = self.pipe.to("cpu")
            logging.info("cpu device is set.")
            
    def save_model(self, pretrain_model: str, save_path: str):
        pass
    
    @abstractmethod
    def run(self):
        raise NotImplementedError