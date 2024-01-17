"""
kakaobrain karlo models from diffusers library
This model generates variouse images based on a given text.

Hyungwon Yang
24.01.16
"""

import argparse
import torch
from PIL.Image import Image
from src.diffuser import BaseDiffuser
from diffusers import UnCLIPPipeline

class KarloDiffuser(BaseDiffuser):
    
    def __init__(self) -> None:
        self.pipe = None
        self.imageDict = dict()
    
    def add_task_arguments(self, parser: argparse.ArgumentParser) -> None:
    
        group = parser.add_argument_group("kalro configuration")
        group.add_argument(
            "--use_cuda",
            type=bool,
            default=True,
            help="use GPU for inference. ",
        )
        group.add_argument("--cuda_device", type=int, default=0, help="cuda device number.")
        group.add_argument("--pretrain_model", type=str, default="kakaobrain/karlo-v1-alpha", help="pretrained model name")
        group.add_argument("--prompt", type=str, help="English sentence for generating an image.", required=True)
    
    def getModel(self, pretrain_model: str, cuda_device: int = 0):
        self.pipe = UnCLIPPipeline.from_pretrained(pretrain_model, torch_dtype=torch.float16)
        self.pipe = self.pipe.to('cuda:{}'.format(cuda_device))
        
    def imageStack(self, imageName: str, image: Image) -> None:
        if imageName in self.imageDict.keys():
            raise FileExistsError("{} is already saved.".format(imageName))
        self.imageDict[imageName] = image
        
    def getImage(self, imageName: str) -> Image:
        return self.imageDict[imageName]
        
    def run(self, args: argparse) -> None:
        
        prompt = args.prompt
        # set model.
        if self.pipe == None:
            self.getModel(args.pretrain_model)
            
        image = self.pipe(prompt).images[0]
        
        image.save("./school_boy.png")