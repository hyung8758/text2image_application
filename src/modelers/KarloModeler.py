"""
kakaobrain karlo models from diffusers library
This model generates variouse images based on a given text.

Hyungwon Yang
24.01.16
"""
import argparse
import logging
from typing import Any
from src.modelers.BaseModeler import BaseModeler

class KarloModeler(BaseModeler):
    
    def __init__(self) -> None:
        super().__init__()
    
    def add_task_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("karlo configuration")
        group.add_argument("--model", type=str, default="karlo", choices=['karlo', 'koala'], help="model name.")
        group.add_argument("--save_image", type=bool, default=False, help="save image or not.")
        group.add_argument("--save_name", type=str, default="karlo_result.png", help="image name for saving.")
        # group.add_argument("--pretrain_model", type=str, default="kakaobrain/karlo-v1-alpha", help="pretrained model name")
        group.add_argument("--prompt", type=str, help="English sentence for generating an image.", required=True)
        
    def run(self, prompt: str, model: str = 'karlo', args: argparse.Namespace = None) -> Any:
        
        # set model.
        if self.pipe == None:
            self.load_model(model)
        
        output_image = self.pipe(prompt).images[0]
        logging.info("karlo output image: {}".format(output_image))
        if args.save_image:
            logging.info("save image as {}".format(args.save_name))
            output_image.save(args.save_name)
        return output_image
        # image.save(save_name)