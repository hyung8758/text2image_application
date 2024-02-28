"""
kakaobrain karlo models from diffusers library
This model generates variouse images based on a given text.

Hyungwon Yang
24.01.16
"""

import argparse
from PIL.Image import Image
from src.imageDiffuser import BaseDiffuser

class KarloDiffuser(BaseDiffuser):
    
    def __init__(self) -> None:
        pass
    
    def add_task_arguments(self, parser: argparse.ArgumentParser) -> None:
        # group = parser.add_argument_group("kalro configuration")
        pass
        
    def run(self, input_prompt: str, save_name: str, model: str = 'karlo') -> None:
        
        prompt = input_prompt
        # set model.
        if self.pipe == None:
            self.load_model(model)
            
        image = self.pipe(prompt).images[0]
        
        image.save(save_name)