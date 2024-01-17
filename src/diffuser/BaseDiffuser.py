"""
control pretrained models from diffusers library

Hyungwon Yang
24.01.16
"""

import logging
import argparse
from abc import ABC
from abc import abstractmethod

class baseDiffuser(ABC):
    
    def __init__(self) -> None:
        pass
    
    def __call__(self):
        parser = self.get_parser()
        args = parser.parse_args()
        
        
    def get_parser(self) -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser()
        
        # add more configuration if needed.
        self.add_task_arguments(parser)
        return parser
    
    @abstractmethod
    def add_task_arguments(self, parser: argparse.ArgumentParser):
        raise NotImplementedError
    
    @abstractmethod
    # download or retrieved already donwloaded pretrained model.
    def getModel(self):
        raise NotImplementedError
    
    @abstractmethod
    def run(self):
        raise NotImplementedError