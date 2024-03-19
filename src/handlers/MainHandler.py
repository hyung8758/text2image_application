"""
MainHandler: base handler for text to image process. It connects 2 modules: translator and text2image.

Hyungwon Yang
24.03.18
"""
from abc import ABC
from abc import abstractmethod
from typing import Any

class MainHandler(ABC):
    
    def __init__(self) -> None:
        pass    
    
    def run(self, input_value: str) -> Any:
        pre_processed_value = self.pre_processor(input_value)
        output_value = self.inference(pre_processed_value)
        post_processed_value = self.post_processor(output_value)
        return post_processed_value
        
    def initialize(self, *args, **kwargs) -> None:
        # initialize values if needed.
        pass
    
    @abstractmethod
    def pre_processor(self, input_value: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def post_processor(self, input_value: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def inference(self, input_value: Any) -> Any:
        raise NotImplementedError
    
    