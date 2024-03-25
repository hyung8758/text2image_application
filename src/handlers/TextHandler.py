"""
TextHanlder: translator.

Hyungwon Yang
24.03.18
"""
from typing import Any
from src.handlers.MainHandler import MainHandler
from src.modelers.TextModeler import TextModeler  

class TextHandler(MainHandler):
    
    def __init__(self) -> None:
        super().__init__()
        
    def initialize(self, model: str = 'ke-t5',  use_cuda: bool = True, cuda_device: int = 0, *args, **kwargs) -> None:
        super().initialize(*args, **kwargs)
        self.textModeler = TextModeler()
        self.textModeler.load_model(model, use_cuda, cuda_device)
    
    def pre_processor(self, input_value: Any) -> Any:
        output_value = input_value
        return output_value
    
    def post_processor(self, input_value: Any) -> Any:
        output_value = input_value
        return output_value
    
    def inference(self, input_value: Any, *args, **kwargs) -> Any:
        return self.textModeler.run(prompt=input_value)