"""
ImageHanlder: image generator.

Hyungwon Yang
24.03.18
"""
from typing import Any
from src.handlers.MainHandler import MainHandler
from src.modelers.KarloModeler import KarloModeler

class ImageHandler(MainHandler):
    
    def __init__(self) -> None:
        super().__init__()
        
    def initialize(self, model: str = 'karlo',  use_cuda: bool = True, cuda_device: int = 0, *args, **kwargs) -> None:
        super().initialize(*args, **kwargs)
        self.karloModeler = KarloModeler()
        self.karloModeler.load_model(model, use_cuda, cuda_device)
    
    def pre_processor(self, input_value: Any) -> Any:
        output_value = input_value
        return output_value
    
    def post_processor(self, input_value: Any) -> Any:
        output_value = input_value
        return output_value
    
    def inference(self, input_value: Any, *args, **kwargs) -> Any:
        return self.karloModeler.run(prompt=input_value, save_image=kwargs['save_image'], save_name=kwargs['save_name'])
        