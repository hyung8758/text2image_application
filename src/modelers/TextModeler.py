"""
text modeler.
    - translator

Hyungwon Yang
24.03.18
"""
import argparse
import logging
from typing import Any
from src.modelers.BaseModeler import BaseModeler

class TextModeler(BaseModeler):
    
    def __init__(self) -> None:
        super().__init__()
        self.max_token_length = 64
        self.num_beams = 5
        self.repetition_penalty = 1.3
        self.no_repeat_ngram_size = 3
        self.num_return_sequences = 1
    
    def add_task_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Translator configuration")
        group.add_argument("--model", type=str, default="ke-t5", choices=['ke-t5'], help="model name.")
        
    def run(self, prompt: str, model: str = 'ke-t5', args: argparse.Namespace = None) -> Any:
        
        logging.info("kor2eng Translator input text: {}".format(prompt))
        # set model.
        if self.tokenizer == None or self.pipe == None:
            self.load_model(model)
            
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True)
        # translate english text to korean.
        translated = self.pipe.generate(
            **inputs,
            max_length=self.max_token_length,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_return_sequences=self.num_return_sequences,
        )
        
        # logging.info("translated token: {}".format(translated))
        output_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        logging.info("kor2eng Translator output text: {}".format(output_text))
        return output_text