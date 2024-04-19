"""
export_kor2eng_translator : download kor2eng t5 translator model and export it to onnx.
reference: https://github.com/Ki6an/fastT5/tree/master

Hyungwon Yang
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastT5 import export_and_get_onnx_model, get_onnx_model

print ("Import models.")
device="cuda:1"
quantized=True
translator_model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
onnx_save_path = "./save_translator_onnx_model"
tokenizer = AutoTokenizer.from_pretrained(translator_model_name)

# export and quantize translator model.
t_model = export_and_get_onnx_model(translator_model_name, onnx_save_path, quantized=quantized)

# import onnx model.
translator_model = get_onnx_model(translator_model_name, onnx_save_path, quantized=quantized)

# inference with onnx
prompt = "안녕하세요."
max_token_length = 64
num_beams = 5
repetition_penalty = 1.3
no_repeat_ngram_size = 3
num_return_sequences = 1
input_text = tokenizer(prompt, return_tensors='pt')
tokens = translator_model.generate(input_ids=input_text.input_ids,
               attention_mask=input_text.attention_mask,
               max_length=max_token_length,
               num_beams=num_beams,
               repetition_penalty=repetition_penalty,
               no_repeat_ngram_size=no_repeat_ngram_size,
               num_return_sequences=num_return_sequences)

output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)