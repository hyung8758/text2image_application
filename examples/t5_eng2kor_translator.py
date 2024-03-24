from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub.hf_api import HfFolder


access_token = "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"
HfFolder.save_token(access_token)
inference_model_name = "QuoQA-NLP/KE-T5-Ko2En-Base" # "QuoQA-NLP/KE-T5-En2Ko-Base"
max_token_length = 64
num_beams = 5
repetition_penalty = 1.3
no_repeat_ngram_size = 3
num_return_sequences = 1

# model_name = "/home/ubuntu/En_to_Ko/ke-t5-base-finetuned-en-to-ko/checkpoint-17850"
model_name = inference_model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)


# https://huggingface.co/datasets/conceptual_captions
# src_text = [
#     "sierra looked stunning in this top and this skirt while performing with person at their former university"
# ]
src_text = [
    "초록색 개구리 한 마리가 연꽃잎 위에 앉았다."
]
translated = model.generate(
    **tokenizer(src_text, return_tensors="pt", padding=True),
    max_length=max_token_length,
    num_beams=num_beams,
    repetition_penalty=repetition_penalty,
    no_repeat_ngram_size=no_repeat_ngram_size,
    num_return_sequences=num_return_sequences,
)
print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])