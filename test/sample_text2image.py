import os
import torch
import numpy as np
import torch.utils
import torch.utils.dlpack
from transformers import AutoTokenizer
from onnxruntime import InferenceSession, GraphOptimizationLevel, ExecutionMode, SessionOptions
from fastT5 import OnnxT5
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)

class CustomOnnxT5(OnnxT5):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:

            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )

            logits, past_key_values = init_onnx_outputs
        else:

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )
            logits, past_key_values = onnx_outputs
        logits = logits.to("cuda:{}".format(input_ids.device if input_ids else decoder_input_ids.device))
        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)
    
def load_onnx_model(model_name: str, model_path: str, quantized: bool=True) -> InferenceSession:
        # find models.
        onnx_models = os.listdir(model_path)
        model_head = model_name.split("/")[-1]
        encoder_path, decoder_path, init_decoder_path = "", "", ""
        print(f"onnx models: {onnx_models}")
        for each_path in onnx_models:
            if quantized:
                if each_path in f"{model_head}-encoder-quantized.onnx":
                    encoder_path = os.path.join(model_path, each_path)
                elif each_path  in f"{model_head}-decoder-quantized.onnx":
                    decoder_path = os.path.join(model_path, each_path)
                elif each_path in f"{model_head}-init-decoder-quantized.onnx":
                    init_decoder_path = os.path.join(model_path, each_path)
                else:
                    print(f"Unrecognized files: {each_path}")
            else:
                if each_path in "Base-encoder.onnx":
                    encoder_path = os.path.join(model_path, each_path)
                elif each_path  in "Base-decoder.onnx":
                    decoder_path = os.path.join(model_path, each_path)
                elif each_path in "Base-init-decoder.onnx":
                    init_decoder_path = os.path.join(model_path, each_path)
                else:
                    print(f"Unrecognized files: {each_path}")
        
        if not (encoder_path and decoder_path and init_decoder_path):
            raise RuntimeError(f"Cannot find onnx path. {encoder_path}/{decoder_path}/{init_decoder_path}")
        print(f"found model path: {encoder_path} , {decoder_path} , {init_decoder_path}")
        # set options
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ExecutionMode.ORT_PARALLEL
        providers = [("CUDAExecutionProvider", {"device_id": cuda_device})]
        # providers = ["CPUExecutionProvider"]
        # initiate inference sessions.
        encoder_sess = InferenceSession(
            str(encoder_path), options, providers=providers
        )
        decoder_sess = InferenceSession(
            str(decoder_path), options, providers=providers
        )
        init_decoder_sess = InferenceSession(
            str(init_decoder_path), options, providers=providers
        )
        
        model_sessions = encoder_sess, decoder_sess, init_decoder_sess
        
        return CustomOnnxT5(model_name, model_sessions)
    

total_text = ['안녕하세요']
cuda_device=0
translator_model_path = "models/quantized_translator_model" # "/workspace/models/quantized_translator_model"
translator_model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
tokenizer = AutoTokenizer.from_pretrained(translator_model_name)
# self.translator_model = self.get_onnx_model(translator_model_name, translator_model_path, quantized=True)
translator_model = load_onnx_model(translator_model_name, translator_model_path, quantized=True).to(f"cuda:{cuda_device}")

# parameters
num_beams = 5
repetition_penalty = 1.3
no_repeat_ngram_size = 3
num_return_sequences = 1


    
tokenized_text = tokenizer(
            total_text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(f"cuda:{cuda_device}")

input_ids_on_gpu = torch.utils.dlpack.from_dlpack(tokenized_text.input_ids)
attention_mask_on_gpu = torch.utils.dlpack.from_dlpack(tokenized_text.attention_mask)

predicted_tokens = translator_model.generate(input_ids=input_ids_on_gpu,
                    attention_mask=attention_mask_on_gpu,
                    max_length=tokenizer.model_max_length,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    num_return_sequences=num_return_sequences) 