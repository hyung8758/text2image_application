"""
text_translation

Hyungwon Yang
"""

import os
import json
import time
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

import triton_python_backend_utils as pb_utils


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
        logits = logits.to("{}".format(input_ids.device if input_ids else decoder_input_ids.device))
        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

class TritonPythonModel:

    def initialize(self, args):
        """
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args["model_config"])
        self.cuda_device = json.loads(args["model_instance_device_id"])
        print("cuda device: {}".format(self.cuda_device))
        print("in text_translation model config: {}".format(model_config))
        # Get output text configuration
        translated_prompt_config = pb_utils.get_output_config_by_name(
            model_config, "translated_prompt"
        )

        # Convert Triton types to numpy types
        self.translated_prompt_dtype = pb_utils.triton_string_to_numpy(
            translated_prompt_config["data_type"]
        )
        
        # Load translator model.
        # NOTE: preparation: download a translator model and save it by quantized.
        translator_model_path = "/workspace/models/quantized_translator_model"
        translator_model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
        self.tokenizer = AutoTokenizer.from_pretrained(translator_model_name)
        # self.translator_model = self.get_onnx_model(translator_model_name, translator_model_path, quantized=True)
        self.translator_model = self.load_onnx_model(translator_model_name, translator_model_path, quantized=True).to(f"cuda:{self.cuda_device}")

        # parameters
        self.num_beams = 5
        self.repetition_penalty = 1.3
        self.no_repeat_ngram_size = 3
        self.num_return_sequences = 1

    def load_onnx_model(self, model_name: str, model_path: str, quantized: bool=True) -> InferenceSession:
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
        providers = [("CUDAExecutionProvider", {"device_id": self.cuda_device})]
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
        
    def preprocess_text_input(self, input_text: str, lang: str = 'kor') -> str:
            if lang == "kor":
                pass
            elif lang == "eng":
                input_text += " a high-resolution photograph"
            else:
                raise ValueError("Unknown lang option: {}".format(lang))
            
            return input_text
        
    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        total_text = []
        for request in requests:
            input_prompt = pb_utils.get_input_tensor_by_name(
                request, "input_prompt"
            )
            # print("[translator] each request number: {}".format(len(input_prompt.as_numpy())))
            decoded_text = input_prompt.as_numpy()[0][0].decode('utf-8')
            # preprocess korean input text
            # pb_utils.Logger.log_info("in text translation, decoded text : {}".format(decoded_text))
            total_text.append(self.preprocess_text_input(input_text=decoded_text, lang='kor'))
        pb_utils.Logger.log_info("[{}] given input: {}".format(len(total_text), total_text))
        i_start_time = time.time()
        # translation
        # pb_utils.Logger.log_info("total text: {}".format(total_text))
        tokenized_text = self.tokenizer(
            total_text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(f"cuda:{self.cuda_device}")
        input_ids_on_gpu = torch.utils.dlpack.from_dlpack(tokenized_text.input_ids)
        attention_mask_on_gpu = torch.utils.dlpack.from_dlpack(tokenized_text.attention_mask)
        # pb_utils.Logger.log_info(f"Tokenized text input_ids device: {input_ids_on_gpu.device}")
        # pb_utils.Logger.log_info(f"Tokenized text attention_mask device: {attention_mask_on_gpu.device}")

        with torch.no_grad():
            predicted_tokens = self.translator_model.generate(input_ids=input_ids_on_gpu,
                        attention_mask=attention_mask_on_gpu,
                        max_length=self.tokenizer.model_max_length,
                        num_beams=self.num_beams,
                        repetition_penalty=self.repetition_penalty,
                        no_repeat_ngram_size=self.no_repeat_ngram_size,
                        num_return_sequences=self.num_return_sequences)


        for tokens in predicted_tokens:
            translated_text = self.tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
            
            translated_text = self.preprocess_text_input(input_text=translated_text, lang='eng')
            translated_tensor = pb_utils.Tensor(
                "translated_prompt", np.array(translated_text.encode('utf-8'), dtype="object").reshape((-1, 1))
            )
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[translated_tensor]
            )
            responses.append(inference_response)
        i_end_time = time.time()
        print('eng -> kor translation duration: {} seconds'.format(round(i_end_time-i_start_time,2)))
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")