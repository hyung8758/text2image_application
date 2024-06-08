# test for text2karloIamgeGpuClinet.py
# export PYTHONIOENCODING=UTF-8
# tritonserver --model-repository=/models

"""gpu 서버 triton version 자기
## original
    docker run --gpus '"device=1,2,3"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/models/cuda/text2image_karlo_model_onnx2:/models nvcr.io/nvidia/tritonserver:24.01-py3 bash

    # 3. Once the container is started, install the following libraries
    pip install torch torchvision torchaudio
    pip install transformers==4.41.1
    pip install ftfy scipy accelerate progress sentencepiece
    pip install diffusers==0.27.2
    pip install onnxruntime-gpu=1.18.0
    pip install fastt5==0.1.4 --no-deps
    pip install transformers[onnxruntime]
    
    tritonserver --model-repository=/models --log-verbose=1

## onnxruntime_gpu 새롭게 세팅한 버전.
    docker run --gpus '"device=1,2,3"' -it --shm-size=2g --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/models/cuda/text2image_karlo_model_onnx2:/models nvcr.io/nvidia/tritonserver:22.12-py3 bash

    # 3. Once the container is started, install the following libraries
    pip install torch torchvision torchaudio
    pip install transformers==4.41.1
    pip install ftfy scipy accelerate progress sentencepiece
    pip install diffusers==0.27.2
    pip install onnxruntime-gpu==1.18.0
    pip install fastt5==0.1.4 --no-deps

    tritonserver --model-repository=/models --log-verbose=1


"""

import time
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype

triton_url = "localhost:8001"
client = grpcclient.InferenceServerClient(url=triton_url)

prompt = "초록색 개구리 한 마리가 나뭇잎 위에 앉아 있다."
text_obj = np.array(prompt, dtype="object").reshape((-1, 1))
start_time = time.time()
# print("[client] start infer: {}".format(start_time))
print("input: {}".format(text_obj))
input_text = grpcclient.InferInput(
    "input_text", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
)
# print(f"input_text: {input_text}")
input_text.set_data_from_numpy(text_obj)
output_img = grpcclient.InferRequestedOutput("generated_image")
# print(f"output_img:{output_img}")
response = client.infer(
    model_name="ensemble_model", inputs=[input_text], outputs=[output_img]
)
end_time = time.time()
# print("[client] end infer: {}".format(end_time))
print("[client] total duration: {}".format(round(end_time-start_time, 2)))
resp_img = response.as_numpy("generated_image")
# print(resp_img.shape)
im = Image.fromarray(np.squeeze(resp_img.astype(np.uint8)))
im.save("sample_triton_image.png")

# ### quantinzed 모델 성능 비교
# import time
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from fastT5 import export_and_get_onnx_model, get_onnx_model

# print ("Import models.")
# device="cuda:1"
# quantized=True
# translator_model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
# onnx_save_path = "./save_translator_onnx_model"
# tokenizer = AutoTokenizer.from_pretrained(translator_model_name)

# # import onnx model.
# qt_model = get_onnx_model(translator_model_name, onnx_save_path, quantized=True)
# t_model = get_onnx_model(translator_model_name, onnx_save_path, quantized=False)
# qt_model.to("cuda:1")
# t_model.to("cuda:2")
# # inference with onnx
# print("do inference")
# prompt = "교육부와 보건복지부가 관련 논의에 들어간 가운데 대통령실도 조속히 결론을 내려 현장 혼란을 최소화해야 한다는 입장이다."
# max_token_length = 64
# num_beams = 5
# repetition_penalty = 1.3
# no_repeat_ngram_size = 3
# num_return_sequences = 1

# qt_start_time = time.time()
# qt_input_text = tokenizer(prompt, return_tensors='pt')
# qt_input_text.to("cuda:1")
# qt_tokens = qt_model.generate(input_ids=qt_input_text.input_ids,
#                attention_mask=qt_input_text.attention_mask,
#                max_length=max_token_length,
#                num_beams=num_beams,
#                repetition_penalty=repetition_penalty,
#                no_repeat_ngram_size=no_repeat_ngram_size,
#                num_return_sequences=num_return_sequences)
# tokenizer.decode(qt_tokens.squeeze(), skip_special_tokens=True)
# qt_end_time = time.time()
# print('qunatized model duration: {} seconds'.format(round(qt_end_time-qt_start_time,2)))

# t_start_time = time.time()
# t_input_text = tokenizer(prompt, return_tensors='pt')
# t_input_text.to("cuda:2")
# t_tokens = t_model.generate(input_ids=t_input_text.input_ids,
#                attention_mask=t_input_text.attention_mask,
#                max_length=max_token_length,
#                num_beams=num_beams,
#                repetition_penalty=repetition_penalty,
#                no_repeat_ngram_size=no_repeat_ngram_size,
#                num_return_sequences=num_return_sequences)
# tokenizer.decode(t_tokens.squeeze(), skip_special_tokens=True)
# t_end_time = time.time()
# print('normal model duration: {} seconds'.format(round(t_end_time-t_start_time,2)))

# ###
# import numpy as np
# import torch
# import onnxruntime as ort

# txt_feat=np.random.randn(2,768).astype(np.float32)
# txt_feat_seq=np.random.randn(2,77,768).astype(np.float32)
# mask=np.ones((2,77)).astype(np.bool_)
# img_feat=np.random.randn(1,768).astype(np.float32)
# decoder_cf_scales_batch=np.array([0.4]).astype(np.float32)

# device="cuda:1"
# txt_feat=torch.randn(2,768).to(device=device)
# txt_feat_seq=torch.randn(2,77,768).to(device=device)
# mask=torch.ones(2,77).to(dtype=torch.bool, device=device)
# img_feat=torch.randn(1,768).to(device=device)
# decoder_cf_scales_batch=torch.tensor([0.4]).to(device=device)

# ort_sess = ort.InferenceSession('decoder/1/model.onnx')
# outputs = ort_sess.run(None, {'txt_feat':txt_feat, 'txt_feat_seq':txt_feat_seq, 'mask':mask, 'img_feat':img_feat, 'decoder_cf_scales_batch':decoder_cf_scales_batch})