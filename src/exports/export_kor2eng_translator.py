"""
export_kor2eng_translator : download kor2eng t5 translator model and export it to onnx.
reference: https://github.com/Ki6an/fastT5/tree/master

Hyungwon Yang
"""
import os
from progress.bar import Bar
from transformers import AutoTokenizer
from fastT5 import onnx_exporter, export_and_get_onnx_model


### 함수 재정의
def modified_quantize(models_name_or_path):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    bar = Bar("Quantizing...", max=3)

    quant_model_paths = []
    for model in models_name_or_path:
        model_name = model.as_posix()
        output_model_name = f"{model_name[:-5]}-quantized.onnx"
        quantize_dynamic(
            model_input=model_name,
            model_output=output_model_name,
            per_channel=True,
            reduce_range=True, # should be the same as per_channel
            # activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
            # optimize_model=False,
        )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul' ],
        quant_model_paths.append(output_model_name)
        bar.next()

    bar.finish()

    return tuple(quant_model_paths)

#####

# device="cuda:1"
# quantized=True
translator_model_name = "QuoQA-NLP/KE-T5-Ko2En-Base"
onnx_save_path = "models/quantized_translator_model"
tokenizer = AutoTokenizer.from_pretrained(translator_model_name)

# export and quantize translator model.
# Step 1. convert huggingfaces t5 model to onnx
onnx_model_paths = onnx_exporter.generate_onnx_representation(
    translator_model_name, output_path=onnx_save_path
)

# Step 2. (recommended) quantize the converted model for fast inference and to reduce model size.
onnx_exporter.quantize = modified_quantize
quant_model_paths = onnx_exporter.quantize(onnx_model_paths)
# t_model = export_and_get_onnx_model(translator_model_name, onnx_save_path, quantized=quantized)

# Step 3. remove original models.
file_list = os.listdir(onnx_save_path)
for each_f in file_list:
    if "quantized" not in each_f:
        os.remove(os.path.join(onnx_save_path, each_f))
print("DONE")