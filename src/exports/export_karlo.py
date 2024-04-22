"""
export_karlo : download karlo model and export it to onnx.
This code will be reorganized later.

Hyungwon Yang
"""

SAMPLING_CONF = {
    "default": {
        "prior_sm": "25",
        "prior_n_samples": 1,
        "prior_cf_scale": 4.0,
        "decoder_sm": "50",
        "decoder_cf_scale": 8.0,
        "sr_sm": "7",
    },
    "fast": {
        "prior_sm": "25",
        "prior_n_samples": 1,
        "prior_cf_scale": 4.0,
        "decoder_sm": "25",
        "decoder_cf_scale": 8.0,
        "sr_sm": "7",
    },
}

CKPT_PATH = {
    "prior": "prior-ckpt-step=01000000-of-01000000.ckpt",
    "decoder": "decoder-ckpt-step=01000000-of-01000000.ckpt",
    "sr_256": "improved-sr-ckpt-step=1.2M.ckpt",
    "clip" : "ViT-L-14.pt",
    "clip_stat" : "ViT-L-14_stats.th"
}

def export_karlo_from_source(cuda_device:int=-1):
    import os, sys
    CURRENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../..")
    sys.path.append(CURRENT_PATH)
    os.chdir(CURRENT_PATH)
    print("current: {}".format(CURRENT_PATH))
    import torch
    import shutil
    import numpy as np
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
    import torchvision.transforms.functional as TVF
    from PIL import Image
    from omegaconf import OmegaConf
    from torchvision.transforms import InterpolationMode
    from src.lib.karlo.karlo.models.prior_model import PriorDiffusionModel
    from src.lib.karlo.karlo.models.decoder_model import Text2ImProgressiveModel
    from src.lib.karlo.karlo.models.sr_64_256 import ImprovedSupRes64to256ProgressiveModel
    from src.lib.karlo.karlo.models.clip import CustomizedCLIP, CustomizedTokenizer


    device="cuda" if cuda_device == -1 else f"cuda:{cuda_device}"
    # save path
    model_path = "karlo_models"
    prior_save_path = "save_prior_model"
    decoder_save_path = "save_decode_model"
    sr_save_path = "save_sr_model"
    model_full_path = os.path.join(CURRENT_PATH, model_path)
    prior_save_full_path = os.path.join(CURRENT_PATH, prior_save_path)
    decoder_save_full_path = os.path.join(CURRENT_PATH, decoder_save_path)
    sr_save_full_path = os.path.join(CURRENT_PATH, sr_save_path)
    if not os.path.exists(model_full_path):
        os.makedirs(model_full_path)
    if not os.path.exists(prior_save_full_path):
        os.makedirs(prior_save_full_path)
    if not os.path.exists(decoder_save_full_path):
        os.makedirs(decoder_save_full_path)
    if not os.path.exists(sr_save_full_path):
        os.makedirs(sr_save_full_path)

    # tokenizer
    tokenizer_model=CustomizedTokenizer()
    # clip
    clip_model = CustomizedCLIP.load_from_checkpoint(
                os.path.join(CURRENT_PATH, model_path, CKPT_PATH['clip']))
    clip_model = torch.jit.script(clip_model)
    clip_model.to(device=device)
    clip_model.eval()
    clip_mean, clip_std = torch.load(
        os.path.join(CURRENT_PATH, model_path, CKPT_PATH['clip_stat']), map_location="cpu")

    # Export prior_model
    print("Export prior model to onnx format.")
    with torch.autocast("cuda", dtype=torch.float16):
        prior_config = OmegaConf.load(os.path.join(model_path, "prior_1B_vit_l.yaml"))
        prior_model = PriorDiffusionModel.load_from_checkpoint(
            prior_config,
            tokenizer_model,
            clip_mean,
            clip_std,
            os.path.join(CURRENT_PATH, model_path, CKPT_PATH['prior']),
            strict=True,
        )
        prior_model.to(device=device)
        prior_model.eval()
        
        txt_feat=torch.randn(2,768).to(device=device)
        txt_feat_seq=torch.randn(2,77,768).to(device=device)
        mask=torch.ones(2,77, dtype=torch.bool).to(device=device)
        prior_cf_scales_batch=torch.tensor([0.4]).to(device=device)
        timestep_respacing="2"
        torch.onnx.export(
            prior_model,
            (txt_feat, txt_feat_seq, mask, prior_cf_scales_batch, timestep_respacing),
            f"{prior_save_full_path}/prior_model.onnx",
            input_names=["txt_feat", "txt_feat_seq", "mask", "prior_cf_scales_batch", "timestep_respacing"],
            output_names=["img_feat"],
            dynamic_axes={
                "txt_feat": {0: "batch"},
                "txt_feat_seq": {0: "batch", 1: "sequence"},
                "mask": {0: "batch"},
                "prior_cf_scales_batch": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    # Export decoder_model
    # NOTE: need to fix source to run the following lines.
    print("Export decoder model to onnx format.")
    with torch.autocast("cuda", dtype=torch.float16):
        # load decoder model.
        decoder_config =OmegaConf.load(os.path.join(model_path, "decoder_900M_vit_l.yaml"))
        decoder_model = Text2ImProgressiveModel.load_from_checkpoint(
            decoder_config,
            tokenizer_model,
            os.path.join(CURRENT_PATH, model_path, CKPT_PATH['decoder']),
            strict=True,
        )
        decoder_model.to(device=device)
        decoder_model.eval()
        
        txt_feat=torch.randn(2,768).to(device=device)
        txt_feat_seq=torch.randn(2,77,768).to(device=device)
        tok=torch.ones(2,77).to(dtype=torch.int32, device=device)
        mask=torch.ones(2,77).to(dtype=torch.bool, device=device)
        img_feat=torch.randn(1,768).to(device=device)
        decoder_cf_scales_batch=torch.tensor([0.4]).to(device=device)
        timestep_respacing="2"
        torch.onnx.export(
            decoder_model,
            (txt_feat, txt_feat_seq, tok, mask, img_feat, decoder_cf_scales_batch, timestep_respacing),
            f"{decoder_save_full_path}/decoder_model.onnx",
            input_names=["txt_feat", "txt_feat_seq", "tok", "mask", "img_feat", "decoder_cf_scales_batch", "timestep_respacing"],
            output_names=["images_64"],
            dynamic_axes={
                "txt_feat": {0: "batch"},
                "txt_feat_seq": {0: "batch", 1: "sequence"},
                "tok": {0: "batch", 1: "sequence"},
                "mask": {0: "batch", 1: "sequence"},
                "img_feat": {0: "batch"},
                "decoder_cf_scales_batch": {0: "batch"},
                "timestep_respacing": {0: "batch"},
                "images_64": {0: "batch", 1: "channel", 2: "height", 3: "width"}
            },
            opset_version=14,
            do_constant_folding=True,
        )

    # Export sr_model
    # NOTE: need to fix source to run the following lines.
    #       This model is too large to export! so it generates multiple onnx files instead of a single file.
    print("Export super resolution model to onnx format.")
    with torch.autocast("cuda", dtype=torch.float16):
        # load super resolution model.
        sr_config = OmegaConf.load(os.path.join(model_path, "improved_sr_64_256_1.4B.yaml"))
        sr_model = ImprovedSupRes64to256ProgressiveModel.load_from_checkpoint(
            sr_config, 
            os.path.join(CURRENT_PATH, model_path, CKPT_PATH['sr_256']), strict=True
        )
        sr_model.to(device=device)
        sr_model.eval()
        sr_save_model_path = f"{sr_save_full_path}/sr_model.onnx"
        sr_save_external_path = f"{sr_save_full_path}/sr_external_model.onnx"
        
        images_256=torch.randn(1, 3, 256, 256).to(device=device)
        sr_sm="7"
        torch.onnx.export(
            sr_model,
            (images_256, sr_sm),
            sr_save_model_path,
            input_names=["images_256", "sr_sm"],
            output_names=["generated_image"],
            dynamic_axes={
                "images_256": {0: "batch", 1: "channel", 2: "height", 3: "width"},
                "generated_image": {0: "batch", 1: "channel", 2: "height", 3: "width"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        
        # Once it saved into multiple files, read the main onnx file again and save a large part of networks to an external data type.
        # Onnx files will be saved into two split files.
        print("Combine the large split onnx files into an external data type")
        sr_onnx_model = onnx.load(sr_save_model_path)
        # remove previous onnx file.
        if os.path.exists(sr_save_full_path):
            shutil.rmtree(sr_save_full_path)
            os.makedirs(sr_save_full_path)
        convert_model_to_external_data(sr_onnx_model, all_tensors_to_one_file=True, location=sr_save_external_path, size_threshold=1024, convert_attribute=False)
        onnx.save_model(sr_onnx_model, sr_save_model_path)


def export_karlo_from_library(cuda_device:int=-1):
    print("Not yet implemented. Use source type.")
    exit(1)
    import torch
    from diffusers import UnCLIPPipeline

    karlo_image_generator = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float32)
    # pipe.save_pretrained("./karlo_models", from_pt=True)
    # karlo_image_generator = karlo_image_generator.to('cuda:1')
    # initialization
    karlo_tokenizer = karlo_image_generator.tokenizer
    karlo_text_encoder = karlo_image_generator.text_encoder
    karlo_text_proj = karlo_image_generator.text_proj

    karlo_prior = karlo_image_generator.prior
    karlo_decoder = karlo_image_generator.decoder

    karlo_sr_fist = karlo_image_generator.super_res_first
    karlo_sr_last = karlo_image_generator.super_res_last

    prompt = "a high-resolution photograph of a big red frog on a green leaf."
    prior_num_inference_steps=25
    decoder_num_inference_steps=25
    super_res_num_inference_steps=7

    torch.onnx.export(
        karlo_image_generator,
        (prompt, prior_num_inference_steps, decoder_num_inference_steps, super_res_num_inference_steps),
        "karlo_image_generator.onnx",
        input_names=["prompt", "prior_num_inference_steps", "decoder_num_inference_steps", "super_res_num_inference_steps"],
        output_names=["generated_image"],
        opset_version=12,  # Use a suitable ONNX opset version
        dynamic_axes={"prompt": {0: "batch", 1: "sequence"}},  # Specify dynamic axes if needed
        do_constant_folding=True,  # Perform constant folding
    )
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
        
    group = parser.add_argument_group("Basic configuration")
    group.add_argument("--type","-t", type=str, default="source", help="Onnx export type: source or library")
    group.add_argument("--cuda_device","-d", type=int, default=-1, help="Use specific cuda device. -1 means all. e.g., set 0 to use 1st gpu device.")
    args = parser.parse_args()
    
    print("export type: {}".format(args.type))
    if args.type == "source":
        export_karlo_from_source(cuda_device=args.cuda_device)
    elif args.type == "library":
        export_karlo_from_library(cuda_device=args.cuda_device)
    else:
        raise RuntimeError("Unknown arguments: {}".format(args.type))