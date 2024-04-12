"""
export_karlo : download karlo model and export it to onnx.

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

def export_karlo_from_source():
    import os, sys
    CURRENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../..")
    sys.path.append(CURRENT_PATH)
    os.chdir(CURRENT_PATH)
    print("current: {}".format(CURRENT_PATH))
    import torch
    import numpy as np
    import torchvision.transforms.functional as TVF
    from PIL import Image
    from omegaconf import OmegaConf
    from torchvision.transforms import InterpolationMode
    from src.lib.karlo.karlo.models.prior_model import PriorDiffusionModel
    from src.lib.karlo.karlo.models.decoder_model import Text2ImProgressiveModel
    from src.lib.karlo.karlo.models.sr_64_256 import ImprovedSupRes64to256ProgressiveModel
    from src.lib.karlo.karlo.models.clip import CustomizedCLIP, CustomizedTokenizer

    ## inputs
    prompt = "a high-resolution photograph of a big red frog on a green leaf."
    prior_num_inference_steps=25
    decoder_num_inference_steps=25
    super_res_num_inference_steps=7

    model_path = "karlo_models"
    batch_size=1
    sampling_type="fast"

    # models
    # tokenizer
    tokenizer_model=CustomizedTokenizer()
    # clip
    clip_model = CustomizedCLIP.load_from_checkpoint(
                os.path.join(CURRENT_PATH, model_path, CKPT_PATH['clip']))
    clip_model = torch.jit.script(clip_model)
    clip_model.cuda(device=1)
    clip_model.eval()
    clip_mean, clip_std = torch.load(
        os.path.join(CURRENT_PATH, model_path, CKPT_PATH['clip_stat']), map_location="cpu")

    # # prior
    # prior_config = OmegaConf.load(os.path.join(model_path, "prior_1B_vit_l.yaml"))
    # prior_model = PriorDiffusionModel.load_from_checkpoint(
    #     prior_config,
    #     tokenizer_model,
    #     clip_mean,
    #     clip_std,
    #     os.path.join(CURRENT_PATH, model_path, CKPT_PATH['prior']),
    #     strict=True,
    # )
    # prior_model.cuda(device=1)
    # prior_model.eval()
    # # decoders
    # decoder_config =OmegaConf.load(os.path.join(model_path, "decoder_900M_vit_l.yaml"))
    # decoder_model = Text2ImProgressiveModel.load_from_checkpoint(
    #     decoder_config,
    #     tokenizer_model,
    #     os.path.join(CURRENT_PATH, model_path, CKPT_PATH['decoder']),
    #     strict=True,
    # )
    # decoder_model.cuda(device=1)
    # decoder_model.eval()
    # # super resolution
    # sr_config = OmegaConf.load(os.path.join(model_path, "improved_sr_64_256_1.4B.yaml"))
    # sr_model = ImprovedSupRes64to256ProgressiveModel.load_from_checkpoint(
    #     sr_config, 
    #     os.path.join(CURRENT_PATH, model_path, CKPT_PATH['sr_256']), strict=True
    # )
    # sr_model.cuda(device=1)
    # sr_model.eval()


    # # """Setup prompts & cfg scales"""
    # prompts_batch = [prompt for _ in range(batch_size)]

    # sampling_type = SAMPLING_CONF[sampling_type]
    # prior_sm = sampling_type["prior_sm"]
    # prior_n_samples = sampling_type["prior_n_samples"]
    # prior_cf_scale = sampling_type["prior_cf_scale"]
    # decoder_sm = sampling_type["decoder_sm"]
    # decoder_cf_scale = sampling_type["decoder_cf_scale"]
    # sr_sm = sampling_type["sr_sm"]
            
    # prior_cf_scales_batch = [prior_cf_scale] * len(prompts_batch)
    # prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="cuda:1")

    # decoder_cf_scales_batch = [decoder_cf_scale] * len(prompts_batch)
    # decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="cuda:1")

    # # """ Get CLIP text feature """
    # max_txt_length = prior_model.model.text_ctx

    # tok, mask = tokenizer_model.padded_tokens_and_mask(prompts_batch, max_txt_length)
    # cf_token, cf_mask = tokenizer_model.padded_tokens_and_mask([""], max_txt_length)
    # if not (cf_token.shape == tok.shape):
    #     cf_token = cf_token.expand(tok.shape[0], -1)
    #     cf_mask = cf_mask.expand(tok.shape[0], -1)

    # tok = torch.cat([tok, cf_token], dim=0)
    # mask = torch.cat([mask, cf_mask], dim=0)

    # tok, mask = tok.to(device="cuda:1"), mask.to(dtype=torch.bool, device="cuda:1")
    # txt_feat, txt_feat_seq = clip_model.encode_text(tok)

    # # get prior
    # timestep_respacing="10"
    # img_feat = prior_model(
    #                 txt_feat,
    #                 txt_feat_seq,
    #                 mask,
    #                 prior_cf_scales_batch,
    #                 timestep_respacing=timestep_respacing,
    #             )
    
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
        prior_model.cuda(device=1)
        prior_model.eval()
        
        txt_feat=torch.randn(2,768).to(device="cuda:1")
        txt_feat_seq=torch.randn(2,77,768).to(device="cuda:1")
        mask=torch.ones(2,77, dtype=torch.bool).to(device="cuda:1")
        prior_cf_scales_batch=torch.tensor([0.4]).to(device="cuda:1")
        timestep_respacing="2"
        torch.onnx.export(
            prior_model,
            (txt_feat, txt_feat_seq, mask, prior_cf_scales_batch, timestep_respacing),
            "save_model_point/prior_model.onnx",
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
    
    # timestep_respacing=decoder_sm
    # images_64_outputs = decoder_model(
    #                 txt_feat,
    #                 txt_feat_seq,
    #                 tok,
    #                 mask,
    #                 img_feat,
    #                 cf_guidance_scales=decoder_cf_scales_batch,
    #                 timestep_respacing=timestep_respacing,
    #             )

    # Export decoder_model
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
        decoder_model.cuda(device=1)
        decoder_model.eval()
        
        txt_feat=torch.randn(2,768).to(device="cuda:1")
        txt_feat_seq=torch.randn(2,77,768).to(device="cuda:1")
        tok=torch.ones(2,77).to(dtype=torch.int32, device="cuda:1")
        mask=torch.ones(2,77).to(dtype=torch.bool, device="cuda:1")
        img_feat=torch.randn(1,768).to(device="cuda:1")
        decoder_cf_scales_batch=torch.tensor([0.4]).to(device="cuda:1")
        timestep_respacing="2"
        torch.onnx.export(
            decoder_model,
            (txt_feat, txt_feat_seq, tok, mask, img_feat, decoder_cf_scales_batch, timestep_respacing),
            "save_model_point/decoder_model.onnx",
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

    # images_64 = None
    # for k, out in enumerate(images_64_outputs):
    #     print(k)
    #     images_64 = out

    # images_64 = torch.clamp(images_64, -1, 1)

    # """ Upsample 64x64 to 256x256 """
    # images_256 = TVF.resize(
    #     images_64,
    #     [256, 256],
    #     interpolation=InterpolationMode.BICUBIC,
    #     antialias=True,
    # )
    # images_256_outputs = sr_model(
    #     images_256, timestep_respacing=sr_sm
    # )

    # Export sr_model
    # NOTE: too large model! so it generates multiple onnx files instead of a single file.
    print("Export super resolution model to onnx format.")
    with torch.autocast("cuda", dtype=torch.float16):
        # load super resolution model.
        sr_config = OmegaConf.load(os.path.join(model_path, "improved_sr_64_256_1.4B.yaml"))
        sr_model = ImprovedSupRes64to256ProgressiveModel.load_from_checkpoint(
            sr_config, 
            os.path.join(CURRENT_PATH, model_path, CKPT_PATH['sr_256']), strict=True
        )
        sr_model.cuda(device=1)
        sr_model.eval()
        
        images_256=torch.randn(1, 3, 256, 256).to(device="cuda:1")
        sr_sm="7"
        torch.onnx.export(
            sr_model,
            (images_256, sr_sm),
            "save_model_point/sr_model.onnx",
            input_names=["images_256", "sr_sm"],
            output_names=["generated_image"],
            dynamic_axes={
                "images_256": {0: "batch", 1: "channel", 2: "height", 3: "width"},
                "generated_image": {0: "batch", 1: "channel", 2: "height", 3: "width"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    # for k, out in enumerate(images_256_outputs):
    #     print(k)
    #     images_256 = out

    # generated_image = torch.clamp(images_256 * 0.5 + 0.5, 0.0, 1.0)

    # images = (torch.permute(generated_image * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy())
    # image = Image.fromarray(images[0])

def export_karlo_from_library():
    
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

    # # tokenizer
    # tokenized_text = karlo_tokenizer(
    #                 [prompt],
    #                 padding="max_length",
    #                 max_length=karlo_tokenizer.model_max_length,
    #                 truncation=True,
    #                 return_tensors="pt",
    #                 ).input_ids
    # tokenized_text_uncond = karlo_tokenizer(
    #                 [""],
    #                 padding="max_length",
    #                 max_length=karlo_tokenizer.model_max_length,
    #                 truncation=True,
    #                 return_tensors="pt",
    #                 ).input_ids

    # # text input preprocessing
    # # Querying the text_encoding model
    # input_ids_1 = pb_utils.Tensor(
    #     "input_ids",
    #     np.concatenate(
    #         [
    #             tokenized_text_uncond.numpy().astype(np.int32),
    #             tokenized_text.numpy().astype(np.int32),
    #         ]
    #     ),
    # )
    # encoding_request = pb_utils.InferenceRequest(
    #     model_name="text_encoder",
    #     requested_output_names=["last_hidden_state"],
    #     inputs=[input_ids_1],
    # )

    # response = encoding_request.exec()
    # # text embedding
    # if response.has_error():
    #     raise pb_utils.TritonModelException(response.error().message())
    # else:
    #     text_embeddings = pb_utils.get_output_tensor_by_name(
    #         response, "last_hidden_state"
    #     )
    # text_embeddings = from_dlpack(text_embeddings.to_dlpack()).clone()
    # text_embeddings = text_embeddings.to("cuda")




    # image = pipe(prompt).images[0]

    # torch.onnx.export(
    #     karlo_image_generator,
    #     (prompt),
    #     "karlo_image_generator.onnx",
    #     input_names=["input_ids"],
    #     output_names=["last_hidden_state", "pooler_output"],
    #     dynamic_axes={
    #         "input_ids": {0: "batch", 1: "sequence"},
    #     },
    #     opset_version=14,
    #     do_constant_folding=True,
    # )
    # decode_result = pipe(prompt)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
        
    group = parser.add_argument_group("Basic configuration")
    group.add_argument("--type","-t", type=str, default="source", help="Onnx export type: source or library")
    args = parser.parse_args()
    
    print("export type: {}".format(args.type))
    if args.type == "source":
        export_karlo_from_source()
    elif args.type == "library":
        export_karlo_from_library()
    else:
        raise RuntimeError("Unknown arguments: {}".format(args.type))