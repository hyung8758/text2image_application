"""
export_karlo : karlo model in detail.

Hyungwon Yang
"""

import os, sys
CURRENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
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


## inputs
prompt = "a high-resolution photograph of a big red frog on a green leaf."

batch_size=1
sampling_type="fast"
cuda_device=1 # set -1 to use all gpus.
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

# models
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

# prior
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
# decoders
decoder_config =OmegaConf.load(os.path.join(model_path, "decoder_900M_vit_l.yaml"))
decoder_model = Text2ImProgressiveModel.load_from_checkpoint(
    decoder_config,
    tokenizer_model,
    os.path.join(CURRENT_PATH, model_path, CKPT_PATH['decoder']),
    strict=True,
)
decoder_model.cuda(device=1)
decoder_model.eval()
# super resolution
sr_config = OmegaConf.load(os.path.join(model_path, "improved_sr_64_256_1.4B.yaml"))
sr_model = ImprovedSupRes64to256ProgressiveModel.load_from_checkpoint(
    sr_config, 
    os.path.join(CURRENT_PATH, model_path, CKPT_PATH['sr_256']), strict=True
)
sr_model.cuda(device=1)
sr_model.eval()


# """Setup prompts & cfg scales"""
prompts_batch = [prompt for _ in range(batch_size)]

sampling_type = SAMPLING_CONF[sampling_type]
prior_sm = sampling_type["prior_sm"]
prior_n_samples = sampling_type["prior_n_samples"]
prior_cf_scale = sampling_type["prior_cf_scale"]
decoder_sm = sampling_type["decoder_sm"]
decoder_cf_scale = sampling_type["decoder_cf_scale"]
sr_sm = sampling_type["sr_sm"]
        
prior_cf_scales_batch = [prior_cf_scale] * len(prompts_batch)
prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="cuda:1")

decoder_cf_scales_batch = [decoder_cf_scale] * len(prompts_batch)
decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="cuda:1")

# """ Get CLIP text feature """
max_txt_length = prior_model.model.text_ctx

tok, mask = tokenizer_model.padded_tokens_and_mask(prompts_batch, max_txt_length)
cf_token, cf_mask = tokenizer_model.padded_tokens_and_mask([""], max_txt_length)
if not (cf_token.shape == tok.shape):
    cf_token = cf_token.expand(tok.shape[0], -1)
    cf_mask = cf_mask.expand(tok.shape[0], -1)

tok = torch.cat([tok, cf_token], dim=0)
mask = torch.cat([mask, cf_mask], dim=0)

tok, mask = tok.to(device="cuda:1"), mask.to(dtype=torch.bool, device="cuda:1")
txt_feat, txt_feat_seq = clip_model.encode_text(tok)

# get prior
img_feat = prior_model(
                txt_feat=txt_feat,
                txt_feat_seq=txt_feat_seq,
                mask=mask,
                cf_guidance_scales=prior_cf_scales_batch,
                timestep_respacing=prior_sm,
            )
images_64_outputs = decoder_model(
                txt_feat=txt_feat,
                txt_feat_seq=txt_feat_seq,
                tok=tok,
                mask=mask,
                img_feat=img_feat,
                cf_guidance_scales=decoder_cf_scales_batch,
                timestep_respacing=decoder_sm,
            )

images_64 = None
for k, out in enumerate(images_64_outputs):
    print(k)
    images_64 = out
# images_64 = torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)
images_64 = torch.clamp(images_64, -1, 1)

""" Upsample 64x64 to 256x256 """
images_256 = TVF.resize(
    images_64,
    [256, 256],
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)
images_256_outputs = sr_model(
    images_256, timestep_respacing=sr_sm
)

for k, out in enumerate(images_256_outputs):
    print(k)
    images_256 = out

generated_image = torch.clamp(images_256 * 0.5 + 0.5, 0.0, 1.0)

images = (torch.permute(generated_image * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy())
image = Image.fromarray(images[0])