import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn


class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
        unet = UNet2DConditionModel.from_config(base_model_id,
                                                subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name),
                                        map_location="cuda"))
        self.pipe = DiffusionPipeline.from_pretrained(base_model_id,
                                                      unet=unet,
                                                      torch_dtype=torch.float16,
                                                      variant="fp16").to("cuda")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

    def forward(self, prompt):
        image = self.pipe(prompt=prompt,
                          num_inference_steps=4,
                          guidance_scale=0,
                          timesteps=[999, 749, 499, 249]).images[0]
        return image
