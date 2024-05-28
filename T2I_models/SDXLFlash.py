import torch
from torch import nn
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler


class SDXL(nn.Module):
    def __init__(self, args):
        super(SDXL, self).__init__()
        self.device = args.device
        self.pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash", torch_dtype=torch.float16).to(device=args.device)
        # Ensure sampler uses "trailing" timesteps.
        self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    def forward(self, prompt):
        negative_prompt = "typical,(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"
        image = self.pipe(prompt, num_inference_steps=15, guidance_scale=6, negative_prompt=negative_prompt).images[0]
        return image
