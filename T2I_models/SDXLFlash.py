import torch
from torch import nn
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler, StableDiffusionPipeline


class SDXL(nn.Module):
    def __init__(self, args):
        super(SDXL, self).__init__()
        self.device = args.device
        if not args.train:
            self.pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash",
                                                                  torch_dtype=torch.float16).to(device=args.device)
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
                                                                           timestep_spacing="trailing")
        else:
            model_id = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipe = self.pipe.to(device='cuda:1')

    def forward(self, prompt):
        # negative_prompt = "typical,(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, " \
        #                   "extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected " \
        #                   "limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW "
        image = self.pipe(prompt, num_inference_steps=20).images[0]
        return image
