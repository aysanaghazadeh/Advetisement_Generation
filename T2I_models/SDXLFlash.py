import torch
from torch import nn
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler
from transformers import BitsAndBytesConfig


class SDXL(nn.Module):
    def __init__(self, args):
        super(SDXL, self).__init__()
        self.device = args.device
        if not (args.train):
            self.pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash",
                                                                  torch_dtype=torch.float16).to(device=args.device)
            # Ensure sampler uses "trailing" timesteps.
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
                                                                           timestep_spacing="trailing")
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            self.pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash", torch_dtype=torch.float16)
            # Ensure sampler uses "trailing" timesteps.
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
                                                                           timestep_spacing="trailing")
            self.apply_quantization(self.pipe)

            # Move the model back to the appropriate device
            self.pipe = self.pipe.to(device=args.device)

    def apply_quantization(self, model):
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

    def forward(self, prompt):
        negative_prompt = "typical,(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"
        image = self.pipe(prompt, num_inference_steps=15, guidance_scale=6, negative_prompt=negative_prompt).images[0]
        return image
