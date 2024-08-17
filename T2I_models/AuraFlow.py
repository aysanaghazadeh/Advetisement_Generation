import torch
from torch import nn
from diffusers import AuraFlowPipeline
from transformers import BitsAndBytesConfig


class AuraFlow(nn.Module):
    def __init__(self, args):
        super(AuraFlow, self).__init__()
        self.device = args.device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.pipeline = AuraFlowPipeline.from_pretrained(
            "fal/AuraFlow-v0.2",
            torch_dtype=torch.float16,
            variant="fp16",
            quantization_config=quantization_config
        ).to("cuda")

    def forward(self, prompt):
        image = self.pipeline(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=28,
                    generator=torch.Generator().manual_seed(666),
                    guidance_scale=5,
                    ).images[0]
        return image
