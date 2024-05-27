import torch
from torch import nn
from diffusers import PixArtAlphaPipeline


class PixArt(nn.Module):
    def __init__(self, args):
        super(PixArt, self).__init__()
        self.device = args.device
        self.pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS",
                                                        torch_dtype=torch.float16,
                                                        device_map='auto')
        # self.pipe = self.pipe.to(device=args.device)

    def forward(self, prompt):
        # prompt = prompt.to(self.device)
        # image = self.pipe(prompt).images[0]
        # return image
        # Ensure the prompt is on the correct device
        if isinstance(prompt, str):
            prompt = [prompt]

        # Move prompt to the correct device
        prompt = [p.to(self.device) if isinstance(p, torch.Tensor) else p for p in prompt]

        # Generate the image using the pipeline
        output = self.pipe(prompt)
        return output.images[0]
