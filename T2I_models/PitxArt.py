import torch
from torch import nn
from diffusers import PixArtAlphaPipeline


class PixArt(nn.Module):
    def __init__(self, args):
        super(PixArt, self).__init__()
        self.device = args.device
        self.pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS",
                                                        torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device=args.device)
        self.pipe = nn.DataParallel(self.pipe)

    def forward(self, prompt):
        # prompt = prompt.to(self.device)
        # image = self.pipe(prompt).images[0]
        # return image
        if isinstance(prompt, str):
            prompt = [prompt]

            # Perform the forward pass with the pipeline
        output = self.pipe(prompt)
        image = output[0].images[0]
        return image
