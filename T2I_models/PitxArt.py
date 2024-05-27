import torch
from torch import nn
from diffusers import PixArtAlphaPipeline


class PixArt(nn.Module):
    def __init__(self, args):
        super(PixArt, self).__init__()
        self.pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS",
                                                        torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device=args.device)
        # self.pipe = self.pipe.to(device=f'cuda:{1}')
    def forward(self, prompt):
        image = self.pipe(prompt).images[0]
        return image
