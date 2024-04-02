from torch import nn
from PitxArt import PixArt


class T2I_model(nn.Module):
    def __init__(self, args):
        model_map = {
            'PixArt': PixArt
        }
        self.model = model_map[args.T2I_model](args)

    def forward(self, prompt):
        self.model(prompt)
