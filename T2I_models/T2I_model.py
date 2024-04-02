from torch import nn
from T2I_models.PitxArt import PixArt


class T2IModel(nn.Module):
    def __init__(self, args):
        model_map = {
            'PixArt': PixArt
        }
        self.model = model_map[args.T2I_model](args)

    def forward(self, prompt):
        return self.model(prompt)
