from torch import nn
from T2I_models.PitxArt import PixArt
from T2I_models.SDXLFlash import SDXL


class T2IModel(nn.Module):
    def __init__(self, args):
        super(T2IModel, self).__init__()
        model_map = {
            'PixArt': PixArt,
            'SDXL': SDXL,
        }
        self.model = model_map[args.T2I_model](args)

    def forward(self, prompt):
        return self.model(prompt)
