from torch import nn
from Mistral7B import Mistral7B
from Mixtral7B import Mixtral7B
from Mistral7BInstruct import Mistral7BInstruct


class LLM(nn.Module):
    def __init__(self, args):
        model_map = {
            'Mistral7B': Mistral7B,
            'Mixtral7B': Mixtral7B,
            'Mistral7BInstruct': Mistral7BInstruct
        }
        self.model = model_map[args.LLM](args)

    def forward(self, prompt):
        self.model(prompt)
