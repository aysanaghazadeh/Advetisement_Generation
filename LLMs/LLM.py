from torch import nn
from LLMs.Mistral7B import Mistral7B
from LLMs.Mixtral7B import Mixtral7B
from LLMs.Mistral7BInstruct import Mistral7BInstruct


class LLM(nn.Module):
    def __init__(self, args):
        super(LLM, self).__init__()
        model_map = {
            'Mistral7B': Mistral7B,
            'Mixtral7B': Mixtral7B,
            'Mistral7BInstruct': Mistral7BInstruct
        }
        self.model = model_map[args.LLM](args)

    def forward(self, prompt):
        output = self.model(prompt)
        return output

