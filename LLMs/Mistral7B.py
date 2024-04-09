from torch import nn
from transformers import pipeline

class Mistral7B(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pipe = pipeline("text-generation",
                             model="mistralai/Mistral-7B-v0.1",
                             device_map='auto',
                             max_length=150)

    def forward(self, prompt):
        output = self.pipe(prompt)
        return output


