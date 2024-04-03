from T2I_models.T2I_model import T2IModel
from utils.prompt_engineering.prompt_generation import PromptGenerator
from torch import nn


class AdvertisementImageGeneration(nn.Module):
    def __init__(self, args):
        super(AdvertisementImageGeneration, self).__init__()
        self.args = args
        self.prompt_generator = PromptGenerator(self.args)
        self.T2I_model = T2IModel(args)

    def forward(self, image_filename):
        prompt = self.prompt_generator.generate_prompt(self.args, image_filename)
        image = self.T2I_model(prompt)
        return image, prompt