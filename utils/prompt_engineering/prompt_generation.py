from T2I_models.T2I_model import T2I_model
from LLMs.LLM import LLM
import torch
from torch import nn
import pandas as pd
from jinja2 import Environment, FileSystemLoader

def get_all_descriptions(file_path, args):
    if args.text_input_type in ['AR', 'LLM']:
        return None
    descriptions = pd.read_csv(file_path)
    return descriptions

def get_description(image_filename, descriptions):
    return descriptions.loc[descriptions['ID'] == image_filename]


def get_combine_prompt(args, image_filename, descriptions):
    description = get_description(image_filename, descriptions)
    env = Environment(loader=FileSystemLoader('path/to/your/templates/directory'))
    template = env.get_template('prompt_template.jinja')
    output = template.render(data)

def process_prompt(args, image_filename):



