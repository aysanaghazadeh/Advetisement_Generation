import json
import os.path
from LLMs.LLM import LLM
import pandas as pd
from jinja2 import Environment, FileSystemLoader


class PromptGenerator:
    def __init__(self, args):
        self.LLM_model = None
        self.descriptions = None
        self.set_LLM(args)
        self.set_descriptions(args)

    def set_LLM(self, args):
        if args.text_input_type == 'LLM':
            self.LLM_model = LLM(args)

    def set_descriptions(self, args):
        if args.text_input_type not in ['LLM', 'AR']:
            self.descriptions = self.get_all_descriptions(args)

    @staticmethod
    def get_all_descriptions(args):
        if args.text_input_type in ['AR', 'LLM']:
            return None
        descriptions = pd.read_csv(args.description_file)
        return descriptions

    @staticmethod
    def get_description(image_filename, descriptions):
        return descriptions.loc[descriptions['ID'] == image_filename]

    @staticmethod
    def get_LLM_input_prompt(args, action_reason):
        data = {'action_reason': action_reason}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template('LLM_input.jinja')
        output = template.render(**data)
        print('LLM input prompt:', output)
        return output

    def get_original_description_prompt(self, args, image_filename):
        data = {'description': self.get_description(image_filename, self.descriptions)}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(''.join([args.text_input_type, '.jinja']))
        output = template.render(**data)
        return output

    def get_LLM_generated_prompt(self, args, image_filename):
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        LLM_input_prompt = self.get_LLM_input_prompt(args, action_reason)
        description = self.LLM_model(LLM_input_prompt)
        print('LLM description: ', description)
        data = {'description': description}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(''.join([args.text_input_type, '.jinja']))
        output = template.render(**data)
        print('LLM generated prompt:', output)
        return output

    @staticmethod
    def get_AR_prompt(args, image_filename):
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        data = {'action_reason': action_reason}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(''.join([args.text_input_type, '.jinja']))
        output = template.render(**data)
        print('AR prompt:', output)
        return output

    def generate_prompt(self, args, image_filename):
        prompt_generator_name = f'get_{args.text_input_type}_prompt'
        print('method: ', prompt_generator_name)
        if prompt_generator_name == 'get_LLM_prompt':
            prompt_generator_name = 'get_LLM_generated_prompt'
        prompt_generation_method = getattr(self, prompt_generator_name)
        prompt = prompt_generation_method(args, image_filename)
        print('prompt: ', prompt)
        return prompt
