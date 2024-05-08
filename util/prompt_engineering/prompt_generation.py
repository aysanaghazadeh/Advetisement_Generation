import json
import os.path
from LLMs.LLM import LLM
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from util.data.mapping import SENTIMENT_MAP
from collections import Counter


class PromptGenerator:
    def __init__(self, args):
        self.LLM_model = None
        self.descriptions = None
        self.sentiments = None
        self.set_LLM(args)
        self.set_descriptions(args)
        self.set_sentiments(args)

    def set_LLM(self, args):
        if args.text_input_type == 'LLM':
            self.LLM_model = LLM(args)

    def set_descriptions(self, args):
        if args.text_input_type not in ['LLM', 'AR']:
            self.descriptions = self.get_all_descriptions(args)

    def set_sentiments(self, args):
        if args.with_sentiment:
            self.sentiments = self.get_all_sentiments(args)

    @staticmethod
    def get_all_sentiments(args):
        if not args.with_sentiment:
            return None
        sentiment_file = os.path.join(args.data_path, 'train/Sentiments_train.json')
        sentiments = json.load(open(sentiment_file))
        return sentiments

    @staticmethod
    def get_all_descriptions(args):
        if args.text_input_type in ['AR', 'LLM']:
            return None
        descriptions = pd.read_csv(args.description_file)
        return descriptions

    @staticmethod
    def get_description(image_filename, descriptions):
        return descriptions.loc[descriptions['ID'] == image_filename]['description'].values[0]

    @staticmethod
    def get_LLM_input_prompt(args, action_reason):
        data = {'action_reason': action_reason}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.llm_prompt)
        output = template.render(**data)
        return output

    @staticmethod
    def get_most_frequent(values):
        tuple_list = [tuple(inner_list) for inner_list in values]
        # Create a Counter object from the tuple list
        counter = Counter(tuple_list)
        # Get the most common tuple
        most_freq_tuple, _ = counter.most_common(1)[0]
        # Convert tuple back to list if necessary
        return list(most_freq_tuple)

    def get_original_description_prompt(self, args, image_filename):
        data = {'description': self.get_description(image_filename, self.descriptions)}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        return output

    def get_LLM_generated_prompt(self, args, image_filename):
        sentiment = ''
        if args.with_sentiment:
            if image_filename in self.sentiments:
                sentiment_ids = self.sentiments[image_filename]
                sentiment_id = self.get_most_frequent(sentiment_ids)
                if sentiment_id in SENTIMENT_MAP:
                    sentiment = SENTIMENT_MAP[sentiment_id]
            else:
                print(f'there is no sentiment for image: {image_filename}')
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        LLM_input_prompt = self.get_LLM_input_prompt(args, action_reason)
        description = self.LLM_model(LLM_input_prompt)
        if 'objects:' in description:
            objects = description.split('objects:')[1]
            description = description.split('objects:')[0]
        else:
            objects = None
        data = {'description': description, 'action_reason': action_reason, 'objects': objects, 'sentiment': sentiment}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
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
        template = env.get_template(args.T2I_prompt)
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
        return prompt
