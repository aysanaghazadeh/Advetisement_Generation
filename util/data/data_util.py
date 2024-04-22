from util.data.trian_test_split import get_train_data
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import os
import json


def get_LLM_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding='right')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(self, data_point):
        action_reason = '\n-'.join(data_point['QA'])
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    Description of the image: {data_point['description']}
                """
        tokens = self.tokenizer(prompt,
                                truncation=True,
                                max_length=256,
                                padding="max_length", )
        tokens["labels"] = tokens['input_ids'].copy()
        return tokens

    descriptions = pd.read_csv(args.description_file)
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    dataset = {'QA': [], 'description': []}
    for image_url in image_urls:
        print(image_url)
        QA = QAs[image_url]
        description = descriptions.loc[descriptions['ID'] == image_url]['description'].values
        dataset['QA'].append(QA)
        dataset['description'].append(description)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(['QA', "description"])
    return dataset


def get_train_Mistral7B_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_LLM_training_data(args, image_urls)
    return dataset

