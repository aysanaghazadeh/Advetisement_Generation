from torch import nn
from transformers import AutoTokenizer
import pandas as pd
import os
import json
from torch.utils.data import Dataset


class Mistral7BTrainingDataset(nn.Module):
    def __init__(self, args, image_urls):
        super(Mistral7BTrainingDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding='right')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.descriptions = pd.read_csv(args.description_file)
        self.image_urls = image_urls
        self.QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))

    def format_dataset(self, data_point):
        action_reason = '\n-'.join(data_point['action_reason'])
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

    def __getitem__(self, item):
        image_url = self.image_urls[item]
        description = self.descriptions.loc[self.descriptions['ID'] == image_url]['description'].values
        QAs = self.QA[image_url][0]
        QAs = '\n'.join([f'{i}. {QA}' for i, QA in enumerate(QAs)])
        prompt = f'Describe an advertisement image that conveys the following messages in detail:\n {QAs}'
        model_inputs = self.tokenizer(prompt, max_length=300, truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(description, max_length=300, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def __len__(self):
        return len(self.image_urls)


class LLAMA3RLAIF(Dataset):
    def __init__(self, args, image_urls):
        super().__init__()
        self.args = args
        self.QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        self.image_urls = image_urls
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                  token='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')

    def __getitem__(self, item):
        image_url = self.image_urls[item]
        print(image_url)
        action_reason = self.QA[image_url][0]
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    Description of the image:
                """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device=self.args.device)
        return prompt, inputs

    def __len__(self):
        return len(self.image_urls)
