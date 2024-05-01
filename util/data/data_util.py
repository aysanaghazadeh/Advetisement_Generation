from util.data.trian_test_split import get_train_data
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import os
import json


def get_Mistral7B_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding='right')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
        # print(data_point['QA'])
        action_reason = '\n-'.join(data_point['QA'][0])
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    Description of the image: {data_point['description']}
                """
        tokens = tokenizer(prompt,
                           truncation=True,
                           max_length=256,
                           padding="max_length", )
        tokens["labels"] = tokens['input_ids'].copy()
        return tokens

    descriptions = pd.read_csv(args.description_file)
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    dataset = {'QA': [], 'description': []}
    for image_url in image_urls:
        QA = QAs[image_url[0]]
        description = descriptions.loc[descriptions['ID'] == image_url[0]]['description'].values
        dataset['QA'].append(QA)
        dataset['description'].append(description)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(['QA', "description"])
    return dataset


def get_train_Mistral7B_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_Mistral7B_training_data(args, image_urls)
    return dataset


def get_LLAMA3_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token='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',
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
        # print(data_point['QA'])
        action_reason = '\n-'.join(data_point['QA'][0])
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    Description of the image: {data_point['description']}
                """
        tokens = tokenizer(prompt,
                           truncation=True,
                           max_length=256,
                           padding="max_length", )
        tokens["labels"] = tokens['input_ids'].copy()
        return tokens

    descriptions = pd.read_csv(args.description_file)
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    dataset = {'QA': [], 'description': []}
    for image_url in image_urls:
        QA = QAs[image_url[0]]
        description = descriptions.loc[descriptions['ID'] == image_url[0]]['description'].values
        dataset['QA'].append(QA)
        dataset['description'].append(description)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(['QA', "description"])
    print(dataset)
    return dataset


def get_train_LLAMA3_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_LLAMA3_training_data(args, image_urls)
    return dataset


def get_Phi3_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                              token='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',
                                              trust_remote_code=True,
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
        # print(data_point['QA'])
        action_reason = '\n-'.join(data_point['QA'][0])
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    Description of the image: {data_point['description']}
                """
        tokens = tokenizer(prompt,
                           truncation=True,
                           max_length=256,
                           padding="max_length", )
        tokens["labels"] = tokens['input_ids'].copy()
        return tokens

    descriptions = pd.read_csv(args.description_file)
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    dataset = {'QA': [], 'description': []}
    for image_url in image_urls:
        QA = QAs[image_url[0]]
        description = descriptions.loc[descriptions['ID'] == image_url[0]]['description'].values
        dataset['QA'].append(QA)
        dataset['description'].append(description)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(['QA', "description"])
    print(dataset)
    return dataset


def get_train_Phi3_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_Phi3_training_data(args, image_urls)
    return dataset
