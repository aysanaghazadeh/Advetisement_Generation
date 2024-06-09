from util.data.trian_test_split import get_train_data
from torch.utils.data import DataLoader
from util.data.dataset import LLAMA3RLAIF
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


def get_LLAMA3_RLHF_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token='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',
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def format_dataset(data_point):
        # print(data_point['QA'])
        kwargs = {"padding": "max_length",
                  "truncation": True,
                  "max_length": 256,
                  "return_tensors": "pt"
                  }
        action_reason = '\n-'.join(data_point['QA'][0])
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    Description of the image:
                """
        prompt_plus_chosen_response = prompt + data_point['chosen']
        prompt_plus_rejected_response = prompt + data_point['rejected']
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0],
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0],
            "attention_mask_rejected": tokens_rejected["attention_mask"][0]
        }

    chosen_descriptions = pd.read_csv(args.description_file)
    product_descriptions = pd.read_csv(args.product_file)
    negative_descriptions = pd.read_csv(args.negative_file)
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    dataset = {'QA': [], 'chosen': [], 'rejected': []}
    for image_url in image_urls:
        QA = QAs[image_url[0]]
        chosen_description = chosen_descriptions.loc[chosen_descriptions['ID'] == image_url[0]]['description'].values[0]
        product_description = product_descriptions.loc[product_descriptions['ID'] == image_url[0]]['description'].values[0]
        negative_description = negative_descriptions.loc[negative_descriptions['ID'] == image_url[0]]['description'].values[0]
        action = QA[0][0].lower().split('because')[0]
        dataset['QA'].append(QA)
        dataset['chosen'].append(chosen_description)
        dataset['rejected'].append(product_description + action)
        dataset['QA'].append(QA)
        dataset['chosen'].append(chosen_description)
        dataset['rejected'].append('image of ' + negative_description + product_description.split('image of')[-1] + action)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset)
    print(dataset)
    return dataset


def get_RLHF_train_LLAMA3_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_LLAMA3_RLHF_training_data(args, image_urls)
    return dataset


# def get_LLAMA3_RLAIF_Dataloader(args):
#     image_urls = get_train_data(args)
#     LLAMA3_data = LLAMA3RLAIF(args, image_urls)
#     return DataLoader(LLAMA3_data, shuffle=False, batch_size=1, num_workers=os.cpu_count())


def get_LLAMA3_RLAIF_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token='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',
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
        # print(data_point['QA'])
        action_reason = '\n-'.join(data_point['query'])
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    Description of the image:
                """
        tokens = tokenizer.encode(prompt,
                           truncation=True,
                           max_length=256,
                           padding="max_length", )
        data_point["input_ids"] = tokens.to(device=args.device)
        return data_point
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    dataset = {'query': []}
    for image_url in image_urls:
        QA = str(QAs[image_url[0]][0])
        # description = descriptions.loc[descriptions['ID'] == image_url[0]]['description'].values
        dataset['query'].append(QA)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset, batched=False)
    # dataset = dataset.remove_columns(['QA', "description"])
    print(dataset)
    return dataset


def get_LLAMA3_RLAIF_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_LLAMA3_RLAIF_training_data(args, image_urls)
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
