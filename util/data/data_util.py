import random
from accelerate import PartialState
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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-instruct",
                                              token='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',
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
        action_reason = '\n-'.join(data_point['QA'][0])
        format = """Follow the following format:
                    Visual scene: {Description of the overall scene, list of objects and the relation between them}
                    Texts in the image: {list of texts in the image}"""

        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {action_reason}
                    
                    {format}
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
        image_url = image_url[0]
        QA = QAs[image_url]
        description = descriptions.loc[descriptions['ID'] == image_url]['description'].values
        dataset['QA'].append(QA)
        dataset['description'].append(description)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(['QA', "description"])
    print(dataset)
    return dataset


def get_train_LLAMA3_Dataloader(args):
    image_urls = get_train_data(args).values[:2300]
    dataset = get_LLAMA3_training_data(args, image_urls)
    return dataset


def get_LLAMA3_CPO_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-instruct",
                                              token='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',
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    descriptions = pd.read_csv(args.description_file)
    dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    negative_QAs = {}
    negative_QAs['reason'] = json.load(
        open(os.path.join(args.data_path, 'train/reason_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['action']= json.load(
        open(os.path.join(args.data_path, 'train/action_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['adjective'] = json.load(
        open(os.path.join(args.data_path, 'train/adjective_hard_QA_Combined_Action_Reason_train.json')))
    negative_QAs['semantic'] = json.load(
        open(os.path.join(args.data_path, 'train/semantic_hard_QA_Combined_Action_Reason_train.json')))

    for image_url in image_urls:
        if image_url in negative_QAs['reason']:
            QA = QAs[image_url]
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values
            prompt = f"""What is the correct interpretation for the described image:
                                         Description: {description}"""
            for AR in QA[0]:
                for negative_type in negative_QAs:
                    for negative_option in negative_QAs[negative_type][image_url][1]:
                        if (negative_option in negative_QAs[negative_type][image_url][0]) or (negative_option in dataset['rejected']):
                            continue
                        dataset['prompt'].append(prompt)
                        dataset['chosen'].append(AR)
                        dataset['rejected'].append(negative_option)
    print(dataset)
    dataset = Dataset.from_dict(dataset)
    with PartialState().local_main_process_first():
        ds = dataset.map(process)

    train_dataset = ds
    return train_dataset


def get_train_LLAMA3_CPO_Dataloader(args):
    image_urls = list(json.load(open(os.path.join(args.data_path,
                                                  'train/reason_hard_QA_Combined_Action_Reason_train.json'))).keys())
    dataset = get_LLAMA3_CPO_training_data(args, image_urls)
    return dataset

def get_LLAMA3_RLHF_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token='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',
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def format_dataset(data_point):
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
        product_description = \
            product_descriptions.loc[product_descriptions['ID'] == image_url[0]]['description'].values[0]
        negative_description = \
            negative_descriptions.loc[negative_descriptions['ID'] == image_url[0]]['description'].values[0]
        action = QA[0][0].lower().split('because')[0]
        dataset['QA'].append(QA)
        dataset['chosen'].append(chosen_description)
        dataset['rejected'].append(product_description + action)
        dataset['QA'].append(QA)
        dataset['chosen'].append(chosen_description)
        dataset['rejected'].append(
            'image of ' + negative_description + product_description.split('image of')[-1] + action)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset)
    print(dataset)
    return dataset


def get_RLHF_train_LLAMA3_Dataloader(args):
    image_urls = get_train_data(args)
    dataset = get_LLAMA3_RLHF_training_data(args, image_urls)
    return dataset


def get_LLAMA3_RLAIF_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint-4350/'),
        token='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',
        padding='max_length',
        max_length=512)
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {data_point['action_reason']}
                    Description of the image:
                """
        data_point['query'] = {'query': prompt, 'action_reason': '\n'.join(data_point['action_reason'])}
        tokens = tokenizer.encode(prompt, padding='max_length', max_length=512)
        # data_point["input_ids"] = tokens
        return data_point

    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    dataset = {'query': [], 'action_reason': []}
    for image_url in image_urls:
        QA = QAs[image_url][0]
        dataset['query'].append(str(QA))
        dataset['action_reason'].append(QA)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset, batched=False)
    print(dataset)
    return dataset


def get_LLAMA3_RLAIF_Dataloader(args):
    image_urls = get_train_data(args).ID.values
    dataset = get_LLAMA3_RLAIF_training_data(args, image_urls)
    return dataset


def get_LLAMA3_DPO_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint-4350/'),
        token='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',
        padding='left')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
        prompt = f"""Describe an advertisement image that conveys the following messages in detail:
                    {data_point['prompt']}
                    Description of the image:
                """
        data_point['prompt'] = prompt
        # data_point['chosen'] = tokenizer.apply_chat_template(data_point['chosen'], tokenize=False)
        # data_point['rejected'] = tokenizer.apply_chat_template(data_point['rejected'], tokenize=False)
        # tokens = tokenizer.encode(prompt)
        # data_point["input_ids"] = tokens
        return data_point

    QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    PA_train_1 = json.load(open(os.path.join(args.results, 'results', 'llama3_FT_generated_description_train_set_persuasiveness_alignment.json_SDXL_train_images_20240615_143729_persuasiveness_alignment.json')))
    PA_train_2 = json.load(open(os.path.join(args.results, 'results', 'llama3_FT_generated_description_new_train_set_persuasiveness_alignment.json_SDXL_train_images_20240617_074807_persuasiveness_alignment.json')))
    llama_descriptions_1 = pd.read_csv(os.path.join(args.data_path, 'train/llama3_FT_generated_description_new_train_set.csv'))
    llama_descriptions_2 = pd.read_csv(os.path.join(args.data_path, 'train/llama3_FT_generated_description_train_set.csv'))
    dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    for image_url in image_urls:
        QA = str(QAs[image_url][0])
        PA1 = PA_train_1[image_url]
        PA2 = PA_train_2[image_url]
        description_1 = llama_descriptions_1.loc[llama_descriptions_1['ID'] == image_url]['description'].values[0]
        description_2 = llama_descriptions_2.loc[llama_descriptions_2['ID'] == image_url]['description'].values[0]
        if PA1 > PA2:
            dataset['chosen'].append(description_1)
            dataset['rejected'].append(description_2)
        if PA2 > PA1:
            dataset['chosen'].append(description_2)
            dataset['rejected'].append(description_1)
        if PA1 == PA2:
            if random.uniform(0, 1) > 0.5:
                dataset['chosen'].append(description_2)
                dataset['rejected'].append(description_1)
            else:
                dataset['chosen'].append(description_1)
                dataset['rejected'].append(description_2)
        dataset['prompt'].append(QA)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(format_dataset, batched=False)
    print(dataset)
    return dataset


def get_LLAMA3_DPO_Dataloader(args):
    image_urls = list(get_train_data(args).ID.values)[:3140]
    dataset = get_LLAMA3_DPO_training_data(args, image_urls)
    return dataset


def get_Phi3_training_data(args, image_urls):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                              token='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',
                                              trust_remote_code=True,
                                              padding='right')
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(data_point):
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
