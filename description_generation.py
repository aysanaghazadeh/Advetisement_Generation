import json
from jinja2 import Environment, FileSystemLoader
from transformers import pipeline
from util.data.trian_test_split import get_test_data, get_train_data
from PIL import Image
import os
import csv
import pandas as pd
from configs.inference_config import get_args
from util.prompt_engineering.prompt_generation import PromptGenerator
from LLMs.LLM import LLM


def get_model(args):
    # Load model directly
    if args.description_type == 'combine':
        pipe = LLM(args)
        return pipe
    model_map = {
        'LLAVA': 'llava-hf/llava-1.5-13b-hf'
    }
    model_id = model_map[args.VLM]
    pipe = pipeline("image-to-text", model=model_id, device_map='auto')
    return pipe


def get_llm(args):
    model = PromptGenerator(args)
    model.set_LLM(args)
    return model


def get_single_description(args, image_url, pipe):
    image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    template = env.get_template(args.VLM_prompt)
    prompt = template.render()
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 512})
    description = outputs[0]['generated_text'].split('ASSISTANT: ')[-1]
    return description


def get_combine_description(args, image_url, pipe):
    IN_descriptions = pd.read_csv(os.path.join(args.data_path, f'train/IN_LLAVA_description_{args.task}_version_2.csv'))
    IN_description = IN_descriptions.loc[IN_descriptions['ID'] == image_url]['description'].values[0]
    UH_descriptions = pd.read_csv(os.path.join(args.data_path, f'train/UH_LLAVA_description_{args.task}_version_2.csv'))
    UH_description = UH_descriptions.loc[UH_descriptions['ID'] == image_url]['description'].values[0]
    v_descriptions = pd.read_csv(os.path.join(args.data_path, f'train/v_LLAVA_description_{args.task}.csv'))
    v_description = v_descriptions.loc[v_descriptions['ID'] == image_url]['description'].values[0]
    data = {'IN': IN_description, 'UH': UH_description, 'v': v_description, 'token_length':None}
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    template = env.get_template(args.VLM_prompt)
    prompt = template.render(**data)
    description = pipe(prompt=prompt)
    return description


def get_descriptions(args):
    if args.task == 'whoops':
        images = [f'{i}.png' for i in range(500)]
    else:
        # images = get_train_data(args)['ID'].values
        images = list(json.load(open(os.path.join(args.data_path, 'train', 'symbolic_random_50.csv'))).keys())
    print(f'number of images in the set: {len(images)}')
    print('*' * 100)
    description_file = os.path.join(args.data_path, 'train',
                                    f'{args.description_type}_{args.VLM}_description_{args.task}_version_2.csv')
    if os.path.exists(description_file):
        return pd.read_csv(description_file)
    with open(description_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    pipe = get_model(args)
    processed_images = set()
    for image_url in images:
        if image_url in processed_images:
            continue
        processed_images.add(image_url)
        if args.description_type == 'combine':
            description = get_combine_description(args, image_url, pipe)
        else:
            description = get_single_description(args, image_url, pipe)
        print(f'output of image {image_url} is {description}')
        print('-' * 80)
        pair = [image_url, description]
        with open(description_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)


def get_llm_generated_prompt(args, test_images):
    print(f'number of images in test set: {len(test_images)}')
    print('*' * 100)
    description_file = os.path.join(args.data_path, f'train/{args.llm_prompt.replace(".jinja", "")}_llama3_ppo_generated_description_8000steps_test_set.csv')
    if os.path.exists(description_file):
        return pd.read_csv(description_file)
    with open(description_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    prompt_generator = get_llm(args)
    processed_images = set()
    for image_url in test_images:
        if image_url in processed_images:
            continue
        processed_images.add(image_url)
        description = prompt_generator.generate_prompt(args, image_url)
        print(f'output of image {image_url} is {description}')
        print('-' * 80)
        pair = [image_url, description]
        with open(description_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)


def get_negative_descriptions(args):
    train_images = get_train_data(args)['ID'].values
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    print(f'number of images in train set: {len(train_images)}')
    print('*' * 100)
    product_file = os.path.join(args.data_path, 'train/product_name_train_set.csv')
    if os.path.exists(product_file):
        return pd.read_csv(product_file)
    with open(product_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    negative_file = os.path.join(args.data_path, 'train/negative_prompt_train_set.csv')
    if os.path.exists(negative_file):
        return pd.read_csv(negative_file)
    with open(negative_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    prompt_generator = get_llm(args)
    for image_url in train_images:
        action_reason = '\n'.join(QA[image_url][0])
        print(f'action reason for image {image_url} is {action_reason}')
        args.T2I_prompt = 'product_image_generation.jinja'
        args.llm_prompt = 'product_detector.jinja'
        product_names = prompt_generator.generate_prompt(args, image_url)
        print(f'products in image {image_url} is {product_names}')
        pair = [image_url, product_names]
        with open(product_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)
        args.T2I_prompt = 'adjective_only.jinja'
        args.llm_prompt = 'negative_adjective_detector.jinja'
        adjective = prompt_generator.generate_prompt(args, image_url)
        print(f'negative adjective in image {image_url} is {adjective}')
        pair = [image_url, adjective]
        with open(negative_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

if __name__ == '__main__':
    args = get_args()
    # test_images = get_test_data(args)['ID'].values[:1340]
    descriptions = get_descriptions(args)
    # descriptions = get_llm_generated_prompt(args, test_images)
    # get_negative_descriptions(args)

