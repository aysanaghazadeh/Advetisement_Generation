import json

from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline
from util.data.trian_test_split import get_test_data, get_train_data
from PIL import Image
import os
import csv
import pandas as pd
from configs.inference_config import get_args
from util.prompt_engineering.prompt_generation import PromptGenerator


def get_model():
    # Load model directly
    model_id = "llava-hf/llava-1.5-13b-hf"
    pipe = pipeline("image-to-text", model=model_id, device_map='auto')
    return pipe

def get_llm(args):
    model = PromptGenerator(args)
    model.set_LLM(args)
    return model


def get_descriptions(args):
    train_images = get_train_data(args)['ID'].values
    print(f'number of images in train set: {len(train_images)}')
    print('*' * 100)
    description_file = os.path.join(args.data_path, 'train/simple_llava_description_large_train_set.csv')
    if os.path.exists(description_file):
        return pd.read_csv(description_file)
    with open(description_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    pipe = get_model()
    processed_images = set()
    for image_url in train_images:
        if image_url in processed_images:
            continue
        processed_images.add(image_url)
        image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
        prompt = f"USER:<image>\nDescribe the image in detail.\nASSISTANT:"
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 300})
        description = outputs[0]['generated_text'].split('ASSISTANT: ')[-1]
        print(f'output of image {image_url} is {description}')
        print('-' * 80)
        pair = [image_url, description]
        with open(description_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)


def get_llm_generated_prompt(args):
    test_images = get_test_data(args)['ID'].values
    print(f'number of images in test set: {len(test_images)}')
    print('*' * 100)
    description_file = os.path.join(args.data_path, 'train/llama3_FT_generated_description_test_set.csv')
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
    # if os.path.exists(product_file):
    #     return pd.read_csv(product_file)
    with open(product_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    negative_file = os.path.join(args.data_path, 'train/negative_prompt_train_set.csv')
    # if os.path.exists(negative_file):
    #     return pd.read_csv(negative_file)
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
    # descriptions = get_descriptions(args)
    descriptions = get_llm_generated_prompt(args)
    # get_negative_descriptions(args)

