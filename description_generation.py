from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline
from util.data.trian_test_split import get_test_data
from PIL import Image
import os
import csv
import pandas as pd
from configs.inference_config import get_args


def get_model():
    # Load model directly
    model_id = "llava-hf/llava-1.5-13b-hf"
    pipe = pipeline("image-to-text", model=model_id, device_map='auto')
    return pipe


def get_descriptions(args):
    test_images = get_test_data(args)['ID'].values
    print(f'number of images in test set: {len(test_images)}')
    print('*' * 100)
    description_file = os.path.join(args.data_path, 'train/simple_llava_description_test_set.csv')
    if os.path.exists(description_file):
        return pd.read_csv(description_file)
    with open(description_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    pipe = get_model()
    processed_images = set()
    for image_url in test_images:
        if image_url in processed_images:
            continue
        processed_images.add(image_url)
        image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
        prompt = f"USER:<image>\nDescribe the image in detail.\nASSISTANT:"
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 512})
        description = outputs[0]['generated_text'].split('ASSISTANT: ')[-1]
        print(f'output of image {image_url} is {description}')
        print('-' * 80)
        pair = [image_url, description]
        with open(description_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)

if __name__ == '__main__':
    args = get_args()
    descriptions = get_descriptions(args)

