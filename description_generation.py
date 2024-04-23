from transformers import AutoProcessor, AutoModelForCausalLM
from util.data.trian_test_split import get_test_data
from PIL import Image
import os
import csv
import pandas as pd


def get_model():
    # Load model directly
    processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-13b", device_map='auto')
    model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-13b", device_map='auto')
    return processor, model


def get_descriptions(args):
    test_images = get_test_data(args)['ID'].values
    description_file = os.path.join(args.data_path, 'train/simple_llava_description_test_set.csv')
    if os.path.exists(description_file):
        return pd.read_csv(description_file)
    with open(description_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['ID', 'description'])
    model, processor = get_model()
    for image_url in test_images:
        image = Image.open(os.path.join(args.data_path, args.test_set_images, image_url))
        prompt = f"USER:<image>\nDescribe the image in detail.\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        generate_ids = model.generate(**inputs, max_new_tokens=512)
        description = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pair = [image_url, description]
        with open(description_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pair)

    return pd.read_csv(description_file)






