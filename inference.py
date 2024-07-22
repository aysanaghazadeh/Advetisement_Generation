from configs.inference_config import get_args
from model.pipeline import AdvertisementImageGeneration
from Evaluation.metrics import Metrics
import json
import os
from datetime import datetime
import csv
from util.data.trian_test_split import get_test_data
import random


def get_QA(args):
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    return QA


def save_image(args, filename, image, experiment_datetime):
    subdirectory = filename.split('/')[0]
    if args.text_input_type == 'AR':
        text_input = 'AR'
    elif args.text_input_type == 'LLM':
        text_input = '_'.join([args.LLM, 'generated_prompt'])
    else:
        text_input = args.description_file.split('/')[-1]
    directory = os.path.join(args.result_path,
                             'generated_images',
                             experiment_datetime,
                             '_'.join([text_input, args.T2I_model]),
                             subdirectory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    image.save(os.path.join(args.result_path,
                            'generated_images',
                            experiment_datetime,
                            '_'.join([text_input, args.T2I_model]),
                            filename))


def save_results(args, prompt, action_reason, filename, experiment_datetime, scores, topics):
    if args.text_input_type == 'AR':
        text_input = 'AR'
    elif args.text_input_type == 'LLM':
        text_input = '_'.join([args.LLM, 'generated_prompt'])
    else:
        text_input = args.description_file.split('/')[-1]
    directory = os.path.join(args.result_path, 'results')
    if not os.path.exists(directory):
        os.makedirs(directory)

    csv_file_name = '_'.join([text_input, args.T2I_model, experiment_datetime])
    csv_file_name = f'{csv_file_name}.csv'
    csv_file = os.path.join(directory, csv_file_name)
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(['image_url',
                             'action_reason',
                             'T2I_prompt',
                             'generated_image_url',
                             'topics',
                             'clip_image_image_score',
                             'clip_image_text_score',
                             'clip_image_action_score',
                             'clip_image_text_score',
                             'FID_score'])
    generated_image_url = os.path.join(args.result_path,
                                       'generated_images',
                                       experiment_datetime,
                                       '_'.join([text_input, args.T2I_model]),
                                       filename)
    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([filename, action_reason, prompt, generated_image_url, topics] + scores)


def evaluate(metrics, args, action_reason, filename, experiment_datetime):
    if args.text_input_type == 'AR':
        text_input = 'AR'
    elif args.text_input_type == 'LLM':
        text_input = '_'.join([args.LLM, 'generated_prompt'])
    else:
        text_input = args.description_file.split('/')[-1]
    generated_image_path = os.path.join(args.result_path,
                                        'generated_images',
                                        experiment_datetime,
                                        '_'.join([text_input, args.T2I_model]),
                                        filename)
    real_image_path = os.path.join(args.data_path, args.test_set_images, filename)

    text_description = action_reason
    scores = metrics.get_scores(text_description, generated_image_path, real_image_path, args)
    return scores


def process_action_reason(action_reasons):
    return '\n'.join([f'({i}) {statement}' for i, statement in enumerate(action_reasons)])


def generate_images(args):
    test_set = get_test_data(args)
    AdImageGeneration = AdvertisementImageGeneration(args)
    QA = get_QA(args)
    experiment_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = Metrics(args)
    print(f'experiment started at {experiment_datetime}')
    test_set_image_url = list(test_set['ID'].values)
    test_set_image_url = test_set_image_url[:1340]
    for filename, content in QA.items():
        if filename not in test_set_image_url:
        # if filename not in test_set:
            continue
        topics = test_set.loc[test_set['ID'] == filename]['topic'].values
        # topics = []
        # action_reasons = content[0]
        action_reasons = []
        for AR in content[1]:
            if AR not in content[0]:
                action_reasons.append(AR)
                break
        image, prompt = AdImageGeneration(filename)
        save_image(args, filename, image, experiment_datetime)
        scores = evaluate(metrics, args, action_reasons, filename, experiment_datetime)
        save_results(args, prompt, action_reasons, filename, experiment_datetime, list(scores.values()), topics)
        print(f'image url: {filename}')
        print(f'topics: {topics}')
        print(f'action-reason statements: {process_action_reason(action_reasons)}')
        print(f'scores: {scores}')
        print('-' * 20)
    finish_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'experiment ended at {finish_datetime}')


if __name__ == '__main__':
    args = get_args()
    generate_images(args)
