import json
import os.path
import random
from PIL import Image
import pandas as pd
from collections import Counter
from jinja2 import Environment, FileSystemLoader
from Evaluation.metrics import *
from configs.evaluation_config import get_args
from util.data.mapping import TOPIC_MAP as topic_map
from model.pipeline import AdvertisementImageGeneration
from Evaluation.action_reason_evaluation import ActionReasonVLM
import csv
from util.data.trian_test_split import get_test_data
from LLMs.LLM import LLM


@staticmethod
def evaluate_action_reason_LLM(args):
    def parse_options(options):
        return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

    def get_prediction(prompt, options, pipe):
        answers = []
        output = pipe(prompt)
        print(output)
        answer = ''.join(i for i in output if i.isdigit())
        if answer != '':
            answers.append(int(answer))
        predictions = set()
        for ind in answers:
            if len(options) > ind:
                predictions.add(options[ind])
                if len(predictions) == 3:
                    break
        answers = list(predictions)
        return answers

    results = {'acc@1': 0}
    fieldnames = ['id', 'acc@1']
    csv_file_path = os.path.join(args.result_path,
                                 f'whoops_caption_new_hard_{args.description_type}_{args.VLM}_description_{args.LLM}_version_2.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    QA_file = os.path.join(args.data_path, f'train/whoops_caption_new_hard.json')
    QAs = json.load(open(QA_file))
    pipe = LLM(args)
    descriptions = pd.read_csv(os.path.join(args.data_path, 'train',
                                            f'{args.description_type}_{args.VLM}_description_{args.task}_version_2.csv'))
    for i in QAs:
        image_url = f'{i}.png'
        description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
        options = QAs[i][1]
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.VLM_prompt)
        data = {'description': description, 'options': parse_options(options)}
        prompt = template.render(**data)
        answers = get_prediction(prompt, options, pipe)
        print(answers)
        if len(answers) == 0:
            result = 0
        else:
            if len(QAs[i]) == 3:
                result = 1 if answers[0] in QAs[i][1] else 0
            else:
                result = 1 if answers[0] in QAs[i][0] else 0
        print(result)
        row = {}
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row['id'] = i
            row['acc@1'] = result
            writer.writerow(list(row.values()))

        for metric in results:
            results[metric] += result
    for metric in results:
        print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')
