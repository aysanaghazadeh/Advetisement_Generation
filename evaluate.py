import os.path
import pandas as pd
import json
from collections import Counter
from Evaluation.metrics import Metrics, PersuasivenessMetric
from configs.inference_config import get_args
from util.data.mapping import TOPIC_MAP as topic_map


TOPIC_FILE = '../Data/PittAd/train/Topics_train.json'
RESULT_FILE = '../experiments/results/AR_PixArt_20240505_231631.csv'


def get_topic_based_results():
    topics_data = json.load(open(TOPIC_FILE))
    all_topics = [topic for topics in topics_data.values() for topic in set(topics)]
    topic_counter = Counter(all_topics)
    most_common_topics = [topic for topic, count in topic_counter.most_common(10)]
    results = pd.read_csv(RESULT_FILE).values
    FIDs = {}
    CLIP_scores = {}
    image_topics = json.load(open(TOPIC_FILE))
    for row in results:
        image_url = row[0]
        FID = row[7]
        CLIP_score = row[5]
        topics = image_topics[image_url]
        for topic in topics:
            if topic in topic_map and topic in most_common_topics:
                topic_string = '-'.join(topic_map[topic])
                if topic_string in FIDs:
                    FIDs[topic_string].append(FID)
                    CLIP_scores[topic_string].append(CLIP_score)
                else:
                    FIDs[topic_string] = [FID]
                    CLIP_scores[topic_string] = [CLIP_score]

    for topic in FIDs:
        print(f'Average FID for {topic} is: {sum(FIDs[topic]) / len(FIDs[topic])}')
        print(f'Average CLIP-score for {topic} is: {sum(CLIP_scores[topic]) / len(CLIP_scores[topic])}')
        print('*' * 80)


def evaluate_results(metrics, args):
    results = pd.read_csv(RESULT_FILE).values
    FIDs = []
    CLIP_scores = []
    QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    for row in results:
        image_url = row[0]
        FID = row[-1]
        CLIP_score = row[5]
        action_reason = QA[image_url][0]
        original_image_url = os.path.join(args.data_path, args.test_set_images, image_url)
        original_image_text_score = metrics.get_text_image_CLIP_score(original_image_url, action_reason, args)['text']
        if original_image_text_score > 0.23:
            FIDs.append(FID)
            CLIP_scores.append(CLIP_score)
        else:
            print(f'Text image score for image {image_url} is {original_image_text_score}')
            print('-' * 100)
    print(f'number of examples: {len(FIDs)}')
    print(f'Average FID is: {sum(FIDs) / len(FIDs)}')
    print(f'Average CLIP-score is: {sum(CLIP_scores) / len(CLIP_scores)}')
    print('*' * 80)


def evaluate_persuasiveness():
    persuasiveness = PersuasivenessMetric()
    saving_path = RESULT_FILE.replace('.csv', '.json')
    print(saving_path)
    results = pd.read_csv(RESULT_FILE).values
    persuasiveness_scores = {}
    for row in results:
        try:
            image_url = row[0]
            generated_image_path = row[3]
            persuasiveness_score = persuasiveness.get_persuasiveness_score(generated_image_path)
            print(f'persuasiveness score of the image {image_url} is {persuasiveness_score} out of 10')
            print('*' * 80)
            persuasiveness_scores[image_url] = persuasiveness_score
        except:
            pass

    print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
    with open(saving_path, "w") as outfile:
        json.dump(persuasiveness_scores, outfile)


if __name__ == '__main__':
    args = get_args()
    metrics = Metrics(args)
    # evaluate_results(metrics, args)
    # get_topic_based_results()
    evaluate_persuasiveness()
