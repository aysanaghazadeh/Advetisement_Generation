import os.path
import pandas as pd
import json
from collections import Counter
from Evaluation.metrics import *
from configs.evaluation_config import get_args
from util.data.mapping import TOPIC_MAP as topic_map
from model.pipeline import AdvertisementImageGeneration
from Evaluation.action_reason_evaluation import ActionReasonLlava
import csv


class Evaluation:
    def __init__(self, metrics, args):
        self.metrics = metrics
        if args.evaluation_type == 'creativity' and args.image_generation:
            self.image_generator = AdvertisementImageGeneration(args)
        if args.evaluation_type == 'action_reason_llava':
            self.ar_llava = ActionReasonLlava(args)

    @staticmethod
    def evaluate_topic_based(args):
        topics_data = json.load(open(os.path.join(args.data_path, 'train', 'Topics_train.json')))
        all_topics = [topic for topics in topics_data.values() for topic in set(topics)]
        topic_counter = Counter(all_topics)
        most_common_topics = [topic for topic, count in topic_counter.most_common(10)]
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        FIDs = {}
        CLIP_scores = {}
        image_topics = json.load(open(os.path.join(args.data_path, 'train', 'Topics_train.json')))
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

    @staticmethod
    def evaluate_persuasiveness(args):
        persuasiveness = PersuasivenessMetric()
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', '.json')
        print(saving_path)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
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

    @staticmethod
    def evaluate_data_persuasiveness(args):
        persuasiveness = PersuasivenessMetric()
        saving_path = os.path.join(args.result_path, 'persuasiveness.json')
        print(saving_path)
        root_directory = os.path.join(args.data_path, args.test_set_images)
        persuasiveness_scores = {}
        for dirpath, _, filenames in os.walk(root_directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                image_url = os.path.relpath(filepath, root_directory)
                persuasiveness_score = persuasiveness.get_persuasiveness_score(filepath)
                persuasiveness_scores[image_url] = persuasiveness_score
                print(f'persuasiveness score for image {image_url} is {persuasiveness_score}')
        print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
        with open(saving_path, "w") as outfile:
            json.dump(persuasiveness_scores, outfile)

    def evaluate_sampled_results(self, args):
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        FIDs = []
        CLIP_scores = []
        QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        for row in results:
            image_url = row[0]
            FID = row[-1]
            CLIP_score = row[5]
            action_reason = QA[image_url][0]
            original_image_url = os.path.join(args.data_path, args.test_set_images, image_url)
            original_image_text_score = self.metrics.get_text_image_CLIP_score(original_image_url, action_reason, args)[
                'text']
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

    def generate_product_images(self, args, results):
        baseline_result_file = 'LLAMA3_generated_prompt_PixArt_20240508_084149.csv'
        baseline_results = pd.read_csv(os.path.join(args.result_path, baseline_result_file)).image_url.values
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            image_path = os.path.join(args.data_path, args.product_images, image_url.split('.')[0])
            if os.path.exists(image_path) or image_url not in baseline_results:
                continue
            else:
                print(image_path)
                os.makedirs(image_path, exist_ok=True)
            for i in range(3):
                image, prompt = self.image_generator(image_url)
                image.save(os.path.join(image_path, str(i) + '.jpg'))

    def evaluate_creativity(self, args):
        results = pd.read_csv(os.path.join(args.result_path, args.result_file))
        baseline_result_file = 'LLAMA3_generated_prompt_PixArt_20240508_084149.csv'
        baseline_results = pd.read_csv(os.path.join(args.result_path, baseline_result_file)).image_url.values
        self.generate_product_images(args, results)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', '.json')
        creativity_scores = {}
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            if image_url not in baseline_results:
                continue
            generated_image_path = results.generated_image_url.values[row]
            action_reason = results.action_reason.values[row]
            directory = os.path.join(args.data_path, args.product_images, image_url.split('.')[0])
            product_image_files = os.listdir(directory)
            product_image_paths = [os.path.join(args.data_path, args.product_images, image_url.split('.')[0], file)
                                   for file in product_image_files]
            creativity_scores[image_url] = self.metrics.get_creativity_scores(text_description=action_reason,
                                                                              generated_image_path=generated_image_path,
                                                                              product_image_paths=product_image_paths,
                                                                              args=args)
            print(
                f'creativity score for image {image_url} is {creativity_scores[image_url]}')
            with open(saving_path, "w") as outfile:
                json.dump(creativity_scores, outfile)

    def evaluate_action_reason_llava(self, args):
        results = {'acc@1': 0, 'acc@2': 0, 'acc@3': 0,
                   'p@1': 0, 'p@2': 0, 'p@3': 0}
        fieldnames = ['acc@1', 'acc@2', 'acc@3', 'p@1', 'p@2', 'p@3', 'id']
        # csv_file_path = os.path.join(args.result_path, ''.join(['action_reason_llava_', args.description_file]))
        csv_file_path = os.path.join(args.result_path, 'action_reason_llava_without_description.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        for image_url in QAs:
            result = self.ar_llava.evaluate_image(image_url)
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = result
                row['id'] = image_url
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

    @staticmethod
    def evaluate_image_text_alignment(args):
        QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        results = pd.read_csv(os.path.join(args.result_path, args.result_file))
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', '_image_text_alignment.json')
        if os.path.exists(saving_path):
            image_text_alignment_scores = json.load(open(saving_path))
        else:
            image_text_alignment_scores = {}
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            if image_url in image_text_alignment_scores:
                continue
            print(f'process on image {image_url} started:')
            generated_image_path = results.generated_image_url.values[row]
            action_reasons = QA[image_url][0]
            image_text_alignment_scores[image_url] = metrics.get_image_text_alignment_scores(action_reasons,
                                                                                             generated_image_path,
                                                                                             args)
            print(
                f'image text alignment score is {image_text_alignment_scores[image_url]}')
            print('-' * 80)
            with open(saving_path, "w") as outfile:
                json.dump(image_text_alignment_scores, outfile)

    @staticmethod
    def evaluate_image_text_ranking(args):
        QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        results = pd.read_csv(os.path.join(args.result_path, args.result_file))
        results_baseline = pd.read_csv(os.path.join(args.result_path, 'AR_PixArt_20240505_231631.csv'))
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', '_image_text_ranking.json')
        if os.path.exists(saving_path):
            image_text_ranking = json.load(open(saving_path))
        else:
            image_text_ranking = {}
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            # if image_url in image_text_ranking:
            #     continue
            if image_url not in results_baseline.image_url.values:
                continue
            print(f'process on image {image_url} started:')
            first_generated_image_path = results.generated_image_url.values[row]
            second_generated_image_path = results.generated_image_url.values[
                results_baseline[results_baseline['image_url'] == image_url].index.tolist()][0]
            action_reasons = QA[image_url][0]
            image_text_ranking[image_url] = metrics.get_image_text_ranking(action_reasons,
                                                                           first_generated_image_path,
                                                                           second_generated_image_path,
                                                                           args)
            print(
                f'image text ranking score is {image_text_ranking[image_url]}')
            print('-' * 80)
            with open(saving_path, "w") as outfile:
                json.dump(image_text_ranking, outfile)

    def evaluate(self, args):
        evaluation_name = 'evaluate_' + args.evaluation_type
        print(f'evaluation method: {evaluation_name}')
        evaluation_method = getattr(self, evaluation_name)
        evaluation_method(args)


if __name__ == '__main__':
    args = get_args()
    metrics = Metrics(args)
    evaluation = Evaluation(metrics, args)
    evaluation.evaluate(args)
