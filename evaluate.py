import json
import os.path
import random
from PIL import Image
import pandas as pd
from collections import Counter
from Evaluation.metrics import *
from configs.evaluation_config import get_args
from util.data.mapping import TOPIC_MAP as topic_map
from model.pipeline import AdvertisementImageGeneration
from Evaluation.action_reason_evaluation import ActionReasonLlava
import csv
from util.data.trian_test_split import get_test_data
from datasets import load_dataset


class Evaluation:
    def __init__(self, metrics, args):
        self.metrics = metrics
        if args.evaluation_type in ['creativity', 'persuasiveness_creativity'] and args.image_generation:
            self.image_generator = AdvertisementImageGeneration(args)
        if args.evaluation_type == 'action_reason_llava':
            self.ar_llava = ActionReasonLlava(args)
        if args.evaluation_type == 'whoops_llava':
            self.whoops = Whoops(args)

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
    def evaluate_persuasiveness_alignment(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', '_persuasiveness_alignment.json')
        print(saving_path)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        persuasiveness_alignment_scores = {}
        for row in results:
            image_url = row[0]
            generated_image_path = row[3]
            persuasiveness_alignment_score = persuasiveness.get_persuasiveness_alignment(generated_image_path)
            print(f'persuasiveness score of the image {image_url} is {persuasiveness_alignment_score} out of 5')
            print('*' * 80)
            persuasiveness_alignment_scores[image_url] = persuasiveness_alignment_score

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path, "w") as outfile:
                json.dump(persuasiveness_alignment_scores, outfile)

    @staticmethod
    def evaluate_persuasiveness(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', '_persuasiveness.json')
        print(saving_path)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        persuasiveness_scores = {}
        for row in results:
            image_url = row[0]
            generated_image_path = row[3]
            persuasiveness_score = persuasiveness.get_persuasiveness_score(generated_image_path)
            print(f'persuasiveness score of the image {image_url} is {persuasiveness_score} out of 5')
            print('*' * 80)
            persuasiveness_scores[image_url] = persuasiveness_score

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path, "w") as outfile:
                json.dump(persuasiveness_scores, outfile)

    @staticmethod
    def evaluate_action_reason_aware_persuasiveness(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               'action_reason_aware_persuasiveness.json')
        print(saving_path)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        persuasiveness_scores = {}
        for row in results:
            image_url = row[0]
            generated_image_path = row[3]
            persuasiveness_score = persuasiveness.get_action_reason_aware_persuasiveness_score(generated_image_path)
            print(
                f'action reason aware persuasiveness score of the image {image_url} is {persuasiveness_score} out of 5')
            print('*' * 80)
            persuasiveness_scores[image_url] = persuasiveness_score

            with open(saving_path, "w") as outfile:
                json.dump(persuasiveness_scores, outfile)

    @staticmethod
    def evaluate_data_persuasiveness(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, 'persuasiveness_2.json')
        print(saving_path)
        root_directory = os.path.join(args.data_path, 'train_images_total')
        persuasiveness_scores = {}
        # image_list = [
        #     "60880.jpg",
        #     "100170.jpg",
        #     "65170.jpg",
        #     "86630.jpg",
        #     "13470.jpg",
        #     "95690.jpg",
        #     "121490.jpg",
        #     "12080.jpg",
        #     "11160.jpg",
        #     "66680.jpg",
        #     "66340.jpg",
        #     "132590.jpg",
        #     "133370.jpg",
        #     "100270.jpg",
        #     "133370.jpg",
        #     "36080.jpg",
        #     "100520.jpg",
        #     "26400.jpg",
        #     "53640.jpg",
        #     "158750.jpg",
        #     "69420.jpg",
        #     "127260.jpg",
        #     "134110.jpg",
        #     "60380.jpg",
        #     "58030.jpg",
        #     "38390.jpg",
        #     "132590.jpg"
        # ]

        for dirpath, _, filenames in os.walk(root_directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                image_url = os.path.relpath(filepath, root_directory)
                # if filename not in image_list:
                #     continue
                persuasiveness_score = persuasiveness.get_persuasiveness_score(filepath)
                persuasiveness_scores[image_url] = persuasiveness_score
                print(f'persuasiveness score for image {image_url} is {persuasiveness_score}')
                # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
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
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            image_path = os.path.join(args.data_path, args.product_images, image_url.split('.')[0])
            if os.path.exists(image_path):
                continue
            else:
                print(image_path)
                os.makedirs(image_path, exist_ok=True)
            for i in range(3):
                image, prompt = self.image_generator(image_url)
                image.save(os.path.join(image_path, str(i) + '.jpg'))

    def evaluate_creativity(self, args):
        results = pd.read_csv(os.path.join(args.result_path, args.result_file))
        # baseline_result_file = 'LLAMA3_generated_prompt_PixArt_20240508_084149.csv'
        # baseline_results = pd.read_csv(os.path.join(args.result_path, baseline_result_file)).image_url.values
        self.generate_product_images(args, results)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', '.json')
        creativity_scores = {}
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            # if image_url not in baseline_results:
            #     continue
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

    def evaluate_persuasiveness_creativity(self, args):
        results = pd.read_csv(os.path.join(args.result_path, args.result_file))
        # baseline_result_file = 'LLAMA3_generated_prompt_PixArt_20240508_084149.csv'
        # baseline_results = pd.read_csv(os.path.join(args.result_path, baseline_result_file)).image_url.values
        self.generate_product_images(args, results)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               args.text_alignment_file.split('_')[-1].split('.')[0] +
                                                                               '_creativity.json')
        creativity_scores = {}
        image_text_alignment_scores = json.load(open(os.path.join(args.result_path,
                                                                  args.text_alignment_file)))
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            print(f'image url: {image_url}')
            # if image_url not in baseline_results:
            #     continue
            no_product_image_count = 0
            image_text_alignment_score = image_text_alignment_scores[image_url]
            generated_image_path = results.generated_image_url.values[row]
            directory = os.path.join(args.data_path, args.product_images, image_url.split('.')[0])
            product_image_files = os.listdir(directory)
            if len(product_image_files) == 0:
                no_product_image_count += 1
                continue
            product_image_paths = [os.path.join(args.data_path, args.product_images, image_url.split('.')[0], file)
                                   for file in product_image_files]
            creativity_scores[image_url] = self.metrics.get_persuasiveness_creativity_score(text_alignment_score=image_text_alignment_score,
                                                                                            generated_image_path=generated_image_path,
                                                                                            product_image_paths=product_image_paths,
                                                                                            args=args)
            print(
                f'creativity score for image {image_url} is {creativity_scores[image_url]}')
            with open(saving_path, "w") as outfile:
                json.dump(creativity_scores, outfile)
        print(f'number of images with no product image is: {no_product_image_count}')

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
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', f'{args.VLM}_image_text_alignment.json')
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

    def evaluate_whoops_llava(self, args):
        results = {'acc@1': 0}
        fieldnames = ['id', 'acc@1']
        # csv_file_path = os.path.join(args.result_path, ''.join(['action_reason_llava_', args.description_file]))
        csv_file_path = os.path.join(args.result_path, 'whoops_llava_without_description.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        # dataset = load_dataset('nlphuji/whoops',
        #                        cache_dir=None,
        #                        use_auth_token='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')
        QA_file = json.load(open(os.path.join(args.data_path, 'whoops_caption.json')))
        # if os.path.exists(QA_file):
        QAs = json.load(QA_file)
        # else:
        #     QAs = {}
        #     for i in range(len(dataset['test'])):
        #         options = set([dataset['test']['selected_caption'][i]])
        #         while len(options) < 15:
        #             j = random.randint(0, len(dataset['test']))
        #             options.add(dataset['test']['selected_caption'][j])
        #         QAs[i] = [[dataset['test']['selected_caption'][i]], list(options)]
        #     with open(QA_file, "w") as outfile:
        #         json.dump(QAs, outfile)
        for i in QAs:
            image = Image.open(os.path.join(args.data_path, 'whoops_images', f'{i}.png'))
            answers = self.whoops.get_prediction(image, QAs[i])
            result = 1 if answers[0] in QAs[i][0] else 0
            row = {}
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row['id'] = i
                row['acc@1'] = result
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

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

    @staticmethod
    def evaluate_original_images(args):
        persuasiveness = PersuasivenessMetric(args)
        QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        metrics = Metrics(args)
        saving_path_persuasiveness = os.path.join(args.result_path,
                                                  'original_images_persuasiveness.json')
        print(f'persuasiveness: {saving_path_persuasiveness}')
        saving_path_persuasiveness_alignment = os.path.join(args.result_path,
                                                            'original_images_persuasiveness_alignment.json')
        print(f'persuasiveness alignment: {saving_path_persuasiveness_alignment}')
        saving_path_ar_aware_persuasiveness = os.path.join(args.result_path,
                                                           'original_images_action_reason_aware_persuasiveness.json')
        print(f'action-reason aware persuasiveness: {saving_path_ar_aware_persuasiveness}')
        saving_path_image_text = os.path.join(args.result_path,
                                              'original_images_image_text.json')
        print(f'image-text: {saving_path_image_text}')
        saving_path_image_action = os.path.join(args.result_path,
                                                'original_images_image_action.json')
        print(f'image-action: {saving_path_image_action}')
        saving_path_image_reason = os.path.join(args.result_path,
                                                'original_images_image_reason.json')
        print(f'image-reason: {saving_path_image_reason}')
        test_images = list(get_test_data(args).ID.values)[:1340]
        persuasiveness_scores = {}
        persuasiveness_alignment_scores = {}
        ar_aware_persuasiveness_scores = {}
        image_text_scores = {}
        image_action_scores = {}
        image_reason_scores = {}
        for image_url in test_images:
            image_path = os.path.join(args.data_path, args.test_set_images, image_url)
            action_reason_statements = QAs[image_url][0]
            persuasiveness_score = persuasiveness.get_persuasiveness_score(image_path)
            ar_aware_persuasiveness_score = persuasiveness.get_action_reason_aware_persuasiveness_score(image_path)
            persuasiveness_alignment_score = persuasiveness.get_persuasiveness_alignment(image_path)
            clip_image_text = metrics.get_text_image_CLIP_score(image_path, action_reason_statements, args)
            image_text_score = clip_image_text['text']
            image_action_score = clip_image_text['action']
            image_reason_score = clip_image_text['reason']
            print(f'persuasiveness score of the image {image_url} is {persuasiveness_score} out of 5')
            print(f'persuasiveness alignment score of the image {image_url} is {persuasiveness_alignment_score} out of 5')
            print(f'action-reason aware persuasiveness score of the image {image_url} is {ar_aware_persuasiveness_score} out of 5')
            print(f'image-text score of the image {image_url} is {image_text_score}')
            print(f'image-action score of the image {image_url} is {image_action_score}')
            print(f'image-reason score of the image {image_url} is {image_reason_score}')
            print('*' * 80)
            persuasiveness_scores[image_url] = persuasiveness_score
            ar_aware_persuasiveness_scores[image_url] = ar_aware_persuasiveness_score
            persuasiveness_alignment_scores[image_url] = persuasiveness_alignment_score
            image_text_scores[image_url] = image_text_score
            image_reason_scores[image_url] = image_reason_score
            image_action_scores[image_url] = image_action_score

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path_persuasiveness, "w") as outfile:
                json.dump(persuasiveness_scores, outfile)
            with open(saving_path_ar_aware_persuasiveness, "w") as outfile:
                json.dump(ar_aware_persuasiveness_scores, outfile)
            with open(saving_path_persuasiveness_alignment, "w") as outfile:
                json.dump(persuasiveness_alignment_scores, outfile)
            with open(saving_path_image_text, "w") as outfile:
                json.dump(image_text_scores, outfile)
            with open(saving_path_image_action, "w") as outfile:
                json.dump(image_action_scores, outfile)
            with open(saving_path_image_reason, "w") as outfile:
                json.dump(image_reason_scores, outfile)

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
