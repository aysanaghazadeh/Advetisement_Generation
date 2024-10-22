import os.path
import pandas as pd
from collections import Counter

import requests

from Evaluation.metrics import *
from configs.evaluation_config import get_args
from util.data.mapping import TOPIC_MAP as topic_map
from model.pipeline import AdvertisementImageGeneration
from Evaluation.action_reason_evaluation import ActionReasonVLM
import csv
from util.data.trian_test_split import get_test_data
from LLMs.LLM import LLM
from io import BytesIO


class Evaluation:
    def __init__(self, args):
        if args.evaluation_type in ['creativity', 'persuasiveness_creativity'] and args.image_generation:
            self.metrics = Metrics(args)
            self.image_generator = AdvertisementImageGeneration(args)
        if args.evaluation_type == 'action_reason_VLM':
            self.ar_VLM = ActionReasonVLM(args)
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
    def evaluate_multi_question_persuasiveness(args):
        score_metricts = Metrics(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               f'_{args.VLM}_multi_question_persuasiveness_new_prompts.json')
        print(saving_path)
        print(args.result_path)
        print(args.result_file)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        print(results)
        persuasiveness_scores = {}
        for row in results:
            image_url = row[0]
            print(image_url)
            generated_image_path = row[3]
            persuasiveness_score = score_metricts.get_multi_question_persuasiveness(generated_image_path)
            print(f'persuasiveness scores of the image {image_url} is: \n {persuasiveness_score}')
            print('*' * 80)
            persuasiveness_scores[image_url] = list(persuasiveness_score.values())

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path, "w") as outfile:
                json.dump(persuasiveness_scores, outfile)

    @staticmethod
    def evaluate_multi_question_persuasiveness_ranking(args):
        score_metrics = Metrics(args)
        saving_path = os.path.join(args.result_path, f'{args.VLM}_multi_question_persuasiveness_ranking.json')
        print(saving_path)
        print(args.result_path)
        print(args.result_file)
        results1 = pd.read_csv(os.path.join(args.result_path,
                                            'LLM_input_LLAMA3_instruct_FTFalse_PSA.csv_AuraFlow_20240925_112154.csv'))
        results2 = pd.read_csv(os.path.join(args.result_path,
                                            'AR_AuraFlow_20240924_210335.csv'))
        persuasiveness_scores = {}
        for row in results1.values:
            image_url = row[0]
            if image_url not in results2.image_url.values:
                continue
            print(image_url)
            generated_image_path1 = row[3]
            generated_image_path2 = results2.loc[results2['image_url'] == image_url]['generated_image_url'].values[0]
            persuasiveness_score = score_metrics.get_multi_question_persuasiveness_ranking(generated_image_path1,
                                                                                            generated_image_path2)
            print(f'persuasiveness scores of the image {image_url} is: \n {persuasiveness_score}')
            print('*' * 80)
            persuasiveness_scores[image_url] = list(persuasiveness_score.values())

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path, "w") as outfile:
                json.dump(persuasiveness_scores, outfile)

    @staticmethod
    def evaluate_llm_multi_question_persuasiveness_ranking(args):
        score_metrics = Metrics(args)
        saving_path = os.path.join(args.result_path, f'{args.VLM}_llm_multi_question_persuasiveness_ranking_new.json')
        print(saving_path)
        print(args.result_path)
        print(args.result_file)
        results2 = pd.read_csv(os.path.join(args.result_path,
                                            'LLM_input_LLAMA3_instruct_FTFalse_PSA.csv_AuraFlow_20240925_112154.csv'))
        results1 = pd.read_csv(os.path.join(args.result_path,
                                            'AR_AuraFlow_20240924_210335.csv'))
        descriptions1 = pd.read_csv(os.path.join(args.result_path,
                                                'IN_InternVL_LLM_input_LLAMA3_instruct_FTFalse_PSA_AuraFlow_20240925_112154_description_old.csv'))
        descriptions2 = pd.read_csv(os.path.join(args.result_path,
                                                'IN_InternVL_AR_AuraFlow_20240924_210335_description.csv'))
        persuasiveness_scores = {}
        for row in results1.values:
            image_url = row[0]
            if image_url not in results2.image_url.values:
                continue
            print(image_url)
            # generated_image_path1 = row[3]
            # generated_image_path2 = results2.loc[results2['image_url'] == image_url]['generated_image_url'].values[0]
            generated_image1 = descriptions1.loc[descriptions1['ID'] == image_url, 'description'].values[0]
            generated_image2 = descriptions2.loc[descriptions2['ID'] == image_url, 'description'].values[0]
            persuasiveness_score = score_metrics.get_llm_multi_question_persuasiveness_ranking(generated_image1,
                                                                                               generated_image2,
                                                                                               image_url)
            print(f'persuasiveness scores of the image {image_url} is: \n {persuasiveness_score}')
            print('*' * 80)
            persuasiveness_scores[image_url] = list(persuasiveness_score.values())

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path, "w") as outfile:
                json.dump(persuasiveness_scores, outfile)

    @staticmethod
    def evaluate_persuasiveness_alignment(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv', f'{args.VLM}_persuasiveness_alignment_10.json')
        print(saving_path)
        print(args.result_path)
        print(args.result_file)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        print(results)
        persuasiveness_alignment_scores = {}
        for row in results:
            image_url = row[0]
            print(image_url)
            generated_image_path = row[3]
            if args.VLM == 'GPT4v':
                persuasiveness_alignment_score = persuasiveness.get_GPT4v_persuasiveness_alignment(generated_image_path)
            else:
                persuasiveness_alignment_score = persuasiveness.get_persuasiveness_alignment(generated_image_path)
            print(f'persuasiveness score of the image {image_url} is {persuasiveness_alignment_score} out of 10')
            print('*' * 80)
            persuasiveness_alignment_scores[image_url] = persuasiveness_alignment_score

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path, "w") as outfile:
                json.dump(persuasiveness_alignment_scores, outfile)

    @staticmethod
    def evaluate_text_image_alignment(args):
        alignment_score_model = Metrics(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               f'{args.VLM}_text_image_alignment'
                                                                               f'_isFineTuned{args.fine_tuned}_3000.json')
        print(saving_path)
        print(args.result_path)
        print(args.result_file)
        action_reasons_all = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        descriptions = pd.read_csv(args.description_file)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        alignment_scores = {}
        for row in results:
            image_url = row[0]
            print(image_url)
            action_reasons = action_reasons_all[image_url][0]
            if image_url not in descriptions.ID.values:
                continue
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
            print(description)
            generated_image_message, alignment_score = alignment_score_model.get_text_image_alignment_score(action_reasons,
                                                                            description,
                                                                            args)
            print(alignment_score)
            # print(generated_image_message)
            print(f'text image alignment score of the image {image_url} is {alignment_score} out of 10')
            print('*' * 80)
            alignment_scores[image_url] = [generated_image_message, alignment_score]

            # print(f'average persuasiveness is {sum(persuasiveness_scores) / len(persuasiveness_scores)}')
            with open(saving_path, "w") as outfile:
                json.dump(alignment_scores, outfile)

    @staticmethod
    def evaluate_multi_question(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               f'{args.VLM}_multi_question_new.csv')
        fieldnames = [
            'image_url',
            # 'has_story',
            # 'is_unusual',
            'properties_score',
            'audience_score',
            'audiences',
            # 'memorability_score',
            'benefit_score',
            'appealing_score',
            'appealing_type',
            # 'maslow_pyramid_needs'
        ]
        with open(saving_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        for row in results:
            image_url = row[0]
            generated_image_path = row[3]
            if args.VLM == 'GPT4v':
                QA_pairs = persuasiveness.get_GPT4v_persuasiveness_alignment(generated_image_path)
            else:
                QA_pairs = persuasiveness.get_multi_question_evaluation(generated_image_path)
            print(f'The answers for image {image_url} is:')
            for question in QA_pairs:
                print(f'Answer of {question} question is: {QA_pairs[question]}')
            answers = list(QA_pairs.values())
            answers = [image_url] + answers
            with open(saving_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(answers)

    @staticmethod
    def evaluate_multi_question_score(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               f'{args.VLM}_multi_question_score.csv')
        fieldnames = [
            'image_url',
            'has_story',
            'is_unusual',
            'properties_score',
            'audience_score',
            # 'audiences',
            'memorability_score',
            'benefit_score',
            'appealing_score',
            'appealing_type',
            'maslow_pyramid_needs'
        ]
        with open(saving_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        for row in results:
            image_url = row[0]
            # generated_image_path = row[3]
            generated_image_path = os.path.join(args.result_path, args.test_set_images, image_url)
            if args.VLM == 'GPT4v':
                QA_pairs = persuasiveness.get_GPT4v_persuasiveness_alignment(generated_image_path)
            else:
                QA_pairs = persuasiveness.get_multi_question_score_evaluation(generated_image_path)
            print(f'The answers for image {image_url} is:')
            for question in QA_pairs:
                print(f'Answer of {question} question is: {QA_pairs[question]}')
            answers = list(QA_pairs.values())
            answers = [image_url] + answers
            with open(saving_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(answers)

    @staticmethod
    def evaluate_multi_question_ImageARG(args):
        data = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               f'ImageARG_'
                                                                               f'{args.VLM}_'
                                                                               f'multi_question_new.csv')
        fieldnames = [
            'image_url',
            # 'has_story',
            # 'is_unusual',
            # 'properties_score',
            # 'audience_score',
            # 'audiences',
            # 'memorability_score',
            # 'benefit_score',
            'appealing_score',
            'appealing_type',
            # 'maslow_pyramid_needs'
        ]
        with open(saving_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        for row in data:
            image_url = row['media_url']
            if 'persuasion_mode' not in row:
                continue
            try:
                # generated_image_path = requests.get(image_url, stream=True).raw
                response = requests.get(image_url)
                response.raise_for_status()
                generated_image_path = BytesIO(response.content)
                # Open the image with Pillow
                if args.VLM == 'GPT4v':
                    QA_pairs = persuasiveness.get_GPT4v_persuasiveness_alignment(generated_image_path)
                else:
                    QA_pairs = persuasiveness.get_multi_question_evaluation(generated_image_path)
                print(f'The answers for image {image_url} is:')
                for question in QA_pairs:
                    print(f'Answer of {question} question is: {QA_pairs[question]}')
                answers = list(QA_pairs.values())
                answers = [image_url] + answers
                with open(saving_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(answers)
            except:
            # print
                continue


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
    def evaluate_MLLM_alignment(args):
        Metric = Metrics(args)
        saving_path = os.path.join(args.result_path,
                                   '_'.join([
                                       args.VLM_prompt.replace('.jinja', ''),
                                       args.test_set_images.split('/')[-1],
                                       args.result_file.replace('.csv', 'MLLM_alignment_score.json')]))
        print(saving_path)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file)).values
        alignment_scores = {}
        for row in results:
            image_url = row[0]
            # generated_image_path = row[3]
            generated_image_path = os.path.join(args.result_path, args.test_set_images, image_url)
            alignment_score = Metric.get_MLLM_alignment_score(generated_image_path)
            print(
                f'action reason aware alignment score of the image {image_url} is {alignment_score} out of 5')
            print('*' * 80)
            alignment_scores[image_url] = alignment_score

            with open(saving_path, "w") as outfile:
                json.dump(alignment_scores, outfile)

    @staticmethod
    def evaluate_data_persuasiveness(args):
        persuasiveness = PersuasivenessMetric(args)
        saving_path = os.path.join(args.result_path, 'persuasiveness_2.json')
        print(saving_path)
        root_directory = os.path.join(args.data_path, 'train_images_total')
        persuasiveness_scores = {}

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
        metrics = Metrics(args)
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
            original_image_text_score = metrics.get_text_image_CLIP_score(original_image_url, action_reason, args)[
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
        topics = json.load(open(os.path.join(args.data_path, 'trian/Topics_train.json')))
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            image_topics = topics[image_url]
            for topic_id in image_topics:
                if topic_id in topic_map:
                    topic_names = topic_map[topic_id]
                else:
                    topic_names = [topic_id]
                for topic_name in topic_names:
                    image_path = os.path.join(args.data_path,
                                              args.product_images,
                                              topic_id,
                                              f'{topic_name.replace(" ", "_")}_{args.T2I_model}.jpg')
                    if os.path.exists(image_path):
                        continue
                    else:
                        print(image_path)
                        os.makedirs(image_path, exist_ok=True)
                    # for i in range(2):
                    prompt = f'image of {topic_name}'
                    image, prompt = self.image_generator(image_url, prompt)
                    # image.save(os.path.join(image_path, str(i) + '.jpg'))
                    image.save(image_path)

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
        metrics = Metrics(args)
        results = pd.read_csv(os.path.join(args.result_path, args.result_file))
        baseline_result_file = 'LLM_input_LLAMA3_instruct_FTFalse_PSA.csv_AuraFlow_20240925_112154.csv'
        baseline_results = pd.read_csv(os.path.join(args.result_path, baseline_result_file)).image_url.values
        self.generate_product_images(args, baseline_results)
        saving_path = os.path.join(args.result_path, args.result_file).replace('.csv',
                                                                               args.text_alignment_file.split('_')[-1].split('.')[0] +
                                                                               '_creativity.json')
        creativity_scores = {}
        image_text_alignment_scores = json.load(open(os.path.join(args.result_path,
                                                                  args.text_alignment_file)))
        topics = json.load(open(os.path.join(args.data_path, 'trian/Topics_train.json')))
        for row in range(len(results.values)):
            image_url = results.image_url.values[row]
            image_topics = topics[image_url]
            print(f'image url: {image_url}')
            if image_url not in baseline_results:
                continue
            no_product_image_count = 0
            image_text_alignment_score = image_text_alignment_scores[image_url]
            generated_image_path = results.generated_image_url.values[row]
            # directory = os.path.join(args.data_path, args.product_images, image_url.split('.')[0])
            # product_image_files = os.listdir(directory)
            # product_image_paths = [os.path.join(args.data_path, args.product_images, image_url.split('.')[0], file)
            #                        for file in product_image_files]
            product_image_paths = []
            for topic_id in image_topics:
                directory = os.path.join(args.data_path,
                                         args.product_images,
                                         topic_id)
                product_image_paths += [os.path.join(directory,
                                                     file) for file in os.listdir(directory)]
            if len(product_image_paths) == 0:
                no_product_image_count += 1
                continue

            creativity_scores[image_url] = metrics.get_persuasiveness_creativity_score(text_alignment_score=image_text_alignment_score,
                                                                                       generated_image_path=generated_image_path,
                                                                                       product_image_paths=product_image_paths,
                                                                                       args=args)
            print(
                f'creativity score for image {image_url} is {creativity_scores[image_url]}')
            with open(saving_path, "w") as outfile:
                json.dump(creativity_scores, outfile)
        print(f'number of images with no product image is: {no_product_image_count}')

    def evaluate_action_reason_VLM(self, args):
        results = {'acc@1': 0, 'acc@2': 0, 'acc@3': 0,
                   'p@1': 0, 'p@2': 0, 'p@3': 0}
        fieldnames = ['acc@1', 'acc@2', 'acc@3', 'p@1', 'p@2', 'p@3', 'id']
        # csv_file_path = os.path.join(args.result_path, ''.join(['action_reason_llava_', args.description_file]))
        csv_file_path = os.path.join(args.result_path, f'action_reason_{args.VLM}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QAs = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        for image_url in QAs:
            result = self.ar_VLM.evaluate_image(image_url)
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
        metrics = Metrics(args)
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

    def evaluate_whoops_VLM(self, args):
        results = {}
        for i in range(0, args.top_k):
            results[f'acc@{i + 1}'] = 0
        fieldnames = ['id']
        for i in range(0, args.top_k):
            fieldnames.append(f'acc@{i + 1}')
        csv_file_path = os.path.join(args.result_path, f'{args.test_set_QA.split("/")[-1].replace(".json", "")}'
                                                       f'_{args.description_type}'
                                                       f'_{args.VLM}_description_{args.LLM}_'
                                                       f'{args.VLM_prompt.replace(".jinja", "")}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QA_file = os.path.join(args.data_path, args.test_set_QA)
        QAs = json.load(open(QA_file))
        descriptions = pd.read_csv(os.path.join(args.data_path,
                                                'train',
                                                f'{args.description_type}_{args.VLM}_description_{args.task}.csv'))
        for image_index in QAs:
            image_url = f'{image_index}.png'
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
            image = Image.open(os.path.join(args.data_path, 'whoops_images', image_url))
            answers = self.whoops.get_prediction(image, description, QAs[image_index])
            correct_options = QAs[image_index][1] if len(QAs[image_index][0]) == 3 else QAs[image_index][0]
            print(answers)
            if len(answers) == 0:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
            else:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
                for i, answer in enumerate(answers[0: args.top_k]):
                    if answer in correct_options:
                        for j in range(i, args.top_k):
                            result[f'acc@{j + 1}'] = 1
            print(result)
            row = {}
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row['id'] = i
                for metric in result:
                    row[metric] = result[metric]
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

    @staticmethod
    def evaluate_whoops_LLM(args):
        def parse_options(options):
            return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

        def get_prediction(prompt, options, pipe):
            answers = []
            output = pipe(prompt)
            print(output)
            predictions = [''.join(i for i in prediction if i.isdigit()) for prediction in output.split(',')]
            for prediction in predictions:
                if prediction != '':
                    answers.append(int(prediction))
            predictions = set()
            for ind in answers:
                if len(options) > ind:
                    predictions.add(options[ind])
            answers = list(predictions)
            return answers
        results = {}
        for i in range(0, args.top_k):
            results[f'acc@{i+1}'] = 0
        fieldnames = ['id']
        for i in range(0, args.top_k):
            fieldnames.append(f'acc@{i+1}')
        csv_file_path = os.path.join(args.result_path, f'{args.test_set_QA.split("/")[-1].replace(".json", "")}'
                                                       f'_{args.description_type}'
                                                       f'_{args.VLM}_description_{args.LLM}_'
                                                       f'{args.VLM_prompt.replace(".jinja", "")}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QA_file = os.path.join(args.data_path, args.test_set_QA)
        QAs = json.load(open(QA_file))
        pipe = LLM(args)
        descriptions = pd.read_csv(os.path.join(args.data_path, 'train', f'{args.description_type}_{args.VLM}_description_{args.task}.csv'))
        for image in QAs:
            image_url = f'{image}.png'
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
            options = QAs[image][1]
            env = Environment(loader=FileSystemLoader(args.prompt_path))
            template = env.get_template(args.VLM_prompt)
            data = {'description': description, 'options': parse_options(options)}
            prompt = template.render(**data)
            answers = get_prediction(prompt, options, pipe)
            correct_options = QAs[image][1] if len(QAs[image][0]) == 3 else QAs[image][0]
            print(answers)
            if len(answers) == 0:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
            else:
                result = {}
                for i in range(0, args.top_k):
                    result[f'acc@{i + 1}'] = 0
                for i, answer in enumerate(answers[0: args.top_k]):
                    if answer in correct_options:
                        for j in range(i, args.top_k):
                            result[f'acc@{j+1}'] = 1
            print(result)
            row = {}
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row['id'] = i
                for metric in result:
                    row[metric] = result[metric]
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')

    def evaluate_action_reason_LLM(self, args):
        def parse_options(options):
            return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

        def get_prediction(prompt, options, pipe):
            answers = []
            output = pipe(prompt)
            print(output)
            outputs = output.split(',')
            predictions = [''.join(i for i in output if i.isdigit()) for output in outputs]
            print(predictions)
            for answer in predictions:
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

        results = {'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'p@1': 0, 'p@2': 0, 'p@3': 0}
        fieldnames = ['acc@1', 'acc@2', 'acc@3', 'p@1', 'p@2', 'p@3', 'id', 'prediction']
        csv_file_path = os.path.join(args.result_path,
                                     f'PittAd_{args.description_type}_{args.VLM}_description_{args.LLM}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        QA_file = os.path.join(args.data_path, args.test_set_QA)
        QAs = json.load(open(QA_file))
        pipe = LLM(args)
        descriptions = pd.read_csv(args.description_file)
        for image_url in QAs:
            if image_url not in descriptions.ID.values:
                continue
            description = descriptions.loc[descriptions['ID'] == image_url]['description'].values[0]
            print('description:', description)
            options = QAs[image_url][1]
            correct_options = QAs[image_url][0]
            env = Environment(loader=FileSystemLoader(args.prompt_path))
            template = env.get_template(args.VLM_prompt)
            data = {'description': description, 'options': parse_options(options)}
            prompt = template.render(**data)
            answers = get_prediction(prompt, options, pipe)
            result = {'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'p@1': 0, 'p@2': 0, 'p@3': 0}
            print(answers)
            correct_count = 0
            if len(answers) != 0:
                for i, answer in enumerate(answers[0:3]):
                    if answer in correct_options:
                        correct_count += 1
                        for j in range(i, 3):
                            result[f'acc@{j+1}'] = 1
                result['p@1'] = min(correct_count, 1)
                result['p@2'] = min(correct_count/2, 1)
                result['p@3'] = min(correct_count/3, 1)

            # for key in result:
            #     results[key] += result[key]
            print(result)
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = result
                row['id'] = image_url
                row['prediction'] = answers
                writer.writerow(list(row.values()))

            for metric in results:
                results[metric] += result[metric]
        for metric in results:
            print(f'average {metric} is: {results[metric] / len(list(QAs.keys()))}')


    @staticmethod
    def evaluate_image_text_ranking(args):
        metrics = Metrics(args)
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
    # metrics = Metrics(args)
    evaluation = Evaluation(args)
    evaluation.evaluate(args)
