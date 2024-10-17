import json
from jinja2 import Environment, FileSystemLoader
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import functional as TF
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import tempfile
from transformers import pipeline, BitsAndBytesConfig
import re
import base64
import requests
from VLMs.InternVL2 import InternVL
from VLMs.multi_image_InternVL import MultiInternVL
import itertools
from LLMs.LLM import LLM
from FlagEmbedding import BGEM3FlagModel

api_key = "sk-proj-zfkbSHxUNuF7Ev8TEWWRT3BlbkFJieFKktR5T8tIUVNAJRBz"


# Function to convert an image file to a tensor
def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    return tensor


class Metrics:
    def __init__(self, args):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device=args.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.args = args
        alignment_scores = [
            'image_text_alignment',
            'image_text_ranking',
            'MLLM_alignment',
            'multi_question_persuasiveness',
            'multi_question_persuasiveness_ranking'
        ]
        if args.evaluation_type in alignment_scores:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                torch_dtype="float16"
            )
            if args.VLM == 'LLAVA':
                self.pipe = pipeline("image-to-text", model='llava-hf/llava-1.5-13b-hf',
                                     model_kwargs={"quantization_config": quantization_config})
            elif args.VLM == 'InternVL':
                self.pipe = InternVL(args)
            elif args.VLM == 'MultiInternVL':
                self.pipe = MultiInternVL(args)
            self.QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
        llm_needed_evaluation = [
            'text_image_alignment',
            'multi_question_persuasiveness',
            'multi_question_persuasiveness_ranking',
            'llm_multi_question_persuasiveness_ranking'
        ]
        if args.evaluation_type in llm_needed_evaluation:
            self.llm = LLM(args)
            self.QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
            if args.evaluation_type == 'text_image_alignment':
                # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                #
                # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                # self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        if args.evaluation_type == 'image_text_ranking':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                                           token='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',
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                                              token='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',
                                                              device_map="auto",
                                                              trust_remote_code=True)

    @staticmethod
    def get_FID(generated_image_path, real_image_path, args):
        image_tensor_1 = image_to_tensor(generated_image_path)
        image_tensor_2 = image_to_tensor(real_image_path)

        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path_1 = os.path.join(tempdir, 'set1')
            dataset_path_2 = os.path.join(tempdir, 'set2')

            os.makedirs(dataset_path_1, exist_ok=True)
            os.makedirs(dataset_path_2, exist_ok=True)

            # Save the tensors as images
            TF.to_pil_image(image_tensor_1.squeeze()).save(os.path.join(dataset_path_1, 'image1.png'))
            TF.to_pil_image(image_tensor_2.squeeze()).save(os.path.join(dataset_path_2, 'image2.png'))

            # Calculate FID score
            fid_value = calculate_fid_given_paths([dataset_path_1, dataset_path_2],
                                                  batch_size=1,
                                                  device=torch.device(args.device),
                                                  dims=2048)

            return fid_value

    def get_image_image_CLIP_score(self, generated_image_path, real_image_path, args):
        # Load images
        image1 = Image.open(real_image_path).convert("RGB")
        image2 = Image.open(generated_image_path).convert("RGB")

        # Process images
        inputs = self.clip_processor(images=[image1, image2],
                                     return_tensors="pt",
                                     padding=True).to(device=args.device)

        # Extract image features from the CLIP model
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        # Normalize the feature vectors
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

        # Calculate cosine similarity between the two images
        cosine_similarity = torch.nn.functional.cosine_similarity(image_features[0].unsqueeze(0),
                                                                  image_features[1].unsqueeze(0)).item()

        return cosine_similarity

    def get_text_image_CLIP_score(self, generated_image_path, action_reason, args):
        image = Image.open(generated_image_path).convert("RGB")
        cosine_similarity = {'text': 0,
                             'action': 0,
                             'reason': 0}
        for AR in action_reason:
            reason = AR.lower().split('because')[-1]
            action = AR.lower().split('because')[0]
            inputs_image = self.clip_processor(images=image,
                                               return_tensors="pt",
                                               padding=True).to(device=args.device)
            inputs_text = self.clip_processor(text=AR,
                                              return_tensors="pt",
                                              padding=True).to(device=args.device)
            inputs_reason = self.clip_processor(text=reason,
                                                return_tensors="pt",
                                                padding=True).to(device=args.device)
            inputs_action = self.clip_processor(text=action,
                                                return_tensors="pt",
                                                padding=True).to(device=args.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs_image)
                text_features = self.clip_model.get_text_features(**inputs_text)
                reason_features = self.clip_model.get_text_features(**inputs_reason)
                action_features = self.clip_model.get_text_features(**inputs_action)
            # Normalize the feature vectors
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            reason_features = torch.nn.functional.normalize(reason_features, p=2, dim=1)
            action_features = torch.nn.functional.normalize(action_features, p=2, dim=1)

            # Calculate cosine similarity between the two images
            cosine_similarity['text'] += torch.nn.functional.cosine_similarity(image_features, text_features).item()
            cosine_similarity['reason'] += torch.nn.functional.cosine_similarity(image_features, reason_features).item()
            cosine_similarity['action'] += torch.nn.functional.cosine_similarity(image_features, action_features).item()
        cosine_similarity['text'] = cosine_similarity['text'] / len(action_reason)
        cosine_similarity['reason'] = cosine_similarity['reason'] / len(action_reason)
        cosine_similarity['action'] = cosine_similarity['action'] / len(action_reason)
        return cosine_similarity

    def get_action_reason_image_CLIP_score(self, generated_image_path, action_reason, args):
        image = Image.open(generated_image_path).convert("RGB")
        cosine_similarity = 0
        for AR in action_reason:
            inputs_image = self.clip_processor(images=image,
                                               return_tensors="pt",
                                               padding=True).to(device=args.device)
            inputs_text = self.clip_processor(text=AR,
                                              return_tensors="pt",
                                              padding=True).to(device=args.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs_image)
                text_features = self.clip_model.get_text_features(**inputs_text)
            # Normalize the feature vectors
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

            # Calculate cosine similarity between the two images
            cosine_similarity += torch.nn.functional.cosine_similarity(image_features, text_features).item()
        cosine_similarity = cosine_similarity / len(action_reason)
        return cosine_similarity

    def get_scores(self, text_description, generated_image_path, real_image_path, args):
        text_image_score = self.get_text_image_CLIP_score(generated_image_path, text_description, args)
        scores = {
            'image_image_CLIP_score': self.get_image_image_CLIP_score(generated_image_path, real_image_path, args),
            'image_text_CLIP_score': text_image_score['text'],
            'image_action_CLIP_score': text_image_score['action'],
            'image_reason_CLIP_score': text_image_score['reason'],
            'FID_score': self.get_FID(generated_image_path, real_image_path, args)}

        return scores

    def get_creativity_scores(self, text_description, generated_image_path, product_image_paths, args):
        image_scores = []
        for product_image in product_image_paths[:1]:
            image_scores.append(self.get_image_image_CLIP_score(generated_image_path, product_image, args))
        avg_image_score = sum(image_scores) / len(image_scores)
        text_score = self.get_action_reason_image_CLIP_score(generated_image_path, text_description, args)
        creativity = text_score / (avg_image_score + 0.1)
        return creativity

    def get_persuasiveness_creativity_score(self, text_alignment_score, generated_image_path, product_image_paths,
                                            args):
        image_scores = []
        for product_image in product_image_paths[:1]:
            image_scores.append(self.get_image_image_CLIP_score(generated_image_path, product_image, args))
        avg_image_score = sum(image_scores) / len(image_scores)
        creativity = text_alignment_score[1] / (avg_image_score + 0.01)
        return creativity

    @staticmethod
    def get_image_message_prompt():
        answer_format = 'Answer: I should ${action} because {reason}'
        prompt = (f'USER:<image>\n'
                  f'This image is designed to convince the audience to take an action because of some reason. What is '
                  f'the action and reason in this image? '
                  f'Your answer must follow the format of: {answer_format} and must be meaningful in English.'
                  f'Assistant: ')
        return prompt

    def get_image_text_alignment_scores(self, action_reasons, generated_image_path, args):
        prompt = self.get_image_message_prompt()
        image = Image.open(generated_image_path)
        output = self.pipe(image,
                           prompt=prompt,
                           generate_kwargs={"max_new_tokens": 45})
        message = output[0]["generated_text"].split(':')[-1]
        print(f'detected message is: {message}')
        inputs_message = self.clip_processor(text=message,
                                             return_tensors="pt",
                                             padding=True).to(device=args.device)
        message_features = self.clip_model.get_text_features(**inputs_message)
        text_text_similarity = 0
        for i, action_reason in enumerate(action_reasons):
            print(f'action_reason {i} is: {action_reason}')
            inputs_AR = self.clip_processor(text=action_reason,
                                            return_tensors="pt",
                                            padding=True).to(device=args.device)
            AR_features = self.clip_model.get_text_features(**inputs_AR)
            text_text_similarity += torch.nn.functional.cosine_similarity(AR_features, message_features).item()
        return text_text_similarity / len(action_reasons)

    def get_text_image_alignment_score(self, action_reasons, description, args):
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

        if args.fine_tuned:
            format = ''
        else:
            format = 'The interpretation format is: I should ${action} because ${reason}. ONLY RETURN A SINGLE SENTENCE IN THIS FORMAT'
        prompt = f"""What is the correct interpretation for the described image:
                     Description: {description}.
                     {format}"""
        generated_image_message = self.llm(prompt)
        generated_image_message = generated_image_message.lower()
        print(generated_image_message)
        # tokenized_generated_image_message = self.llm.model.tokenizer(generated_image_message,
        #                                                    padding=True,
        #                                                    max_length=25,
        #                                                    return_tensors="pt").to(device=args.device)
        # tokenized_generated_image_message = tokenized_generated_image_message['input_ids'].to(torch.float16)
        # encoded_input = self.tokenizer([generated_image_message], padding=True, truncation=True, return_tensors='pt')
        # with torch.no_grad():
        #     model_output = self.model(**encoded_input)
        # generated_image_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        #
        # # Normalize embeddings
        # generated_image_embeddings = nn.functional.normalize(generated_image_embeddings, p=2, dim=1)
        #
        similarity_score = 0
        for action_reason in action_reasons:
            print(action_reason)
            action_reason = action_reason.lower()
            # encoded_input = self.tokenizer([action_reason], padding=True, truncation=True,
            #                                return_tensors='pt')
            # with torch.no_grad():
            #     model_output = self.model(**encoded_input)
            # action_reason_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            #
            # # Normalize embeddings
            # action_reason_embeddings = nn.functional.normalize(action_reason_embeddings, p=2, dim=1)
            # similarity_score += self.cos(action_reason_embeddings, generated_image_embeddings)
            # print(similarity_score)
            similarity_score += self.model.compute_score([action_reason, generated_image_message],
                                                         max_passage_length=128,
                                                         weights_for_different_modes=[0.4, 0.2, 0.4])['colbert+sparse+dense']

        return generated_image_message, (similarity_score.item() / len(action_reasons))

    @staticmethod
    def get_image_description_prompt():
        prompt = (f'USER:<image>\n'
                  f'This image is designed to convince the audience to take an action because of some reason.'
                  f'Describe the image.'
                  f'Assistant: ')
        return prompt

    @staticmethod
    def get_ranking_prompt(action_reason, first_message, second_message):
        answer_format = 'answer: ${index of the best answer}'
        prompt = (
            f'Which of the following descriptions better conveys {action_reason}? The answer must be in format of {answer_format} not any other form.'
            f'0. {first_message}'
            f'1. {second_message}'
            f'Assistant:')
        return prompt

    def get_multi_question_persuasiveness_ranking(self, generated_image1, generated_image2):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                print("No numeric value found in the input string")
                return 0

        def evaluate_story(generated_image1, generated_image2, image_url):
            action_reason = self.QA[image_url][0][0]
            action = action_reason.lower().split('/')[0]
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            data = {'action_reason': action_reason,
                    'action': action}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasStory_relative.jinja')
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output
            print(f'has story: {output}')
            has_story = extract_number(output)
            return has_story

        def evaluate_unusualness(generated_image1, generated_image2):
            data = {}
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasUnusual_relative.jinja')
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output
            print(f'unusualness:{output}')
            return extract_number(output)

        def evaluate_originality(generated_image1, generated_image2):
            data = {}
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('originality_relative.jinja')
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output
            print(f'originalilty: {output}')
            return extract_number(output)

        def evaluate_audience(generated_image1, generated_image2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('getAudience.jinja')
            prompt = template.render(**data)
            audience = self.llm(prompt)
            template = env.get_template('isAudienceCorrect_relative.jinja')
            data = {'audience': audience}
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'audience: {output}')
            return extract_number(output)

        def evaluate_artistic(generated_image1, generated_image2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('artistic_relative.jinja')
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'artistic: {output}')
            if 'yes' in output.lower():
                return 1
            else:
                return 0

        def evaluate_imagination(generated_image1, generated_image2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('imagination_relative.jinja')
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'imagination: {output}')
            return extract_number(output)

        def evaluate_appeal(generated_image1, generated_image2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('appeal_type.jinja')
            prompt = template.render(**data)
            appealing_type = self.llm(prompt)
            template = env.get_template('appealing_score_relative.jinja')
            data = {'appealing_type': appealing_type}
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'appeal score: {output}')
            return extract_number(output) / 5

        def evaluate_maslow_need(generated_image1, generated_image2, image_url):
            action_reason = self.QA[image_url][0][0]
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('maslow_needs_statement.jinja')
            data = {'action_reason': action_reason}
            prompt = template.render(**data)
            output = self.llm(prompt)
            statement_needs = output.split(',')
            template = env.get_template('maslow_needs_relative.jinja')
            data = {'need': statement_needs}
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})

            print(f'statement needs: {statement_needs}')
            print(f'maslow\'s need: {output}')
            return extract_number(output)

        def evaluate_benefit(generated_image1, generated_image2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image1 = Image.open(generated_image1)
            image2 = Image.open(generated_image2)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasBenefit_relative.jinja')
            prompt = template.render(**data)
            output = self.pipe(image1,
                               image2,
                               prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'benefit score: {output}')
            return extract_number(output)

        image_url = '/'.join(generated_image1.split('/')[-2:])
        scores = {
            'has_story': evaluate_story(generated_image1, generated_image2, image_url),
            'unusualness': evaluate_unusualness(generated_image1, generated_image2, ),
            'originality': evaluate_originality(generated_image1, generated_image2, ),
            'artistic': evaluate_artistic(generated_image1, generated_image2, image_url),
            'imagination': evaluate_imagination(generated_image1, generated_image2, image_url),
            'audience': evaluate_audience(generated_image1, generated_image2, image_url),
            'maslow_need': evaluate_maslow_need(generated_image1, generated_image2, image_url),
            'benefit': evaluate_benefit(generated_image1, generated_image2, image_url),
            'appeal': evaluate_appeal(generated_image1, generated_image2, image_url),
        }
        scores['persuasiveness'] = sum(list(scores.values())) / len(scores)
        persasiveness = scores['persuasiveness']
        print(f'persuasiveness score: {persasiveness}')

        return scores

    def get_llm_multi_question_persuasiveness_ranking(self, generated_image1, generated_image2, image_url):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                print("No numeric value found in the input string")
                return 0

        def evaluate_story(description1, description2, image_url):
            action_reason = self.QA[image_url][0][0]
            action = action_reason.lower().split('/')[0]
            data = {'action_reason': action_reason,
                    'description1': description1,
                    'description2': description2,
                    'action': action}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasStory_relative_llm.jinja')
            prompt = template.render(**data)
            output = self.llm(prompt)
            output = output
            print(f'has story: {output}')
            has_story = extract_number(output)
            return has_story

        def evaluate_unusualness(description1, description2):
            data = {'description1': description1,
                    'description2': description2}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasUnusual_relative_llm.jinja')
            prompt = template.render(**data)
            output = self.llm(prompt)
            output = output
            print(f'unusualness:{output}')
            return extract_number(output)

        def evaluate_originality(description1, description2):
            data = {'description1': description1,
                    'description2': description2}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('originality_relative_llm.jinja')
            prompt = template.render(**data)
            output = self.llm(prompt)
            output = output
            print(f'originalilty: {output}')
            return extract_number(output)

        def evaluate_audience(description1, description2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('getAudience.jinja')
            prompt = template.render(**data)
            audience = self.llm(prompt)
            template = env.get_template('isAudienceCorrect_relative_llm.jinja')
            data = {'audience': audience,
                    'description1': description1,
                    'description2': description2}
            prompt = template.render(**data)
            output = self.llm(prompt)
            print(f'audience: {output}')
            return extract_number(output)

        def evaluate_artistic(description1, description2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason,
                    'description1': description1,
                    'description2': description2}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('artistic_relative_llm.jinja')
            prompt = template.render(**data)
            output = self.llm(prompt)
            print(f'artistic: {output}')
            if 'yes' in output.lower():
                return 1
            else:
                return 0

        def evaluate_imagination(description1, description2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason,
                    'description1': description1,
                    'description2': description2}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('imagination_relative_llm.jinja')
            prompt = template.render(**data)
            output = self.llm(prompt)
            print(f'imagination: {output}')
            return extract_number(output)

        def evaluate_appeal(description1, description2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('appeal_type.jinja')
            prompt = template.render(**data)
            appealing_type = self.llm(prompt)
            template = env.get_template('appealing_score_relative_llm.jinja')
            data = {'appealing_type': appealing_type,
                    'description1': description1,
                    'description2': description2}
            prompt = template.render(**data)
            output = self.llm(prompt)
            print(f'appeal score: {output}')
            return extract_number(output) / 5

        def evaluate_maslow_need(description1, description2, image_url):
            action_reason = self.QA[image_url][0][0]
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('maslow_needs_statement.jinja')
            data = {'action_reason': action_reason}
            prompt = template.render(**data)
            output = self.llm(prompt)
            statement_needs = output.split(',')
            template = env.get_template('maslow_needs_relative_llm.jinja')
            data = {'need': statement_needs,
                    'description1': description1,
                    'description2': description2}
            prompt = template.render(**data)
            output = self.llm(prompt)
            print(f'statement needs: {statement_needs}')
            print(f'maslow\'s need: {output}')
            return extract_number(output)

        def evaluate_benefit(description1, description2, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason,
                    'description1': description1,
                    'description2': description2}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasBenefit_relative_llm.jinja')
            prompt = template.render(**data)
            output = self.llm(prompt)
            print(f'benefit score: {output}')
            return extract_number(output)

        # image_url = '/'.join(generated_image1.split('/')[-2:])
        scores = {
            'has_story': evaluate_story(generated_image1, generated_image2, image_url),
            'unusualness': evaluate_unusualness(generated_image1, generated_image2, ),
            'originality': evaluate_originality(generated_image1, generated_image2, ),
            'artistic': evaluate_artistic(generated_image1, generated_image2, image_url),
            'imagination': evaluate_imagination(generated_image1, generated_image2, image_url),
            'audience': evaluate_audience(generated_image1, generated_image2, image_url),
            'maslow_need': evaluate_maslow_need(generated_image1, generated_image2, image_url),
            'benefit': evaluate_benefit(generated_image1, generated_image2, image_url),
            'appeal': evaluate_appeal(generated_image1, generated_image2, image_url),
        }
        scores['persuasiveness'] = sum(list(scores.values())) / len(scores)
        persasiveness = scores['persuasiveness']
        print(f'persuasiveness score: {persasiveness}')

        return scores

    def get_multi_question_persuasiveness(self, generated_image):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                print("No numeric value found in the input string")
                return 0

        def evaluate_story(generated_image, image_url):
            action_reason = self.QA[image_url][0][0]
            action = action_reason.lower().split('/')[0]
            image = Image.open(generated_image)
            data = {'action_reason': action_reason,
                    'action': action}
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasStory.jinja')
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output
            has_story_score = 0
            print(f'has story: {output}')
            if 'yes' in output.lower():
                has_story_score += 1
                template = env.get_template('isRelatedAction.jinja')
                prompt = template.render(**data)
                output = self.pipe(image, prompt=prompt,
                                   generate_kwargs={"max_new_tokens": 45})
                output = output
                print(f'is related to action: {output}')
                if 'yes' in output.lower():
                    has_story_score += 1
                    template = env.get_template('isRelatedActionReason.jinja')
                    prompt = template.render(**data)
                    output = self.pipe(image, prompt=prompt,
                                       generate_kwargs={"max_new_tokens": 45})
                    output = output
                    print(f'is related to statement: {output}')
                    if 'yes' in output.lower():
                        has_story_score += 1
            has_story_score = has_story_score / 3
            print(f'has story: {has_story_score}')
            return has_story_score

        def evaluate_unusualness(generated_image):
            data = {}
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasUnusualConnection.jinja')
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output
            print(f'unusualness:{output}')
            if 'yes' in output:
                return 1
            else:
                return 0

        def evaluate_originality(generated_image):
            data = {}
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('originality.jinja')
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output
            print(f'originalilty: {output}')
            if 'yes' in output:
                return 1
            else:
                return 0

        def evaluate_audience(generated_image, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('getAudience.jinja')
            prompt = template.render(**data)
            audience = self.llm(prompt)
            template = env.get_template('isAudienceCorrect.jinja')
            data = {'audience': audience}
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'audience: {output}')
            if 'yes' in output.lower():
                return 1
            else:
                return 0

        def evaluate_artistic(generated_image, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('artistic.jinja')
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'artistic: {output}')
            if 'yes' in output.lower():
                return 1
            else:
                return 0

        def evaluate_imagination(generated_image, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('imagination.jinja')
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'imagination: {output}')
            if 'yes' in output.lower():
                return 1
            else:
                return 0

        def evaluate_appeal(generated_image, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('appeal_type.jinja')
            prompt = template.render(**data)
            appealing_type = self.llm(prompt)
            template = env.get_template('appealing_score.jinja')
            data = {'appealing_type': appealing_type}
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'appeal score: {output}')
            return extract_number(output) / 5

        def evaluate_maslow_need(generated_image, image_url):
            action_reason = self.QA[image_url][0][0]
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('maslow_needs_statement.jinja')
            data = {'action_reason': action_reason}
            prompt = template.render(**data)
            output = self.llm(prompt)
            statement_needs = output.split(',')
            template = env.get_template('maslow_needs_image.jinja')
            data = {'action_reason': action_reason}
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            image_needs = output.split(',')
            for i in range(len(image_needs)):
                image_needs[i] = image_needs[i].lower()
                image_needs[i] = image_needs[i].strip()
            count = 0
            for need in statement_needs:
                if need.lower().strip() in image_needs:
                    count += 1
            print(f'image needs: {image_needs}')
            print(f'statement needs: {statement_needs}')
            return count / len(statement_needs)

        def evaluate_benefit(generated_image, image_url):
            action_reason = self.QA[image_url][0][0]
            data = {'action_reason': action_reason}
            image = Image.open(generated_image)
            env = Environment(loader=FileSystemLoader(self.args.prompt_path))
            template = env.get_template('hasBenefit.jinja')
            prompt = template.render(**data)
            output = self.pipe(image, prompt=prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(f'benefit score: {output}')
            if 'yes' in output.lower():
                return 1
            else:
                return 0

        image_url = '/'.join(generated_image.split('/')[-2:])
        scores = {
            'has_story': evaluate_story(generated_image, image_url),
            'unusualness': evaluate_unusualness(generated_image),
            'originality': evaluate_originality(generated_image),
            'artistic': evaluate_artistic(generated_image, image_url),
            'imagination': evaluate_imagination(generated_image, image_url),
            'audience': evaluate_audience(generated_image, image_url),
            'maslow_need': evaluate_maslow_need(generated_image, image_url),
            'benefit': evaluate_benefit(generated_image, image_url),
            'appeal': evaluate_appeal(generated_image, image_url),
        }
        scores['persuasiveness'] = sum(list(scores.values())) / len(scores)
        persasiveness = scores['persuasiveness']
        print(f'persuasiveness score: {persasiveness}')

        return scores

    def get_image_text_ranking(self, action_reasons, first_generated_image_path, second_generated_image_path, args):
        prompt = self.get_image_description_prompt()
        first_image = Image.open(first_generated_image_path)
        second_image = Image.open(second_generated_image_path)
        first_output = self.pipe(first_image,
                                 prompt=prompt,
                                 generate_kwargs={"max_new_tokens": 512})
        first_message = first_output[0]["generated_text"].split(':')[-1]
        print(f'detected message for first image is: {first_message}')

        second_output = self.pipe(second_image,
                                  prompt=prompt,
                                  generate_kwargs={"max_new_tokens": 512})
        second_message = second_output[0]["generated_text"].split(':')[-1]
        print(f'detected message for first image is: {second_message}')
        first_image = 0
        for AR in action_reasons:
            prompt = self.get_ranking_prompt(AR, first_message, second_message)
            inputs = self.tokenizer(prompt,
                                    return_tensors="pt").to(device=args.device)
            generated_ids = self.model.generate(**inputs,
                                                max_new_tokens=20)
            output = self.tokenizer.batch_decode(generated_ids,
                                                 skip_special_tokens=True)[0].strip()
            output = output.split('Assistant:')[-1]
            print(output)
            if '1' not in output:
                first_image += 1
        return first_image / len(action_reasons)

    def get_MLLM_alignment_score(self, generated_image):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                return 0
                raise ValueError("No numeric value found in the input string")

        if type(generated_image) == str:
            image = Image.open(generated_image).convert("RGB")
            print(generated_image.split('/'))
            image_url = '/'.join(generated_image.split('/')[-2:])
            action_reason = self.QA[image_url][0][0]

        else:
            image = generated_image
            action_reason = ''
        prompt = f"""
        <image>\n USER:
        Context:Assume you are a human evaluating the image. You are given a statement and you are asked to check if the given message is represented in the image or not.
        You must return a score in range of 0 to 5. You must return 5 if the message is represented in the image or 0 if the image is totally irrelevant.
        For example if the action-reason is 'I should drink beer because it is as light as feather', and the image or text in the image is fully showing this message, then the score is 5.
        If the image or the text in the image is only showing I should drink beer or it is showing the beer is as light as feather, then the score is 3.
        If the image is totally irrelevant to beer or drinking or the lightness of the drink, then the score is 0.
        Question: Based on the context evaluate the image for '{action_reason}' in range of (0, 5).
        Your output format is only Answer: score\n form, no other form. Empty is not allowed.
        ASSISTANT:
        """
        output = self.pipe(image, prompt=prompt,
                           generate_kwargs={"max_new_tokens": 45})
        print(output)
        # output = output[0]["generated_text"].split(':')[-1]
        output = output.split(':')[-1]
        print(output)
        numeric_value = extract_number(output)
        print(f'persuasiveness: {numeric_value}')
        return numeric_value


class PersuasivenessMetric:
    def __init__(self, args):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        model_id_map = {
            'LLAVA': "llava-hf/llava-1.5-13b-hf",
            'VILA': "Efficient-Large-Model/VILA1.5-13b"
        }
        task_map = {
            'LLAVA': "image-to-text",
            'VILA': "text-generation"
        }
        if args.VLM != 'GPT4v':
            if args.VLM in model_id_map:
                model_id = model_id_map[args.VLM]
                task = task_map[args.VLM]
                self.pipe = pipeline(task,
                                     model=model_id,
                                     model_kwargs={"quantization_config": quantization_config},
                                     trust_remote_code=True,
                                     device_map='auto')
                self.LLM_model = LLM(args)
            else:
                print('InternVL')
                self.LLM_model = LLM(args)
                self.pipe = InternVL(args)
        self.QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))

    def get_persuasiveness_score(self, generated_image):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                return 0
                raise ValueError("No numeric value found in the input string")

        if type(generated_image) == str:
            image = Image.open(generated_image).convert("RGB")
            print(generated_image.split('/'))

        else:
            image = generated_image
        prompt = f"""
        <image>\n USER:
        Context: If the image convinces the audience to take an action like buying a product, etc, then the image is considered persuasive.
        Question: Based on the context score the persuasiveness of the image in range of (-5, 5). For totally not persuasive images choose -5, for totally persuasive choose 5.
        Your output format is only Answer: score\n form, no other form. Empty is not allowed.
        ASSISTANT:
        """
        output = self.pipe(image, prompt=prompt,
                           generate_kwargs={"max_new_tokens": 45})
        output = output[0]["generated_text"].split(':')[-1]
        print(output)
        numeric_value = extract_number(output)
        print(f'persuasiveness: {numeric_value}')
        return numeric_value

    def get_action_reason_aware_persuasiveness_score(self, generated_image):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                return 0
                raise ValueError("No numeric value found in the input string")

        if type(generated_image) == str:
            image = Image.open(generated_image).convert("RGB")
            print(generated_image.split('/'))
            image_url = '/'.join(generated_image.split('/')[-2:])
            action_reason = 'in convincing the message of ' + self.QA[image_url][0][0]

        else:
            image = generated_image
            action_reason = ''
        prompt = f"""
        <image>\n USER:
        Context: If the image convinces the audience to take an action like buying a product, etc, then the image is considered persuasive.
        Question: Based on the context score the persuasiveness of the image {action_reason} in range of (-5, 5). For totally not persuasive images choose -5, for totally persuasive choose 5.
        Your output format is only Answer: score\n form, no other form. Empty is not allowed.
        ASSISTANT:
        """
        output = self.pipe(image, prompt=prompt,
                           generate_kwargs={"max_new_tokens": 45})
        output = output[0]["generated_text"].split(':')[-1]
        print(output)
        numeric_value = extract_number(output)
        print(f'persuasiveness: {numeric_value}')
        return numeric_value

    def get_persuasiveness_alignment(self, generated_image, action_reasons=None):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                print("No numeric value found in the input string")
                return 0

        if type(generated_image) != str:
            image = generated_image
        else:
            image = Image.open(generated_image).convert("RGB")
            print(generated_image.split('/'))
            image_url = '/'.join(generated_image.split('/')[-2:])
            action_reasons = self.QA[image_url][0]
        statements_count = len(action_reasons)
        action_numeric_value = 0
        reason_numeric_value = 0
        for action_reason in action_reasons:
            print(action_reason)
            action = action_reason.lower().split('because')[0]
            reason = action_reason.lower().split('because')[-1]
            answer_format = 'Answer: ${score}'
            action_score_prompt = f"""
                    <image>\n USER:
                    Imagine you are a human evaluating how convincing is an image. Your task is to score how convincing an image is given an action on a scale from -10 to 10. 
                    Context: If the image convinces the audience to take an action considered convincing.
                    Question: Given the message of {action} provide the score in the following format: {answer_format}
                    ASSISTANT:
                    """
            output = self.pipe(image, prompt=action_score_prompt,
                               generate_kwargs={"max_new_tokens": 45})  # [0]['generated_text']
            output = output.split(':')[-1]
            print('action:', output)
            action_numeric_value += extract_number(output)
            reason_score_prompt = \
                f"""
                <image>\n USER:
                Imagine you are a human evaluating how related an image is to a reason. Your task is to score the relatedness of an image and a reason on a scale from -10 to 10. 
                Context: Imagine the image is convincing the audience to take the action in the message of {action}. If the reason in the image is the same as the given message in the next sentence, the score is 5 and if it is totally irrelevant the score is -5.
                Question: Given the message of {reason} and the image, provide the score in the following format: {answer_format}. 
                ASSISTANT:
                """

            output = self.pipe(image, prompt=reason_score_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output  # [0]['generated_text']
            output = output.split(':')[-1]
            print('reason:', output)
            reason_numeric_value += extract_number(output)
        return (reason_numeric_value + action_numeric_value) / (2 * statements_count)

    def get_multi_question_evaluation(self, generated_image):
        def parse_options(options):
            return '\n-'.join(options)

        def get_audience_list():
            age = ['baby', 'child', 'young adult', 'middle age', 'retired']
            gender = ['woman', 'man', 'nonbinary', 'everyone']
            education = ['with college degree', 'without college degree']
            marital_status = ['single', 'married', 'married with kids']
            combinations = list(itertools.product(gender, age, education, marital_status))
            people_strings = [
                f" {m} {a} {g} {e}"
                for g, a, e, m in combinations
            ]
            people_strings.append('family')
            return people_strings

        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                print("No numeric value found in the input string")
                return 0

        # if type(generated_image) != str:
        #     image = generated_image
        # else:
        image = Image.open(generated_image)  # .convert("RGB")
        # print(generated_image.split('/'))
        # image_url = '/'.join(generated_image.split('/')[-2:])
        # action_reasons = self.QA[image_url][0]
        action_reasons = ['aysan']
        # statements_count = len(action_reasons)
        statements_count = 1
        # has_story = 0
        # is_unusual = 0
        # properties_score = 0
        # audience_score = 0
        # audiences = []
        # memorability_score = 0
        # benefit_score = 0
        appealing_score = 0
        appealing_type = []
        # maslow_pyramid_needs = []
        for action_reason in action_reasons[:1]:
            print(action_reason)
            binary_answer_format = 'Answer: ${answer}'
            string_answer_format = 'Answer: ${answer}'
            score_answer_format = 'Answer: ${score}'
            # has_story_prompt = f"""
            #         <image>\n USER:
            #         Context: This is an advertisement image convincing the audience to take an action.
            #         The image might show only the product being advetised or it can be creative and have a story.
            #         Question: Does this image has a story?
            #         Your output must either be 1 for Yes or 0 for No without any further explanation. Follow the answer format of {binary_answer_format}.
            #         ASSISTANT:
            #         """
            # output = self.pipe(image, prompt=has_story_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('Does the image have story:', output)
            # has_story += extract_number(output)
            # is_unusual_prompt = \
            #     f"""
            #     <image>\n USER:
            #     Context: This is an advertisement image convincing the audience to take an action.
            #     Question: Is there any unusual object/objects in the image?
            #     Your output must either be 1 for Yes or 0 for No without any further explanation. Follow the answer format of {binary_answer_format}.
            #     ASSISTANT:
            #     """
            #
            # output = self.pipe(image, prompt=is_unusual_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('Does the image have any unusual objects:', output)
            # is_unusual += extract_number(output)
            properties_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take an action.
                Question: How strong of an association does it create between product and properties?
                Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
                ASSISTANT:
                """
            #
            # output = self.pipe(image, prompt=properties_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('How strong of an association does it create between product and properties:', output)
            # properties_score += extract_number(output)
            # audience_prompt = \
            #     f"""
            #     <image>\n USER:
            #     Context: This is the message of an advertisement image convincing the audience to take the action in the sentence. The message is: {action_reason}.
            #     Question: Choose the best option describing the audience of this advertisement?
            #     Options:
            #     {parse_options(get_audience_list())}
            #     Your output must be the possible audience for the advertisement without considering the image without any further explanation. Follow the answer format of {string_answer_format}.
            #     ASSISTANT:
            #     """
            # output = self.pipe(image, prompt=audience_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('Who is the audience of the advertisement message:', output)
            # audiences.append(output)
            # audience_score_prompt = \
            #     f"""
            #     <image>\n USER:
            #     Context: This is an advertisement image convincing the audience to take the action in the sentence. The audience of this image is {output}.
            #     You are a human rating how well the image resonate with the given audience, considering the needs of the described audience.
            #     Question:  How well does it resonate with its audience?
            #     Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
            #     ASSISTANT:
            #     """
            # output = self.pipe(image, prompt=audience_score_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('How well does it resonate with its audience:', output)
            # audience_score += extract_number(output)
            # memorability_prompt = \
            #     f"""
            #     <image>\n USER:
            #     Context: This is an advertisement image convincing the audience to take the action in the sentence.
            #     Question:  How memorable is the image?
            #     Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
            #     ASSISTANT:
            #     """
            # output = self.pipe(image, prompt=memorability_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('How memorable is the image:', output)
            # memorability_score += extract_number(output)
            # benefit_prompt = \
            #     f"""
            #     <image>\n USER:
            #     Context: In advertisement designers try to show the features of their product into benefits for consumers to show how well their product improves consumers life.
            #     You are given an advertisement image convincing the audience to take an action, while trying to turn the features of the product into benefits.
            #     Question:  How well does it turn features into benefits?
            #     Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
            #     ASSISTANT:
            #     """
            # output = self.pipe(image, prompt=benefit_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('How well does it turn features into benefits:', output)
            # benefit_score += extract_number(output)
            appealing_type_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take the action in the sentence. 
                Each advertisement image, aims to either target the audiences' emotion, ethics, or logic.
                Question:  Which category is the image appealing to? emotion, ethics, or logic?
                Your output must be the category among emotion, ethics, and logic, without any further explanation. Follow the answer format of {string_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=appealing_type_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('Which category is the image appealing to:', output)

            appealing_score_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take the action in the sentence.
                Question:  How much does the image appeal to {output}?
                Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=appealing_score_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('How appealing is the image:', output)
            appealing_score += extract_number(output)
            # maslow_pyramid_needs_prompt = \
            #     f"""
            #     <image>\n USER:
            #     Context: This is an advertisement image convincing the audience to take the action in the sentence.
            #     Each ad image is designed to target one of the Maslow’s pyramid needs. The needs are:
            #         - self actualisation: meeting one's full potential in life
            #         - esteem: For example, respect, status, recognition, strength
            #         - love/belonging: For example, friendship, intimacy, family, connections
            #         - safety: For example, security, health, finance
            #         - biological & physiological: For example, food, sleep, water
            #     Question:  What needs does it appeal to in Maslow’s pyramid? Choose among (self actualisation, esteem, love/belonging, saftey, biological & physiological)
            #     Your output must be the category among options, without any further explanation. Follow the answer format of {string_answer_format}.
            #     ASSISTANT:
            #     """
            # output = self.pipe(image, prompt=maslow_pyramid_needs_prompt,
            #                    generate_kwargs={"max_new_tokens": 45})
            # output = output.split(':')[-1]
            # print('What needs does it appeal to in Maslow’s pyramid:', output)
            # maslow_pyramid_needs.append(output)
        outputs = {
            # 'has_story': has_story/statements_count,
            # 'is_unusual': is_unusual/statements_count,
            # 'properties_score': properties_score/statements_count,
            # 'audience_score': audience_score/statements_count,
            # 'audiences': audiences,
            # 'memorability_score': memorability_score/statements_count,
            # 'benefit_score': benefit_score/statements_count,
            'appealing_score': appealing_score / statements_count,
            'appealing_type': appealing_type,
            # 'maslow_pyramid_needs': maslow_pyramid_needs
        }
        return outputs

    def get_multi_question_score_evaluation(self, generated_image):
        def parse_options(options):
            return '\n-'.join(options)

        def get_audience_list():
            age = ['baby', 'child', 'young adult', 'middle age', 'retired']
            gender = ['woman', 'man', 'nonbinary', 'everyone']
            education = ['with college degree', 'without college degree']
            marital_status = ['single', 'married', 'married with kids']
            combinations = list(itertools.product(gender, age, education, marital_status))
            people_strings = [
                f" {m} {a} {g} {e}"
                for g, a, e, m in combinations
            ]
            people_strings.append('family')
            return people_strings

        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                print("No numeric value found in the input string")
                return 0

        # if type(generated_image) != str:
        #     image = generated_image
        # else:
        image = Image.open(generated_image)  # .convert("RGB")
        print(generated_image.split('/'))
        image_url = '/'.join(generated_image.split('/')[-2:])
        action_reasons = self.QA[image_url][0]
        # action_reasons = ['aysan']
        statements_count = len(action_reasons)
        # statements_count = 1
        has_story = 0
        is_unusual = 0
        properties_score = 0
        audience_score = 0
        audiences = []
        memorability_score = 0
        benefit_score = 0
        appealing_score = 0
        appealing_type = 0
        maslow_pyramid_needs = 0
        for action_reason in action_reasons:
            print(action_reason)
            binary_answer_format = 'Answer: ${answer}'
            string_answer_format = 'Answer: ${answer}'
            score_answer_format = 'Answer: ${score}'
            has_story_prompt = f"""
                    <image>\n USER:
                    Context: This is an advertisement image convincing the audience to take an action.
                    The image might show only the product being advetised or it can be creative and have a story.
                    Question: Does this image has a story?
                    Your output must either be 1 for Yes or 0 for No without any further explanation. Follow the answer format of {binary_answer_format}.
                    ASSISTANT:
                    """
            output = self.pipe(image, prompt=has_story_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            print(output)
            output = output.split(':')[-1]
            print('Does the image have story:', output)
            has_story += extract_number(output)
            is_unusual_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take an action.
                Question: Is there any unusual object/objects in the image?
                Your output must either be 1 for Yes or 0 for No without any further explanation. Follow the answer format of {binary_answer_format}.
                ASSISTANT:
                """

            output = self.pipe(image, prompt=is_unusual_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('Does the image have any unusual objects:', output)
            is_unusual += extract_number(output)
            properties_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take an action.
                Question: How strong of an association does it create between product and properties?
                Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
                ASSISTANT:
                """

            output = self.pipe(image, prompt=properties_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('How strong of an association does it create between product and properties:', output)
            properties_score += extract_number(output)
            audience_prompt = \
                f"""
                <image>\n USER:
                Context: This is the message of an advertisement image convincing the audience to take the action in the sentence. The message is: {action_reason}.
                Question: Choose the best option describing the audience of this advertisement?
                Options:
                {parse_options(get_audience_list())}
                Your output must be the possible audience for the advertisement without considering the image without any further explanation. Follow the answer format of {string_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=audience_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('Who is the audience of the advertisement message:', output)
            audiences.append(output)
            audience_score_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take the action in the sentence. The audience of this image is {output}.
                You are a human rating how well the image resonate with the given audience, considering the needs of the described audience.
                Question:  How well does it resonate with its audience?
                Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=audience_score_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('How well does it resonate with its audience:', output)
            audience_score += extract_number(output)
            memorability_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take the action in the sentence.
                Question:  How memorable is the image?
                Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=memorability_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('How memorable is the image:', output)
            memorability_score += extract_number(output)
            benefit_prompt = \
                f"""
                <image>\n USER:
                Context: In advertisement designers try to show the features of their product into benefits for consumers to show how well their product improves consumers life.
                You are given an advertisement image convincing the audience to take an action, while trying to turn the features of the product into benefits.
                Question:  How well does it turn features into benefits?
                Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=benefit_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('How well does it turn features into benefits:', output)
            benefit_score += extract_number(output)
            appealing_type_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take the action in the sentence. 
                Each advertisement image, aims to either target the audiences' emotion, ethics, or logic.
                Question:  Which category is the image appealing to? emotion, ethics, or logic?
                Your output must be the category among emotion, ethics, and logic, without any further explanation. Follow the answer format of {string_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=appealing_type_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('Which category is the image appealing to:', output)
            appealing_type_prompt = \
                f"""
                Context: This is an advertisement message convincing the audience to take the action in the sentence. 
                Each advertisement message, aims to either target the audiences' emotion, ethics, or logic.
                Question:  Which category is the message of {action_reason} appealing to? emotion, ethics, or logic?
                Your output must be the category among emotion, ethics, and logic, without any further explanation. Follow the answer format of {string_answer_format}.
                ASSISTANT:
                """
            main_category = self.LLM_model(appealing_type_prompt).split(':')[-1]
            if main_category.lower() == output.lower():
                appealing_type += 1

            appealing_score_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take the action in the sentence.
                Question:  How much does the image appeal to {output}?
                Your output must be a score between -5 to 5 without any further explanation. Follow the answer format of {score_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=appealing_score_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('How appealing is the image:', output)
            appealing_score += extract_number(output)
            maslow_pyramid_needs_prompt = \
                f"""
                <image>\n USER:
                Context: This is an advertisement image convincing the audience to take the action in the sentence.
                Each ad image is designed to target one of the Maslow’s pyramid needs. The needs are:
                    - self actualisation: meeting one's full potential in life
                    - esteem: For example, respect, status, recognition, strength
                    - love/belonging: For example, friendship, intimacy, family, connections
                    - safety: For example, security, health, finance
                    - biological & physiological: For example, food, sleep, water
                Question:  What needs does it appeal to in Maslow’s pyramid? Choose among (self actualisation, esteem, love/belonging, saftey, biological & physiological)
                Your output must be the category among options, without any further explanation. Follow the answer format of {string_answer_format}.
                ASSISTANT:
                """
            output = self.pipe(image, prompt=maslow_pyramid_needs_prompt,
                               generate_kwargs={"max_new_tokens": 45})
            output = output.split(':')[-1]
            print('What needs does it appeal to in Maslow’s pyramid:', output)
            maslow_pyramid_needs_prompt = \
                f"""
                Context: This is an advertisement message convincing the audience to take the action in the sentence.
                Each ad message is designed to target one of the Maslow’s pyramid needs. The needs are:
                    - self actualisation: meeting one's full potential in life
                    - esteem: For example, respect, status, recognition, strength
                    - love/belonging: For example, friendship, intimacy, family, connections
                    - safety: For example, security, health, finance
                    - biological & physiological: For example, food, sleep, water
                Question:  What needs does the message of {action_reason} appeal to in Maslow’s pyramid? Choose among (self actualisation, esteem, love/belonging, saftey, biological & physiological)
                Your output must be the category among options, without any further explanation. Follow the answer format of {string_answer_format}.
                ASSISTANT:
                """
            main_need = self.LLM_model(maslow_pyramid_needs_prompt).split(':')[-1]
            if main_need.lower() == output.lower():
                maslow_pyramid_needs += 1
        outputs = {
            'has_story': has_story / statements_count,
            'is_unusual': is_unusual / statements_count,
            'properties_score': properties_score / statements_count,
            'audience_score': audience_score / statements_count,
            'memorability_score': memorability_score / statements_count,
            'benefit_score': benefit_score / statements_count,
            'appealing_score': appealing_score / statements_count,
            'appealing_type': appealing_type / statements_count,
            'maslow_pyramid_needs': maslow_pyramid_needs / statements_count
        }
        return outputs

    def get_GPT4v_persuasiveness_alignment(self, generated_image, action_reasons=None):
        def extract_number(string_number):
            match = re.search(r'-?\d+', string_number)
            if match:
                return int(match.group(0))
            else:
                print("No numeric value found in the input string")
                return 0

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        def get_payload(prompt, base64_image):
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 20
            }
            return payload

        if type(generated_image) != str:
            image = generated_image
        else:
            image = Image.open(generated_image).convert("RGB")
            print(generated_image.split('/'))
            image_url = '/'.join(generated_image.split('/')[-2:])
            action_reasons = self.QA[image_url][0]
        base64_image = encode_image(generated_image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        statements_count = len(action_reasons)
        action_numeric_value = 0
        reason_numeric_value = 0
        for action_reason in action_reasons:
            print(action_reason)
            action = action_reason.lower().split('because')[0]
            reason = action_reason.lower().split('because')[-1]
            answer_format = 'Answer: score'
            action_score_prompt = f"""
                    Imagine you are a human evaluating how convincing is an image ignoring the dictations and misspelling. 
                    Your task is to score how convincing an image is given an action on a scale from -10 to 10. 
                    Context: If the image convinces the audience to take an action considered convincing.
                    Question: Given the message of {action} provide the score in the following format: {answer_format}
                    Only return the score and do not explain. The only accepted output is the score.
                    """
            payload = get_payload(action_score_prompt, base64_image)
            output = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
            # print(output)
            output = output['choices'][0]["message"]['content'].split(':')[-1]
            print('action:', output)
            action_numeric_value += extract_number(output)
            reason_score_prompt = f"""
                Imagine you are a human evaluating how related an image is to a reason. Your task is to score the relatedness of an image and a reason on a scale from -10 to 10. 
                Context: Imagine the image is convincing the audience to take the action in the message of {action}. If the reason in the image is the same as the given message in the next sentence, the score is 5 and if it is totally irrelevant the score is -5.
                Question: Given the message of {reason} and the image, provide the score in the following format: {answer_format}. 
                Only return the score and do not explain. The only accepted output is the score. 
                """

            payload = get_payload(reason_score_prompt, base64_image)
            output = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
            # print(output)
            output = output['choices'][0]["message"]['content'].split(':')[-1]
            print('reason:', output)
            reason_numeric_value += extract_number(output)
        return (reason_numeric_value + action_numeric_value) / (2 * statements_count)


class Whoops:
    def __init__(self, args):
        self.args = args
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            torch_dtype="float16"
        )
        model_id_map = {
            'LLAVA': "llava-hf/llava-1.5-13b-hf",
            'LLAVA_phi': "xtuner/llava-phi-3-mini-hf",
            'VILA': "Efficient-Large-Model/VILA-7b",
            'InternVL': "OpenGVLab/InternVL-Chat-V1-5"
        }
        model_id = model_id_map[args.VLM]
        task_map = {
            'LLAVA': "image-to-text",
            'LLAVA_LLAMA': "image-to-text",
            'VILA': "text-generation",
            'InternVL': "visual-question-answering"
        }
        task = task_map[args.VLM]
        self.pipe = pipeline(task,
                             model=model_id,
                             model_kwargs={"quantization_config": quantization_config},
                             trust_remote_code=True,
                             device_map='auto',
                             token='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', )
        self.QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))

    @staticmethod
    def parse_options(options):
        return '\n'.join([f'{str(i)}. {option}' for i, option in enumerate(options)])

    @staticmethod
    def get_answer_format():
        answer_format = """
        Answer: ${index of the best option}\n
        """
        return answer_format

    def get_prompt(self, options, question=None, description=None):
        options = self.parse_options(options)
        data = {'options': options, 'question': question, 'description': description}
        env = Environment(loader=FileSystemLoader(self.args.prompt_path))
        template = env.get_template(self.args.VLM_prompt)
        prompt = template.render(**data)
        return prompt

    def get_prediction(self, image, description, QA):
        answers = []
        options = QA[-1]
        if len(QA) == 3:
            question = QA[0]
        else:
            question = None
        prompt = self.get_prompt(options, question, description)
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 45})
        output = output[0]["generated_text"].split(':')[-1]
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
        return answers
