from transformers import CLIPProcessor, CLIPModel, LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import tempfile
from transformers import pipeline


# Function to convert an image file to a tensor
def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    return tensor


class Metrics:
    def __init__(self, args):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device=args.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def get_FID(generated_image_path, real_image_path, args):
        # Convert images to tensors
        image_tensor_1 = image_to_tensor(generated_image_path)
        image_tensor_2 = image_to_tensor(real_image_path)

        # You need to save these tensors as images in a directory as `pytorch_fid` works with image paths

        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path_1 = os.path.join(tempdir, 'set1')
            dataset_path_2 = os.path.join(tempdir, 'set2')

            os.makedirs(dataset_path_1, exist_ok=True)
            os.makedirs(dataset_path_2, exist_ok=True)

            # Save the tensors as images
            TF.to_pil_image(image_tensor_1.squeeze()).save(os.path.join(dataset_path_1, 'image1.png'))
            TF.to_pil_image(image_tensor_2.squeeze()).save(os.path.join(dataset_path_2, 'image2.png'))

            # Calculate FID score
            fid_value = calculate_fid_given_paths([dataset_path_1, dataset_path_2], batch_size=1,
                                                  device=torch.device(args.device),
                                                  dims=2048)

            return fid_value

    def get_image_image_CLIP_score(self, generated_image_path, real_image_path, args):
        # Load images
        image1 = Image.open(real_image_path).convert("RGB")
        image2 = Image.open(generated_image_path).convert("RGB")

        # Process images
        inputs = self.clip_processor(images=[image1, image2], return_tensors="pt", padding=True).to(device=args.device)

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
            inputs_image = self.clip_processor(images=image, return_tensors="pt", padding=True).to(device=args.device)
            inputs_text = self.clip_processor(text=AR, return_tensors="pt", padding=True).to(device=args.device)
            inputs_reason = self.clip_processor(text=reason, return_tensors="pt", padding=True).to(device=args.device)
            inputs_action = self.clip_processor(text=action, return_tensors="pt", padding=True).to(device=args.device)

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
        cosine_similarity['text'] = cosine_similarity['text']/len(action_reason)
        cosine_similarity['reason'] = cosine_similarity['reason']/len(action_reason)
        cosine_similarity['action'] = cosine_similarity['action']/len(action_reason)
        return cosine_similarity

    def get_action_reason_image_CLIP_score(self, generated_image_path, action_reason, args):
        image = Image.open(generated_image_path).convert("RGB")
        cosine_similarity = 0
        for AR in action_reason:
            inputs_image = self.clip_processor(images=image, return_tensors="pt", padding=True).to(device=args.device)
            inputs_text = self.clip_processor(text=AR, return_tensors="pt", padding=True).to(device=args.device)

            # Process images
            # inputs = processor(images=[image1, image2], return_tensors="pt", padding=True).to(device=args.device)

            # Extract image features from the CLIP model
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs_image)
                text_features = self.clip_model.get_text_features(**inputs_text)
            # Normalize the feature vectors
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

            # Calculate cosine similarity between the two images
            cosine_similarity += torch.nn.functional.cosine_similarity(image_features, text_features).item()
        cosine_similarity = cosine_similarity/len(action_reason)
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
        avg_image_score = sum(image_scores)/len(image_scores)
        text_score = self.get_action_reason_image_CLIP_score(generated_image_path, text_description, args)
        creativity = text_score/(avg_image_score + 0.1)
        return creativity


class PersuasivenessMetric:
    def __init__(self):
        model_id = "llava-hf/llava-1.5-7b-hf"
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map='auto')
        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_persuasiveness_score(self, generated_image_path):
        image = Image.open(generated_image_path).convert("RGB")
        prompt = """
        <image>\n USER:
        Context: If the image convinces the audience to take an action like buying a product, etc, then the image is considered persuasive.
        Question: Based on the context score the persuasiveness of the image in range 0-10.
        Your output format is only Answer: score\n form, no other form. Empty is not allowed.
        ASSISTANT:
        """
        # outputs = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to('cuda')
        # Generate
        generate_ids = self.model.generate(**inputs, max_length=200)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = output.strip().split(':')[-1]
        if output.strip().isnumeric():
            persuasiveness = int(output)
        else:
            persuasiveness = 5
            print(output)
        return persuasiveness
