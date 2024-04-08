# from torcheval.metrics import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


class Metrics:
    def __init__(self, args):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device=args.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def get_FID(generated_image_path, real_image_path, args):
        # fid_value = calculate_fid_given_paths([[real_image_path], [generated_image_path]],
        #                                       device=args.device)
        fid_value = 0
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

    def get_text_image_CLIP_score(self, generated_image_path, text_description, args):
        similarity_score = 0

        return similarity_score

    def get_scores(self, text_description, generated_image_path, real_image_path, args):
        scores = {
            'image_image_CLIP_score': self.get_image_image_CLIP_score(generated_image_path, real_image_path, args),
            'image_text_CLIP_score': self.get_text_image_CLIP_score(generated_image_path, text_description, args),
            'FID_score': self.get_FID(generated_image_path, real_image_path, args)}

        return scores
