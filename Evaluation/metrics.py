from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import tempfile

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

    def get_text_image_CLIP_score(self, generated_image_path, text_description, args):
        similarity_score = 0

        return similarity_score

    def get_scores(self, text_description, generated_image_path, real_image_path, args):
        scores = {
            'image_image_CLIP_score': self.get_image_image_CLIP_score(generated_image_path, real_image_path, args),
            'image_text_CLIP_score': self.get_text_image_CLIP_score(generated_image_path, text_description, args),
            'FID_score': self.get_FID(generated_image_path, real_image_path, args)}

        return scores
