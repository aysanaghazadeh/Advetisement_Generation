import pandas as pd
import json

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
QA = json.load(open('../Data/PittAd/train/QA_Combined_Action_Reason_train.json'))
# Prepare the image and text
image_path = "/Users/aysanaghazadeh/experiments/33576.jpg"  # Change to your image path
image = Image.open(image_path)
for AR in QA['6/33576.jpg'][0]:
    reason = AR.lower().split('because')[1]
    action = AR.lower().split('because')[0]
    inputs_image = processor(images=image, return_tensors="pt", padding=True)  # Change to your text
    inputs_text = processor(text=reason, return_tensors="pt", padding=True)
    # Process images
    # inputs = processor(images=[image1, image2], return_tensors="pt", padding=True).to(device=args.device)

    # Extract image features from the CLIP model
    with torch.no_grad():
        image_features = model.get_image_features(**inputs_image)
        text_features = model.get_text_features(**inputs_text)
    # Normalize the feature vectors
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

    # Calculate cosine similarity between the two images
    cosine_similarity = torch.nn.functional.cosine_similarity(image_features,
                                                              text_features).item()

    print(cosine_similarity)
    print(AR)

