from pytorch_fid.fid_score import calculate_fid_given_paths
import clip
from PIL import Image
import torch


def get_FID(generated_image_path, real_image_path, args):
    fid_value = calculate_fid_given_paths([[real_image_path], [generated_image_path]],
                                          device=args.device)
    return fid_value


def get_image_image_CLIP_score(generated_image_path, real_image_path, args):
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    # Preprocess the images
    image1 = preprocess(Image.open(generated_image_path)).unsqueeze(0).to(args.device)
    image2 = preprocess(Image.open(real_image_path)).unsqueeze(0).to(args.device)

    # Compute the features
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    # Normalize the features
    image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
    image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

    # Compute the cosine similarity
    cosine_similarity = (image_features1 @ image_features2.T).item()

    return cosine_similarity


def get_text_image_CLIP_score(generated_image_path, text_description, args):
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    image = preprocess(Image.open(generated_image_path)).unsqueeze(0).to(args.device)
    text_tokens = clip.tokenize([text_description]).to(args.device)

    # Compute the image and text features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    # Compute the similarity score (cosine similarity)
    similarity_score = (image_features @ text_features.T).softmax(dim=-1)

    return similarity_score


def get_scores(text_description, generated_image_path, real_image_path, args):
    scores = {'image_image_CLIP_score': get_image_image_CLIP_score(generated_image_path, real_image_path, args),
              'image_text_CLIP_score': get_text_image_CLIP_score(generated_image_path, text_description, args),
              'FID_score': get_FID(generated_image_path, real_image_path, args)}

    return scores
