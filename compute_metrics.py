# script to compute fairness metrics
import argparse
import os
from collections import Counter
import clip
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from PIL import Image
import torch

def get_clip_preprocess(n_px=224):
    def Convert(image):
        return image.convert("RGB")

    image_preprocess = Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def text_preprocess(text):
        return clip.tokenize(text, truncate=True)

    return image_preprocess, text_preprocess

def absolute_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

gender_map = {'female': 0, 'male': 1}
race_map = {'black': 0, 'white': 1}

def main(args):
    images_path = args.images_path
    attribute = args.attribute
    concept = args.concept

    assert attribute in ["gender", "race"], "Only gender and race supported for now!"

    if attribute == "gender":
        text_male = f"a photo of a male {concept}"
        text_female = f"a photo of a female {concept}"
        texts = [text_male, text_female]

    assert os.path.exists(images_path), "The path is invalid!"
    image_paths = absolute_file_paths(images_path)
    model, _ = clip.load("ViT-B/32", device="cuda")

    image_preprocess, text_preprocess = get_clip_preprocess(
        224
    )

    predictions = []

    for image_path in image_paths:

        texts_feats = text_preprocess(texts).cuda()
        texts_feats = model.encode_text(texts_feats)
    
        # extract all images
        images_feats = [image_preprocess(Image.open(image_path))]
        images_feats = torch.stack(images_feats, dim=0).cuda()
        images_feats = model.encode_image(images_feats)
    
        # compute the similarity
        images_feats = images_feats / images_feats.norm(dim=-1, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=-1, keepdim=True)
        
        logits = 100.0 * images_feats @ texts_feats.T
        probs = logits.softmax(dim=-1)[0]

        if attribute == 'gender':
            if probs[0] > probs[1]:
                predictions += [gender_map['male']]
            else:
                predictions += [gender_map['female']]
            
        else:
            raise ValueError("Only gender metrics can be computed for the moment")
    
    print(f"Concept: {args.concept}, debias: {args.approach}, female ratio: {Counter(predictions)[0]/100}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(usage="script to compute fairness metrics on the generated images")
    parser.add_argument("--images_path", type=str, help="path of the directory where generated images are located")
    parser.add_argument("--concept", type=str, help="concept for the photo")
    parser.add_argument("--attribute", type=str, help="attribute to choose the classifier -- gender/race")
    parser.add_argument("--approach", type=str, help="debias method")
    parser.add_argument("--results_path", type=str, help="path to store the fairness metrics")
    args = parser.parse_args()
    main(args)