
# script to compute fairness metrics
import argparse
import base64
import json
import os
import re
import time
from collections import Counter
from urllib import error, request

import clip
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

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

def absolute_file_paths(directory, max_seed: int | None = None):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if max_seed is not None:
                try:
                    if int(f.split('-')[0]) > max_seed:
                        continue
                except (ValueError, IndexError):
                    pass
            yield os.path.abspath(os.path.join(dirpath, f))

gender_map = {'female': 0, 'male': 1}
eyeglasses_map = {'no_eyeglasses': 0, 'eyeglasses': 1}
race_labels = ['white', 'black', 'asian', 'indian', 'latino']
race_map = {race: idx for idx, race in enumerate(race_labels)}
age_labels = ['young', 'middle-aged', 'elderly']
age_map = {age: idx for idx, age in enumerate(age_labels)}
body_labels = ['slim', 'average build', 'heavy']
body_map = {body: idx for idx, body in enumerate(body_labels)}
_HF_VQA_PIPELINE = None
_HF_VQA_PIPELINE_MODEL = None


def normalize_concept_for_attribute(concept, attribute):
    cleaned = concept.strip()
    if attribute == "gender":
        for prefix in ("female ", "male "):
            if cleaned.lower().startswith(prefix):
                return cleaned[len(prefix):].strip()
    return cleaned

def infer_concept_from_path(images_path):
    normalized = os.path.normpath(images_path)
    concept_dir = os.path.basename(normalized)
    prefix = "a photo of a "
    if concept_dir.startswith(prefix):
        return concept_dir[len(prefix):]

    parent_dir = os.path.basename(os.path.dirname(normalized))
    if parent_dir.endswith("_5cls"):
        return parent_dir[:-5].replace("_", " ")
    return concept_dir.replace("_", " ")

def infer_approach_from_path(images_path):
    normalized = os.path.normpath(images_path)
    path_parts = set(normalized.split(os.sep))
    if "eyeglasses_orig" in path_parts:
        return "orig"
    if "eyeglasses_debiased" in path_parts:
        return "debiased"
    if "results_orig" in path_parts:
        return "orig"
    if "results" in path_parts:
        return "multiplier"
    return "n/a"


def image_to_data_url(image_path):
    suffix = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_yes_no(text):
    normalized = text.strip().lower()
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    yes_match = re.search(r"\byes\b", normalized)
    no_match = re.search(r"\bno\b", normalized)
    if yes_match and not no_match:
        return True
    if no_match and not yes_match:
        return False
    raise ValueError(f"Unexpected judge response: {text!r}")


def get_retry_delay_seconds(http_error, attempt, base_sleep_seconds, max_sleep_seconds):
    if isinstance(http_error, error.HTTPError) and http_error.code == 429:
        retry_after = http_error.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(float(retry_after), max_sleep_seconds)
            except ValueError:
                pass
    return min(base_sleep_seconds * (2 ** attempt), max_sleep_seconds)


def judge_eyeglasses_with_llm(
    image_path,
    model_name,
    api_key,
    max_retries=5,
    request_sleep_seconds=0.0,
    retry_base_sleep_seconds=2.0,
    retry_max_sleep_seconds=60.0,
):
    if request_sleep_seconds > 0:
        time.sleep(request_sleep_seconds)

    image_data_url = image_to_data_url(image_path)
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 5,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict visual classifier. "
                    "Answer with exactly YES or NO."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Does a person in this image wear eyeglasses? Answer YES or NO only."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    for attempt in range(max_retries):
        try:
            with request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return parse_yes_no(text)
        except (error.HTTPError, error.URLError, TimeoutError, KeyError, ValueError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"LLM judge failed for {image_path}: {e}") from e
            delay = get_retry_delay_seconds(
                e,
                attempt,
                base_sleep_seconds=retry_base_sleep_seconds,
                max_sleep_seconds=retry_max_sleep_seconds,
            )
            print(f"LLM judge retry for {image_path} in {delay:.1f}s after error: {e}")
            time.sleep(delay)


def judge_eyeglasses_with_ollama(
    image_path,
    model_name,
    ollama_url,
    max_retries=5,
    request_sleep_seconds=0.0,
    retry_base_sleep_seconds=2.0,
    retry_max_sleep_seconds=60.0,
):
    if request_sleep_seconds > 0:
        time.sleep(request_sleep_seconds)

    image_b64 = image_to_base64(image_path)
    payload = {
        "model": model_name,
        "prompt": "Does a person in this image wear eyeglasses? Answer YES or NO only.",
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0},
    }
    req = request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(max_retries):
        try:
            with request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data.get("response", "")
            return parse_yes_no(text)
        except (error.HTTPError, error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Ollama judge failed for {image_path}: {e}") from e
            delay = get_retry_delay_seconds(
                e,
                attempt,
                base_sleep_seconds=retry_base_sleep_seconds,
                max_sleep_seconds=retry_max_sleep_seconds,
            )
            print(f"Ollama judge retry for {image_path} in {delay:.1f}s after error: {e}")
            time.sleep(delay)


def judge_eyeglasses_with_hf(
    image_path,
    model_name,
    question,
    max_retries=5,
    request_sleep_seconds=0.0,
    retry_base_sleep_seconds=2.0,
    retry_max_sleep_seconds=60.0,
):
    global _HF_VQA_PIPELINE, _HF_VQA_PIPELINE_MODEL

    if request_sleep_seconds > 0:
        time.sleep(request_sleep_seconds)

    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "Hugging Face judge requires transformers. "
            "Install with: pip install --user transformers accelerate safetensors"
        ) from e

    if _HF_VQA_PIPELINE is None or _HF_VQA_PIPELINE_MODEL != model_name:
        device = 0 if torch.cuda.is_available() else -1
        _HF_VQA_PIPELINE = pipeline("visual-question-answering", model=model_name, device=device)
        _HF_VQA_PIPELINE_MODEL = model_name

    for attempt in range(max_retries):
        try:
            outputs = _HF_VQA_PIPELINE(Image.open(image_path).convert("RGB"), question=question, top_k=1)
            if not outputs:
                raise ValueError("Empty VQA outputs")
            answer = outputs[0].get("answer", "")
            return parse_yes_no(answer)
        except (RuntimeError, ValueError, KeyError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"HF judge failed for {image_path}: {e}") from e
            delay = min(retry_base_sleep_seconds * (2 ** attempt), retry_max_sleep_seconds)
            print(f"HF judge retry for {image_path} in {delay:.1f}s after error: {e}")
            time.sleep(delay)


def main(args):
    images_path = args.images_path
    attribute = args.attribute
    concept = args.concept or infer_concept_from_path(images_path)
    concept = normalize_concept_for_attribute(concept, attribute)
    approach = args.approach or infer_approach_from_path(images_path)

    assert attribute in ["gender", "race", "eyeglasses", "age", "body"], "Only gender, race, age, body, and eyeglasses supported for now!"

    if attribute == "gender":
        text_male = f"a photo of a male"
        text_female = f"a photo of a female"
        texts = [text_male, text_female]
    elif attribute == "race":
        if args.race_template == "concept":
            # Embed the profession in the CLIP race prompt:
            # e.g. "a photo of a white CEO".
            texts = [f"a photo of a {race} {concept}" for race in race_labels]
        else:
            texts = [f"a photo of a {race} person" for race in race_labels]
    elif attribute == "age":
        texts = [f"a photo of a {age} person" for age in age_labels]
    elif attribute == "body":
        texts = [f"a photo of a {body} person" for body in body_labels]
    elif attribute == "eyeglasses":
        text_no_eyeglasses = f"a photo of a {concept}"
        text_eyeglasses = f"a photo of a {concept} wearing eyeglasses"
        texts = [text_no_eyeglasses, text_eyeglasses]

    assert os.path.exists(images_path), "The path is invalid!"
    image_paths = [
        path for path in absolute_file_paths(images_path, max_seed=args.max_seed)
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    ]
    assert len(image_paths) > 0, "No image files found in the target directory!"

    predictions = []

    use_llm_judge = (attribute == "eyeglasses" and args.eyeglasses_judge == "llm")
    use_ollama_judge = (attribute == "eyeglasses" and args.eyeglasses_judge == "ollama")
    use_hf_judge = (attribute == "eyeglasses" and args.eyeglasses_judge == "hf")
    if use_llm_judge:
        api_key = os.environ.get(args.llm_api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing API key: set environment variable {args.llm_api_key_env}"
            )
        texts_feats = None
        model = None
        image_preprocess = None
    elif use_ollama_judge:
        texts_feats = None
        model = None
        image_preprocess = None
    elif use_hf_judge:
        texts_feats = None
        model = None
        image_preprocess = None
    else:
        model, _ = clip.load("ViT-L/14", device="cuda")
        image_preprocess, text_preprocess = get_clip_preprocess(224)
        texts_feats = text_preprocess(texts).cuda()
        texts_feats = model.encode_text(texts_feats)
        texts_feats = texts_feats / texts_feats.norm(dim=-1, keepdim=True)

    for image_path in image_paths:

        try:
            if use_llm_judge:
                has_eyeglasses = judge_eyeglasses_with_llm(
                    image_path=image_path,
                    model_name=args.llm_model,
                    api_key=api_key,
                    max_retries=args.llm_max_retries,
                    request_sleep_seconds=args.llm_request_sleep,
                    retry_base_sleep_seconds=args.llm_retry_base_sleep,
                    retry_max_sleep_seconds=args.llm_retry_max_sleep,
                )
                predictions += [eyeglasses_map['eyeglasses' if has_eyeglasses else 'no_eyeglasses']]
                continue
            if use_ollama_judge:
                has_eyeglasses = judge_eyeglasses_with_ollama(
                    image_path=image_path,
                    model_name=args.ollama_model,
                    ollama_url=args.ollama_url,
                    max_retries=args.llm_max_retries,
                    request_sleep_seconds=args.llm_request_sleep,
                    retry_base_sleep_seconds=args.llm_retry_base_sleep,
                    retry_max_sleep_seconds=args.llm_retry_max_sleep,
                )
                predictions += [eyeglasses_map['eyeglasses' if has_eyeglasses else 'no_eyeglasses']]
                continue
            if use_hf_judge:
                has_eyeglasses = judge_eyeglasses_with_hf(
                    image_path=image_path,
                    model_name=args.hf_model,
                    question=args.hf_question,
                    max_retries=args.llm_max_retries,
                    request_sleep_seconds=args.llm_request_sleep,
                    retry_base_sleep_seconds=args.llm_retry_base_sleep,
                    retry_max_sleep_seconds=args.llm_retry_max_sleep,
                )
                predictions += [eyeglasses_map['eyeglasses' if has_eyeglasses else 'no_eyeglasses']]
                continue

            # extract all images
            images_feats = [image_preprocess(Image.open(image_path))]
            images_feats = torch.stack(images_feats, dim=0).cuda()
            images_feats = model.encode_image(images_feats)

            # compute the similarity
            images_feats = images_feats / images_feats.norm(dim=-1, keepdim=True)
            logits = 100.0 * images_feats @ texts_feats.T
            probs = logits.softmax(dim=-1)[0]

            if attribute == 'gender':
                if probs[0] > probs[1]:
                    predictions += [gender_map['male']]
                else:
                    predictions += [gender_map['female']]
                
            elif attribute == 'race':
                predictions += [torch.argmax(probs).item()]
            elif attribute == 'age':
                predictions += [torch.argmax(probs).item()]
            elif attribute == 'body':
                predictions += [torch.argmax(probs).item()]
            elif attribute == 'eyeglasses':
                if probs[1] > probs[0]:
                    predictions += [eyeglasses_map['eyeglasses']]
                else:
                    predictions += [eyeglasses_map['no_eyeglasses']]
            else:
                raise ValueError("Only gender, race, and eyeglasses metrics can be computed for the moment")
    
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    if len(predictions) == 0:
        print(f"Concept: {concept}, debias: {approach}, no valid predictions generated.")
        return

    if attribute == "gender":
        ratio = Counter(predictions)[0] / len(predictions)
        print(f"Concept: {concept}, debias: {approach}, female ratio: {ratio}")
    elif attribute == "race":
        counts = Counter(predictions)
        race_ratios = {race.title(): counts[idx] / len(predictions) for race, idx in race_map.items()}
        print(f"Concept: {concept}, debias: {approach}, race ratios: {race_ratios}")
    elif attribute == "age":
        counts = Counter(predictions)
        age_ratios = {age.title(): counts[idx] / len(predictions) for age, idx in age_map.items()}
        print(f"Concept: {concept}, debias: {approach}, age ratios: {age_ratios}")
    elif attribute == "body":
        counts = Counter(predictions)
        body_ratios = {body.title(): counts[idx] / len(predictions) for body, idx in body_map.items()}
        print(f"Concept: {concept}, debias: {approach}, body ratios: {body_ratios}")
    else:
        ratio = Counter(predictions)[1] / len(predictions)
        print(f"Concept: {concept}, debias: {approach}, eyeglasses ratio: {ratio}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(usage="script to compute fairness metrics on the generated images")
    parser.add_argument("--images_path", type=str, help="path of the directory where generated images are located")
    parser.add_argument("--concept", type=str, help="concept for the photo")
    parser.add_argument("--attribute", type=str, help="attribute to choose the classifier -- gender/race/eyeglasses")
    parser.add_argument("--approach", type=str, help="debias method")
    parser.add_argument("--results_path", type=str, help="path to store the fairness metrics")
    parser.add_argument("--max_seed", type=int, default=None,
                        help="Optional cap: only count files whose seed prefix (before '-') is <= max_seed.")
    parser.add_argument("--eyeglasses_judge", type=str, default="llm", choices=["clip", "llm", "ollama", "hf"], help="judge backend for eyeglasses metric")
    parser.add_argument("--race_template", type=str, default="person", choices=["person", "concept"], help="CLIP race prompt template: 'person' uses 'a photo of a {race} person'; 'concept' uses 'a photo of a {race} {concept}'")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="OpenAI model for LLM judge")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY", help="env var name containing OpenAI API key")
    parser.add_argument("--llm_max_retries", type=int, default=5, help="max retries for LLM judge calls")
    parser.add_argument("--llm_request_sleep", type=float, default=0.3, help="sleep between LLM requests in seconds")
    parser.add_argument("--llm_retry_base_sleep", type=float, default=2.0, help="base backoff sleep in seconds")
    parser.add_argument("--llm_retry_max_sleep", type=float, default=60.0, help="max backoff sleep in seconds")
    parser.add_argument("--ollama_model", type=str, default="llava", help="Ollama vision model for local judging")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate", help="Ollama generate endpoint URL")
    parser.add_argument("--hf_model", type=str, default="dandelin/vilt-b32-finetuned-vqa", help="HF VQA model for eyeglasses judging")
    parser.add_argument("--hf_question", type=str, default="Is any person in this image wearing eyeglasses? Answer yes or no.", help="Question used for HF VQA judging")
    args = parser.parse_args()
    main(args)