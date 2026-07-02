"""Classifier-calibration analysis for the FairSteer rebuttal.

Compares CLIP-ViT-L/14 zero-shot classification (used in the paper) to a strong
VLM oracle (GPT-4o by default) on a stratified sample of generated images, for
gender / race / eyeglasses, and reports:

  - Per-cell (profession, mode) agreement rate.
  - Per-class recall and per-class CLIP bias (over- / under-predicting any class).
  - Per-mode bias check: agreement on vanilla vs FairSteer images (rules out a
    classifier-OOD confounder for the paper's Delta numbers).
  - Reliability diagram for the binary attributes (gender, eyeglasses) of CLIP
    softmax confidence vs empirical accuracy from the oracle.
  - Recomputed parity gap Delta using oracle labels and CLIP labels.

Inputs:
  --root_vanilla   --root_fairsteer
  --attribute {gender|race|eyeglasses}
  --backbone {sdxl|sana15}
  --professions    --n_per_cell    --seed

Each cell = (profession, mode in {vanilla, fairsteer}).
"""
import argparse
import base64
import json
import os
import random
import time
from collections import Counter, defaultdict
from glob import glob
from urllib import error, request

import clip
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix as sklearn_cm
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


# -------------------- attribute config --------------------------

ATTRIBUTE_CONFIG = {
    "gender": {
        "classes": ["male", "female"],
        "clip_prompts": ["a photo of a male", "a photo of a female"],
        "clip_paraphrase_prompts": ["a man", "a woman"],
        "target_uniform": 0.5,
        "gpt4o_system": ("You are a strict visual classifier. Identify the apparent "
                          "gender of the most prominent person in the image. "
                          "Answer with exactly one of: MALE, FEMALE, UNCERTAIN."),
        "gpt4o_user": ("What is the apparent gender of the person in this image? "
                        "Answer with one word: MALE, FEMALE, or UNCERTAIN."),
        "gpt4o_labels": ["MALE", "FEMALE", "UNCERTAIN"],
    },
    "race": {
        "classes": ["white", "black", "asian", "indian", "latino"],
        "clip_prompts": [
            "a photo of a white person",
            "a photo of a black person",
            "a photo of an asian person",
            "a photo of an indian person",
            "a photo of a latino person",
        ],
        "clip_paraphrase_prompts": [
            "a white person", "a black person", "an asian person",
            "an indian person", "a latino person",
        ],
        "target_uniform": 0.2,
        "gpt4o_system": ("You are a strict visual classifier. Identify the most "
                          "likely apparent racial / ethnic group of the most prominent "
                          "person. Answer with exactly one of: WHITE, BLACK, ASIAN, "
                          "INDIAN, LATINO, UNCERTAIN."),
        "gpt4o_user": ("What is the apparent racial / ethnic group of the most "
                        "prominent person? Answer with one word: WHITE, BLACK, "
                        "ASIAN, INDIAN, LATINO, or UNCERTAIN."),
        "gpt4o_labels": ["WHITE", "BLACK", "ASIAN", "INDIAN", "LATINO", "UNCERTAIN"],
    },
    "eyeglasses": {
        "classes": ["no_eyeglasses", "eyeglasses"],
        # CLIP prompt set will be specialised per-profession in compute_clip_text_feats
        "clip_prompts": None,
        "clip_paraphrase_prompts": None,
        "target_uniform": 0.5,
        "gpt4o_system": ("You are a strict visual classifier. Determine whether the "
                          "most prominent person in the image is wearing eyeglasses. "
                          "Answer with exactly one of: YES, NO, UNCERTAIN."),
        "gpt4o_user": ("Is the most prominent person in this image wearing eyeglasses? "
                        "Answer with one word: YES, NO, or UNCERTAIN."),
        "gpt4o_labels": ["YES", "NO", "UNCERTAIN"],
    },
    "age": {
        "classes": ["young", "middle-aged", "elderly"],
        "clip_prompts": [
            "a photo of a young person",
            "a photo of a middle-aged person",
            "a photo of an elderly person",
        ],
        "clip_paraphrase_prompts": [
            "a young person",
            "a middle aged adult",
            "an old person",
        ],
        "target_uniform": 1.0 / 3.0,
        "gpt4o_system": ("You are a strict visual classifier. Estimate the apparent "
                          "age range of the most prominent person in the image. "
                          "Answer with exactly one of: YOUNG, MIDDLE-AGED, ELDERLY, UNCERTAIN."),
        "gpt4o_user": ("What is the apparent age range of the most prominent person? "
                        "YOUNG covers approximately 18-30; MIDDLE-AGED 31-60; ELDERLY 60+. "
                        "Answer with one word: YOUNG, MIDDLE-AGED, ELDERLY, or UNCERTAIN."),
        "gpt4o_labels": ["YOUNG", "MIDDLE-AGED", "ELDERLY", "UNCERTAIN"],
    },
    "body": {
        "classes": ["slim", "average build", "heavy"],
        "clip_prompts": [
            "a photo of a slim person",
            "a photo of a average build person",
            "a photo of a heavy person",
        ],
        "clip_paraphrase_prompts": [
            "a thin person",
            "a person of average build",
            "an overweight person",
        ],
        "target_uniform": 1.0 / 3.0,
        "gpt4o_system": ("You are a strict visual classifier. Estimate the apparent "
                          "body type of the most prominent person in the image. "
                          "Answer with exactly one of: SLIM, AVERAGE, HEAVY, UNCERTAIN."),
        "gpt4o_user": ("What is the apparent body type of the most prominent person? "
                        "SLIM = visibly thin / slender; AVERAGE = neither slim nor heavy; "
                        "HEAVY = visibly overweight or large build. "
                        "Answer with one word: SLIM, AVERAGE, HEAVY, or UNCERTAIN."),
        "gpt4o_labels": ["SLIM", "AVERAGE", "HEAVY", "UNCERTAIN"],
    },
}


def _gpt4o_label_to_class(label: str, attribute: str) -> str:
    label = label.strip().upper()
    if attribute == "eyeglasses":
        if "YES" in label:
            return "eyeglasses"
        if "NO" in label:
            return "no_eyeglasses"
        return "uncertain"
    if attribute == "gender":
        if "MALE" in label and "FEMALE" not in label:
            return "male"
        if "FEMALE" in label:
            return "female"
        return "uncertain"
    if attribute == "race":
        for cls in ("WHITE", "BLACK", "ASIAN", "INDIAN", "LATINO"):
            if cls in label:
                return cls.lower()
        return "uncertain"
    if attribute == "age":
        # check MIDDLE first because "MIDDLE-AGED" contains "AGE" which is a
        # substring of other labels
        if "MIDDLE" in label or "MIDDLE-AGED" in label:
            return "middle-aged"
        if "ELDERLY" in label or "OLD" in label or "SENIOR" in label:
            return "elderly"
        if "YOUNG" in label:
            return "young"
        return "uncertain"
    if attribute == "body":
        if "AVERAGE" in label or "MEDIUM" in label or "NORMAL" in label:
            return "average build"
        if "HEAVY" in label or "OVERWEIGHT" in label or "OBESE" in label or "FAT" in label or "LARGE" in label:
            return "heavy"
        if "SLIM" in label or "THIN" in label or "SKINNY" in label or "SLENDER" in label:
            return "slim"
        return "uncertain"
    raise ValueError(attribute)


# -------------------- CLIP --------------------------

def get_clip_preprocess(n_px=224):
    def _convert(image):
        return image.convert("RGB")
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        _convert,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def encode_text_set(prompts, model, device):
    tok = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tok)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def compute_clip_text_feats(model, attribute: str, profession: str, device):
    """Return (text_feats_default, text_feats_paraphrase, classes)."""
    cfg = ATTRIBUTE_CONFIG[attribute]
    classes = cfg["classes"]
    if attribute == "eyeglasses":
        # eyeglasses CLIP prompts depend on the profession
        prof_clean = profession.replace("_", " ")
        default = [f"a photo of a {prof_clean}",
                   f"a photo of a {prof_clean} wearing eyeglasses"]
        paraphrase = [f"a {prof_clean} not wearing glasses",
                      f"a {prof_clean} with glasses on"]
    else:
        default = cfg["clip_prompts"]
        paraphrase = cfg["clip_paraphrase_prompts"]
    return (encode_text_set(default, model, device),
            encode_text_set(paraphrase, model, device),
            classes)


def clip_classify(image_path, model, preprocess, text_feats, classes, device):
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(img)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = 100.0 * feats @ text_feats.T
        probs = logits.softmax(dim=-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), {classes[i]: float(probs[i]) for i in range(len(classes))}


# -------------------- GPT-4o oracle --------------------------

def image_to_data_url(image_path):
    suffix = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"


def gpt4o_classify(image_path, attribute, model_name, api_key, max_retries=5,
                   request_sleep_seconds=0.3,
                   retry_base_sleep_seconds=2.0,
                   retry_max_sleep_seconds=60.0):
    if request_sleep_seconds > 0:
        time.sleep(request_sleep_seconds)
    cfg = ATTRIBUTE_CONFIG[attribute]
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": cfg["gpt4o_system"]},
            {"role": "user",
             "content": [
                 {"type": "text", "text": cfg["gpt4o_user"]},
                 {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
             ]},
        ],
    }
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    for attempt in range(max_retries):
        try:
            with request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return _gpt4o_label_to_class(text, attribute)
        except (error.HTTPError, error.URLError, TimeoutError, KeyError, ValueError) as e:
            if attempt == max_retries - 1:
                print(f"[oracle] failed for {image_path}: {e}")
                return "uncertain"
            time.sleep(min(retry_base_sleep_seconds * (2 ** attempt),
                           retry_max_sleep_seconds))


# -------------------- sampling --------------------------

def sample_paths(root: str, professions: list[str], n_per_cell: int,
                 rng: random.Random,
                 prof_subdir_template: str = "{prof}",
                 shard_idx: int = 0, n_shards: int = 1) -> dict[str, list[str]]:
    """{profession: [image_paths]} sampled from {root}/{prof_subdir_template.format(prof=...)}/.

    Sharding: with shard_idx=k and n_shards=N, after a deterministic shuffle,
    return paths[k*n_per_cell : (k+1)*n_per_cell]. Combined across all N shards
    gives a contiguous window of N*n_per_cell unique paths."""
    out = {}
    for prof in professions:
        sub = prof_subdir_template.format(prof=prof)
        prof_dir = os.path.join(root, sub)
        all_paths = (glob(os.path.join(prof_dir, "**", "*.png"), recursive=True)
                     + glob(os.path.join(prof_dir, "**", "*.jpg"), recursive=True))
        rng.shuffle(all_paths)
        offset = shard_idx * n_per_cell
        picked = all_paths[offset : offset + n_per_cell]
        if len(picked) < n_per_cell:
            print(f"[warn] only {len(picked)} images for {prof_dir} shard "
                  f"{shard_idx}/{n_shards} (wanted {n_per_cell})")
        out[prof] = picked
    return out


# -------------------- metrics --------------------------

def parity_gap(df: pd.DataFrame, label_col: str, classes: list[str],
               target: float) -> dict:
    """For each mode, average over professions of |p_class - target| over each class."""
    out = {}
    for mode, mode_df in df.groupby("mode"):
        mode_df = mode_df[mode_df[label_col].isin(classes)]
        per_prof_class = []
        for prof, g in mode_df.groupby("profession"):
            n = len(g)
            if n == 0:
                continue
            for cls in classes:
                p_cls = float((g[label_col] == cls).sum() / n)
                per_prof_class.append({"profession": prof, "class": cls,
                                       "p": p_cls, "delta": abs(p_cls - target),
                                       "n": n})
        if not per_prof_class:
            continue
        # Match the paper's averaging: mean over (prof, class) of |p - target|.
        avg_delta = float(np.mean([r["delta"] for r in per_prof_class]))
        out[mode] = {"avg_delta": avg_delta, "per_prof_class": per_prof_class}
    return out


def confusion(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    sub = df[df["clip_pred"].isin(classes) & df["oracle_pred"].isin(classes)]
    cm = sklearn_cm(sub["clip_pred"], sub["oracle_pred"], labels=classes)
    return pd.DataFrame(cm, index=classes, columns=classes)


def reliability_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability for binary attributes (uses CLIP confidence in [1/n_classes, 1])."""
    if "clip_conf" not in df.columns:
        return pd.DataFrame()
    df = df[df["oracle_pred"].notna() & df["clip_pred"].notna()
            & (df["oracle_pred"] != "uncertain")].copy()
    df["correct"] = (df["clip_pred"] == df["oracle_pred"]).astype(int)
    lo = float(df["clip_conf"].min())
    hi = float(df["clip_conf"].max())
    if hi <= lo:
        return pd.DataFrame()
    df["bin"] = pd.cut(df["clip_conf"], bins=np.linspace(lo, hi, n_bins + 1),
                       include_lowest=True)
    rows = []
    for b, g in df.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        rows.append({"bin_left": b.left, "bin_right": b.right, "n": int(len(g)),
                     "mean_conf": float(g["clip_conf"].mean()),
                     "empirical_acc": float(g["correct"].mean())})
    return pd.DataFrame(rows)


# -------------------- plots --------------------------

def plot_reliability(rel: pd.DataFrame, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if not rel.empty:
        lo = float(rel["mean_conf"].min())
        hi = float(rel["mean_conf"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5,
                label="perfectly calibrated")
        ax.plot(rel["mean_conf"], rel["empirical_acc"], "o-", color="C0",
                label="CLIP vs oracle")
        for _, r in rel.iterrows():
            ax.annotate(f"n={r['n']}", (r["mean_conf"], r["empirical_acc"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="gray")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("CLIP softmax confidence")
    ax.set_ylabel("Empirical accuracy (vs oracle)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion(cm: pd.DataFrame, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(0.6 * len(cm) + 3, 0.6 * len(cm) + 3))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Oracle (GPT-4o)")
    ax.set_ylabel("CLIP-ViT-L/14")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# -------------------- main --------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_vanilla", required=True)
    p.add_argument("--root_fairsteer", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--attribute", required=True,
                   choices=["gender", "race", "eyeglasses", "age", "body"])
    p.add_argument("--professions", nargs="+",
                   default=["CEO", "doctor", "pilot", "technician",
                            "teacher", "librarian", "nurse", "fashion_designer"])
    p.add_argument("--n_per_cell", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0,
                   help="Which slice of the shuffled image list to take.")
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--oracle", choices=["gpt4o", "clip_paraphrase"], default="gpt4o")
    p.add_argument("--openai_model", default="gpt-4o")
    p.add_argument("--openai_api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--prof_subdir_vanilla", default="{prof}")
    p.add_argument("--prof_subdir_fairsteer", default="{prof}")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    cfg = ATTRIBUTE_CONFIG[args.attribute]
    classes = cfg["classes"]
    target_uniform = cfg["target_uniform"]

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-L/14", device=device)
    preprocess = get_clip_preprocess(224)

    api_key = None
    if args.oracle == "gpt4o":
        api_key = os.environ.get(args.openai_api_key_env)
        if not api_key:
            raise SystemExit(f"Set ${args.openai_api_key_env} to use --oracle gpt4o")

    samples = {
        "vanilla":   sample_paths(args.root_vanilla,   args.professions,
                                  args.n_per_cell, random.Random(args.seed),
                                  prof_subdir_template=args.prof_subdir_vanilla,
                                  shard_idx=args.shard_idx, n_shards=args.n_shards),
        "fairsteer": sample_paths(args.root_fairsteer, args.professions,
                                  args.n_per_cell, random.Random(args.seed),
                                  prof_subdir_template=args.prof_subdir_fairsteer,
                                  shard_idx=args.shard_idx, n_shards=args.n_shards),
    }

    # CLIP text encodings: cache by profession (eyeglasses) or compute once
    text_cache = {}
    rows = []
    for mode, prof_to_paths in samples.items():
        for prof, paths in prof_to_paths.items():
            if prof not in text_cache:
                tf, tfp, _ = compute_clip_text_feats(model, args.attribute, prof, device)
                text_cache[prof] = (tf, tfp)
            tf, tfp = text_cache[prof]
            for path in paths:
                clip_pred, clip_conf, clip_probs = clip_classify(
                    path, model, preprocess, tf, classes, device)
                clip_para_pred, clip_para_conf, _ = clip_classify(
                    path, model, preprocess, tfp, classes, device)
                if args.oracle == "gpt4o":
                    oracle_pred = gpt4o_classify(path, args.attribute,
                                                  args.openai_model, api_key)
                else:
                    oracle_pred = clip_para_pred
                row = {"path": path, "profession": prof, "mode": mode,
                       "clip_pred": clip_pred, "clip_conf": clip_conf,
                       "clip_paraphrase_pred": clip_para_pred,
                       "clip_paraphrase_conf": clip_para_conf,
                       "oracle_pred": oracle_pred}
                for c in classes:
                    row[f"p_{c}"] = clip_probs.get(c, float("nan"))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir,
                           f"calibration_{args.backbone}_{args.attribute}.csv"),
              index=False)

    df_clf = df[df["oracle_pred"].isin(classes)]

    overall_agreement = float((df_clf["clip_pred"] == df_clf["oracle_pred"]).mean()) \
        if len(df_clf) > 0 else float("nan")
    per_mode_agreement = {m: float((g["clip_pred"] == g["oracle_pred"]).mean())
                           for m, g in df_clf.groupby("mode")}
    per_cell_agreement = {f"{m}/{p}": {"agreement": float(
                              (g["clip_pred"] == g["oracle_pred"]).mean()),
                                       "n": int(len(g))}
                          for (m, p), g in df_clf.groupby(["mode", "profession"])}

    cm = confusion(df_clf, classes)
    rel = reliability_bins(df_clf, n_bins=10)
    rel.to_csv(os.path.join(args.output_dir,
                            f"reliability_{args.backbone}_{args.attribute}.csv"),
               index=False)

    delta_clip   = parity_gap(df_clf, "clip_pred",   classes, target_uniform)
    delta_oracle = parity_gap(df_clf, "oracle_pred", classes, target_uniform)

    per_class = {}
    for cls in classes:
        sub = df_clf[df_clf["oracle_pred"] == cls]
        if len(sub):
            per_class[cls] = {"n": int(len(sub)),
                              "clip_recall": float((sub["clip_pred"] == cls).mean())}

    summary = {
        "backbone": args.backbone, "attribute": args.attribute,
        "n_total": int(len(df)),
        "n_clf": int(len(df_clf)),
        "n_uncertain": int((df["oracle_pred"] == "uncertain").sum()),
        "overall_agreement": overall_agreement,
        "per_mode_agreement": per_mode_agreement,
        "per_class_recall": per_class,
        "per_cell_agreement": per_cell_agreement,
        "delta_clip": delta_clip,
        "delta_oracle": delta_oracle,
        "confusion_matrix": cm.to_dict(),
    }
    with open(os.path.join(args.output_dir,
                           f"calibration_{args.backbone}_{args.attribute}.json"),
              "w") as f:
        json.dump(summary, f, indent=2)

    if not rel.empty:
        plot_reliability(rel,
                         os.path.join(args.output_dir,
                                      f"reliability_{args.backbone}_{args.attribute}.png"),
                         f"CLIP calibration vs {args.oracle.upper()} "
                         f"({args.backbone} / {args.attribute})")
    if not cm.empty:
        plot_confusion(cm,
                       os.path.join(args.output_dir,
                                    f"confusion_{args.backbone}_{args.attribute}.png"),
                       f"CLIP vs {args.oracle.upper()} ({args.backbone} / {args.attribute})")

    print(json.dumps({
        "attribute": args.attribute, "backbone": args.backbone,
        "overall_agreement": overall_agreement,
        "per_mode_agreement": per_mode_agreement,
        "per_class_recall": per_class,
        "delta_clip_avg": {m: v["avg_delta"] for m, v in delta_clip.items()},
        "delta_oracle_avg": {m: v["avg_delta"] for m, v in delta_oracle.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
