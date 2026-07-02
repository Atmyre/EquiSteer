"""Add BLIP-VQA labels to the existing eyeglasses calibration outputs.

The paper's appendix (§11.2) reports a second eyeglasses classifier:
``Salesforce/blip-vqa-capfilt-large`` with the question
"Is any person in this image wearing eyeglasses? Answer yes or no.".

This script:
  1. Loads `calibration_<bb>_eyeglasses.csv` (the merged output of the CLIP +
     GPT-4o calibration we already ran).
  2. For every image in the manifest, runs BLIP-VQA with the appendix
     question and decodes yes/no into {eyeglasses, no_eyeglasses, uncertain}.
  3. Writes back the manifest with a new ``blip_label`` column and a small
     summary JSON with the three pairwise agreement matrices
     (CLIP-BLIP, CLIP-GPT4o, BLIP-GPT4o) and per-mode / per-class breakdowns.
  4. Re-computes parity gap Delta under each classifier so we can compare
     vanilla / FairSteer numbers under CLIP, BLIP, and GPT-4o on identical
     data.
"""
import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_classifier import ATTRIBUTE_CONFIG, parity_gap


QUESTION = "Is any person in this image wearing eyeglasses? Answer yes or no."
MODEL_NAME = "Salesforce/blip-vqa-capfilt-large"


def parse_answer(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes") or " yes" in f" {t} ":
        return "eyeglasses"
    if t.startswith("no") or " no" in f" {t} ":
        return "no_eyeglasses"
    return "uncertain"


def run_blip(model, processor, paths, device):
    labels = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            labels.append("uncertain")
            print(f"[skip] {path}: {e}")
            continue
        inputs = processor(img, QUESTION, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        answer = processor.decode(out[0], skip_special_tokens=True)
        labels.append(parse_answer(answer))
    return labels


def cell_summary(df: pd.DataFrame) -> dict:
    classes = ATTRIBUTE_CONFIG["eyeglasses"]["classes"]
    target = ATTRIBUTE_CONFIG["eyeglasses"]["target_uniform"]
    out = {}

    for label_col in ("clip_pred", "oracle_pred", "blip_label"):
        if label_col not in df.columns:
            continue
        sub = df[df[label_col].isin(classes)]
        out[f"n_{label_col}"] = int(len(sub))
        out[f"delta_{label_col}"] = parity_gap(sub, label_col, classes, target)

    # Pairwise agreement (only over rows where both classifiers gave a class label)
    pair_specs = [
        ("clip_blip",   "clip_pred",   "blip_label"),
        ("clip_oracle", "clip_pred",   "oracle_pred"),
        ("blip_oracle", "blip_label",  "oracle_pred"),
    ]
    for name, a, b in pair_specs:
        if a not in df.columns or b not in df.columns:
            continue
        sub = df[df[a].isin(classes) & df[b].isin(classes)]
        if not len(sub):
            continue
        d = {
            "n": int(len(sub)),
            "agreement": float((sub[a] == sub[b]).mean()),
            "per_mode": {m: float((g[a] == g[b]).mean())
                         for m, g in sub.groupby("mode")},
        }
        out[f"agreement_{name}"] = d

    # Per-class recall (rows of the form "GPT-4o says X; what does CLIP/BLIP say?")
    for ref, ref_col in (("oracle", "oracle_pred"), ("clip", "clip_pred")):
        for tgt, tgt_col in (("clip", "clip_pred"), ("blip", "blip_label")):
            if ref_col not in df.columns or tgt_col not in df.columns or ref_col == tgt_col:
                continue
            recall = {}
            for cls in classes:
                m = df[df[ref_col] == cls]
                if len(m):
                    recall[cls] = {"n": int(len(m)),
                                    "recall": float((m[tgt_col] == cls).mean())}
            out[f"recall_{ref}_to_{tgt}"] = recall

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration_root",
                   default="/gpfs/scratch/acw685/CA_diffusion_debiasing-main/calibration")
    p.add_argument("--backbone", required=True, choices=["sdxl", "sana15"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cell_dir = os.path.join(args.calibration_root, f"{args.backbone}_eyeglasses")
    csv_path = os.path.join(cell_dir, f"calibration_{args.backbone}_eyeglasses.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"missing {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[load] {csv_path}: {len(df)} rows")

    print(f"[blip] loading {MODEL_NAME} on {args.device}")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()

    df["blip_label"] = run_blip(model, processor, df["path"].tolist(), args.device)
    df.to_csv(csv_path, index=False)

    summary = cell_summary(df)
    summary["backbone"] = args.backbone
    summary["attribute"] = "eyeglasses"
    summary["n_total"] = int(len(df))
    summary["blip_label_counts"] = dict(Counter(df["blip_label"]))

    out_json = os.path.join(cell_dir, f"calibration_{args.backbone}_eyeglasses_blip.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {out_json}")
    print(json.dumps({"agreement_clip_blip":   summary.get("agreement_clip_blip"),
                      "agreement_clip_oracle": summary.get("agreement_clip_oracle"),
                      "agreement_blip_oracle": summary.get("agreement_blip_oracle"),
                      "blip_label_counts":     summary.get("blip_label_counts")},
                     indent=2))


if __name__ == "__main__":
    main()
