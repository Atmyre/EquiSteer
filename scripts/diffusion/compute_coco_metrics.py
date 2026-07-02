"""Tab. 2 metrics for COCO-30k generations.

For each row i in coco-30k.csv expect {gen_dir}/{i}.{png|jpg}.

Outputs CLIPScore (ViT-L/14, w=2.5, mean over per-image scores) and CMMD
(generated set vs real COCO val ref set, ViT-L/14 features).
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import clip

sys.path.append(os.path.dirname(__file__))
from core.eval.cmmd import compute_cmmd, _embed, _list_images


_EXTS = ("png", "jpg", "jpeg")


@torch.no_grad()
def clip_score_mean(
    pairs,  # list of (image_path, prompt)
    device: str = "cuda",
    batch_size: int = 32,
    w: float = 2.5,
) -> float:
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    sims = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="CLIPScore"):
        batch = pairs[i:i + batch_size]
        imgs, txts = [], []
        for p, t in batch:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
                txts.append(t)
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        toks = clip.tokenize(txts, truncate=True).to(device)
        f_i = model.encode_image(x).float()
        f_t = model.encode_text(toks).float()
        f_i = f_i / f_i.norm(dim=-1, keepdim=True)
        f_t = f_t / f_t.norm(dim=-1, keepdim=True)
        sim = (f_i * f_t).sum(dim=-1).clamp(min=0.0).cpu().numpy()
        sims.extend(sim.tolist())
    return float(w * np.mean(sims))


def _find_image(d, idx):
    for ext in _EXTS:
        p = os.path.join(d, f"{idx}.{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--csv", default="/data/home/acw685/CA_diffusion_debiasing-main/coco-30k.csv")
    ap.add_argument("--ref_dir", default=None, help="real COCO val image dir for CMMD")
    ap.add_argument("--metrics", default="clip,cmmd", help="comma-separated subset of {clip,cmmd}")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    pairs = []
    for i in range(len(df)):
        p = _find_image(args.gen_dir, i)
        if p is not None:
            pairs.append((p, str(df.iloc[i]["prompt"])))
    print(f"Found {len(pairs)} / {len(df)} generated images.")
    if not pairs:
        sys.exit(1)

    metrics = set(args.metrics.split(","))

    if "clip" in metrics:
        s = clip_score_mean(pairs, batch_size=args.batch_size)
        print(f"CLIP Score (ViT-L/14, w=2.5, mean over {len(pairs)} pairs): {s:.4f}")

    if "cmmd" in metrics:
        if not args.ref_dir:
            print("CMMD requested but --ref_dir not provided; skipping.")
        else:
            gen_paths = [p for p, _ in pairs]
            ref_paths = _list_images(args.ref_dir)
            print(f"CMMD: {len(gen_paths)} gen vs {len(ref_paths)} ref images")
            gen_f = _embed(gen_paths, batch_size=args.batch_size)
            ref_f = _embed(ref_paths, batch_size=args.batch_size)
            cmmd = compute_cmmd("", "", gen_features=gen_f, ref_features=ref_f)
            print(f"CMMD (x1000): {cmmd:.4f}")


if __name__ == "__main__":
    main()
