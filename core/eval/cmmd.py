"""CMMD (CLIP Maximum Mean Discrepancy) from Jayasumana et al. 2023.

Matches Google's reference implementation
(github.com/google-research/google-research/tree/master/cmmd):

  * CLIP backbone: ViT-L/14 at 336 px
  * Embeddings: L2-normalised
  * Estimator: BIASED MMD^2 (mean over all n*m pairs, including the diagonals
    of the within-set kernels)
  * Gaussian kernel bandwidth: σ = 10
  * Final scale: × 1000
"""

import glob
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import clip


_SIGMA = 10.0    # Gaussian-kernel bandwidth (Jayasumana et al. default)
_SCALE = 1000.0  # x1000 for readability (Google ref impl)
_CLIP_MODEL = "ViT-L/14@336px"  # Google ref uses the 336-px ViT-L/14 variant
_EXTENSIONS = ("png", "jpg", "jpeg")


def _list_images(root: str) -> List[str]:
    files = []
    for ext in _EXTENSIONS:
        files.extend(glob.glob(os.path.join(root, f"**/*.{ext}"), recursive=True))
    files.sort()
    return files


@torch.no_grad()
def _embed(
    paths: List[str],
    device: str = "cuda",
    batch_size: int = 64,
    clip_model: str = _CLIP_MODEL,
) -> np.ndarray:
    # Honour CLIP_CACHE_DIR env var so the ~890 MB ViT-L/14@336px weights
    # download lands on scratch instead of the (often tiny) $HOME quota.
    cache = os.environ.get("CLIP_CACHE_DIR") or os.path.expanduser("~/.cache/clip")
    os.makedirs(cache, exist_ok=True)
    model, preprocess = clip.load(clip_model, device=device, download_root=cache)
    model.eval()
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"CLIP embed ({clip_model})"):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        f = model.encode_image(x).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def _mmd_squared_biased(x: np.ndarray, y: np.ndarray, sigma: float = _SIGMA) -> float:
    """Biased MMD^2 with Gaussian kernel exp(-||a-b||^2 / (2 sigma^2)).

    Matches Google CMMD ref impl: mean over the full n*n / m*m / n*m kernel
    matrices (no diagonal exclusion). Features should be L2-normalised.
    """
    gamma = 1.0 / (2.0 * sigma * sigma)

    def rbf_mean(a: np.ndarray, b: np.ndarray) -> float:
        a_sq = np.sum(a * a, axis=1, keepdims=True)
        b_sq = np.sum(b * b, axis=1)
        d = a_sq + b_sq - 2.0 * (a @ b.T)
        # Numerical hygiene — kernel matrix can produce tiny negatives.
        np.maximum(d, 0.0, out=d)
        return float(np.exp(-gamma * d).mean())

    kxx = rbf_mean(x, x)
    kyy = rbf_mean(y, y)
    kxy = rbf_mean(x, y)
    return kxx + kyy - 2.0 * kxy


def compute_cmmd(
    gen_path: str,
    ref_path: str,
    device: str = "cuda",
    batch_size: int = 64,
    gen_features: Optional[np.ndarray] = None,
    ref_features: Optional[np.ndarray] = None,
    clip_model: str = _CLIP_MODEL,
) -> float:
    if gen_features is None:
        gen_features = _embed(_list_images(gen_path), device, batch_size, clip_model)
    if ref_features is None:
        ref_features = _embed(_list_images(ref_path), device, batch_size, clip_model)
    return _SCALE * _mmd_squared_biased(gen_features, ref_features)
