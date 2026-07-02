"""Analyse FairSteer gate behaviour from dumped projection-score pickles.

Reads pickles produced by `controller.py:debiasing(..., save_vectors=True, ...)`
and computes:

  - Per-(model x attribute) ROC/AUROC + PR/AUPRC at the chosen `l^gate, t=0`.
  - Per-(model x attribute) AUROC heatmap across (denoising_step, CA layer, place).
  - Threshold-sweep FPR / FNR curves at the chosen `l^gate`.
  - Operating-point confusion matrix at the empirical midpoint threshold
    (i.e. `thr['stats']` from the per-direction threshold pickle).

The gating layer per backbone matches `controller.py`:
  sd15 / sd21:  diffusion_step=0, place_in_unet='down', block_index=4
  sdxl:         diffusion_step=0, place_in_unet='down', block_index=17
  sana15:       diffusion_step=0, block_index=5  (place_in_unet ignored for SANA)

Inputs (all under /gpfs/scratch/acw685/CA_diffusion_debiasing-main/):

  tmp/vectors/{model}/{direction}/{neutral_concept}/projection_scores_*.pkl
  tmp/vectors/{model}/{direction}/pos_{neutral_concept}/projection_scores_*.pkl

Outputs:
  <output_dir>/scores.parquet             aggregated DataFrame
  <output_dir>/roc_<direction>.png        ROC curve per direction at l^gate
  <output_dir>/pr_<direction>.png         PR curve per direction at l^gate
  <output_dir>/auroc_heatmap_<direction>.png  AUROC across (step, block) at place=down
  <output_dir>/threshold_sweep_<direction>.png  FPR/FNR vs threshold at l^gate
  <output_dir>/summary.json               numbers (AUROC, AUPRC, FPR, FNR, midpoint thr)
"""
import argparse
import json
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


GATE_BLOCKS = {
    "sd15": (0, "down", 4),
    "sd21": (0, "down", 4),
    "sdxl": (0, "down", 17),
    "sana15": (0, "sana", 5),  # SANA uses 'sana' as place_in_unet
}


def _read_pickle_fast(path: str) -> tuple:
    """Return (step, place, block, counter, dp_max) for one pickle.

    counter is preserved so that, across direction dumps from runs with the
    same seed and prompt, gate-cell pickles can be matched per-image."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    scores = d["scores"]
    dp = float(scores.max().item())
    return (int(d["diffusion_step"]), str(d["place_in_unet"]),
            int(d["block_index"]), int(d.get("counter", -1)), dp)


def _process_dir(directory: str, label: int, prof: str, direction: str,
                 prompt_kind: str) -> list:
    files = glob(os.path.join(directory, "projection_scores_*.pkl"))
    rows = []
    for fpath in files:
        try:
            step, place, block, counter, dp = _read_pickle_fast(fpath)
        except Exception as e:  # noqa
            continue
        rows.append((direction, prof, prompt_kind, label, step, place, block,
                     counter, dp))
    return rows


def _detect_concepts(direction_root: str) -> list[str]:
    """Find profession dirs in {root}/{direction}/. Skip pos_* (those are
    looked up from concept names directly)."""
    if not os.path.isdir(direction_root):
        return []
    out = []
    for name in os.listdir(direction_root):
        if name.startswith("pos_"):
            continue
        if os.path.isdir(os.path.join(direction_root, name)):
            out.append(name)
    return sorted(out)


def collect_scores(model: str, direction: str, root: str,
                   workers: int = 8) -> pd.DataFrame:
    """For a (model, direction), walk neutral and pos_ dirs across all
    profession concepts and build a long-form DataFrame."""
    direction_root = os.path.join(root, "tmp/vectors", model, direction)
    concepts = _detect_concepts(direction_root)
    if not concepts:
        print(f"[warn] no concepts under {direction_root}")
        return pd.DataFrame()

    print(f"[info] {model}/{direction}: {len(concepts)} concepts")

    tasks = []
    for prof in concepts:
        neg_dir = os.path.join(direction_root, prof)
        pos_dir = os.path.join(direction_root, "pos_" + prof)
        if os.path.isdir(neg_dir):
            tasks.append((neg_dir, 0, prof, direction, "neutral"))
        if os.path.isdir(pos_dir):
            tasks.append((pos_dir, 1, prof, direction, "attr_specific"))

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_dir, *t) for t in tasks]
        for fut in as_completed(futures):
            rows.extend(fut.result())

    df = pd.DataFrame(rows, columns=[
        "direction", "profession", "prompt_kind", "label",
        "step", "place", "block", "counter", "dp",
    ])
    return df


def _load_midpoint_threshold(model: str, attribute: str, subgroup: str,
                             gate_key: tuple) -> float | None:
    """Load thr['stats'] for the gating (step, place, block).

    Use only the basename of `subgroup` because race/gender/eyeglasses dirs
    are flat (e.g. 'White', 'male_female', 'eyeglasses')."""
    sub = subgroup.split("/")[-1]
    candidates = [
        f"/data/home/acw685/CA_diffusion_debiasing-main/thresholds/wtf_sdxl/{model}/{sub}/gender_{sub}_{model}.pkl",
        f"/data/home/acw685/CA_diffusion_debiasing-main/thresholds_race/{model}/{sub}/race_{sub}_{model}.pkl",
        f"/data/home/acw685/CA_diffusion_debiasing-main/thresholds/eyeglasses/{model}/{sub}/concepts_{sub}_{model}.pkl",
        f"/data/home/acw685/CA_diffusion_debiasing-main/thresholds_age/{model}/{sub}/age_{sub}_{model}.pkl",
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            with open(cand, "rb") as f:
                thr = pickle.load(f)
            if gate_key in thr:
                return float(thr[gate_key]["stats"])
    return None


def _gate_metrics_at_layer(df_layer: pd.DataFrame, midpoint_thr: float | None
                           ) -> dict:
    y = df_layer["label"].to_numpy()
    s = df_layer["dp"].to_numpy()

    if y.sum() == 0 or y.sum() == len(y):
        return {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}

    auroc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))

    fpr, tpr, thr = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    out = {
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "roc_thr": thr.tolist(),
        "pr_prec": prec.tolist(),
        "pr_rec": rec.tolist(),
    }
    if midpoint_thr is not None:
        pred = (s > midpoint_thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out.update({
            "midpoint_thr": float(midpoint_thr),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "fpr_at_midpoint": float(fp / max(fp + tn, 1)),
            "fnr_at_midpoint": float(fn / max(fn + tp, 1)),
            "tpr_at_midpoint": float(tp / max(tp + fn, 1)),
        })
    return out


def _ensemble_metrics(all_dfs: list[pd.DataFrame], directions: list[str],
                      thrs: dict[str, float], gate_key: tuple) -> dict:
    """Compute the realised inference-time gate AUROC.

    The actual gate at inference fires (skips debiasing) iff dp_d > thr_d for
    SOME direction d. Equivalently the score is s = max_d (dp_d - thr_d), and
    the gate fires iff s > 0.

    Per-image alignment: across direction dumps from runs with matching seed
    and prompt, the (counter, step, place, block) tuple at the gate cell maps
    one-to-one. We match on counter to recover per-image rows.

    If only a subset of directions has dumped data on disk, the ensemble is
    computed over that subset (and ``directions_used`` is recorded)."""
    if not all_dfs:
        return {}
    big = pd.concat(all_dfs, ignore_index=True)
    gate_step, gate_place, gate_block = gate_key
    sub = big[(big["step"] == gate_step)
              & (big["place"] == gate_place)
              & (big["block"] == gate_block)]
    if sub.empty:
        return {}

    # Pick the subset of directions with both data and a loaded threshold.
    available_dirs = sorted({d for d in sub["direction"].unique()
                             if d in thrs and thrs.get(d) is not None})
    if not available_dirs:
        return {"reason": "no directions with data and threshold"}

    # Build per-(prof, prompt_kind, counter) row vector of dp values keyed by
    # direction. Only keep rows where every available direction has a dp value.
    rows = []
    skipped_missing = 0
    for (prof, prompt_kind, counter), g in sub.groupby(
            ["profession", "prompt_kind", "counter"]):
        labels = g["label"].unique()
        if len(labels) != 1:
            continue  # inconsistent labels for the same image — skip
        label = int(labels[0])
        dps = {row["direction"]: float(row["dp"]) for _, row in g.iterrows()}
        if not all(d in dps for d in available_dirs):
            skipped_missing += 1
            continue
        # Realised gate score: max over directions of (dp_d - thr_d).
        s_max_shift = max(dps[d] - thrs[d] for d in available_dirs)
        # Also the un-shifted max for diagnostic.
        s_max_raw = max(dps[d] for d in available_dirs)
        rows.append({
            "profession": prof, "prompt_kind": prompt_kind,
            "counter": int(counter), "label": label,
            "ensemble_score_shifted": s_max_shift,
            "ensemble_score_raw": s_max_raw,
            **{f"dp_{d.replace('/', '_')}": dps[d] for d in available_dirs},
        })
    df_ens = pd.DataFrame(rows)
    if df_ens.empty or df_ens["label"].nunique() < 2:
        return {"n": int(len(df_ens)),
                "directions_used": available_dirs,
                "skipped_missing_directions": skipped_missing}

    y = df_ens["label"].to_numpy()
    out = {
        "n": int(len(df_ens)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "directions_used": available_dirs,
        "skipped_missing_directions": skipped_missing,
        "auroc_shifted": float(roc_auc_score(y, df_ens["ensemble_score_shifted"])),
        "auprc_shifted": float(average_precision_score(y, df_ens["ensemble_score_shifted"])),
        "auroc_raw": float(roc_auc_score(y, df_ens["ensemble_score_raw"])),
        "auprc_raw": float(average_precision_score(y, df_ens["ensemble_score_raw"])),
    }
    # Operating point at threshold 0 on the shifted score (== inference rule)
    pred = (df_ens["ensemble_score_shifted"] > 0).astype(int).to_numpy()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update({
        "tp_at_zero": int(tp), "fp_at_zero": int(fp),
        "tn_at_zero": int(tn), "fn_at_zero": int(fn),
        "fpr_at_zero": float(fp / max(fp + tn, 1)),
        "fnr_at_zero": float(fn / max(fn + tp, 1)),
        "tpr_at_zero": float(tp / max(tp + fn, 1)),
    })
    return out, df_ens


def _auroc_heatmap(df: pd.DataFrame, place: str) -> pd.DataFrame:
    sub = df[df["place"] == place]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (step, block), g in sub.groupby(["step", "block"]):
        y = g["label"].to_numpy()
        s = g["dp"].to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        rows.append((step, block, float(roc_auc_score(y, s))))
    return pd.DataFrame(rows, columns=["step", "block", "auroc"])


def _plot_roc(metrics: dict, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(metrics["roc_fpr"], metrics["roc_tpr"],
            label=f"AUROC = {metrics['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    if "fpr_at_midpoint" in metrics:
        ax.scatter([metrics["fpr_at_midpoint"]], [metrics["tpr_at_midpoint"]],
                   color="red", zorder=5, label="midpoint thr op-point")
    ax.set_xlabel("False positive rate (neutral classified as attr-specific)")
    ax.set_ylabel("True positive rate (attr-specific classified correctly)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_pr(metrics: dict, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(metrics["pr_rec"], metrics["pr_prec"],
            label=f"AUPRC = {metrics['auprc']:.3f}")
    ax.set_xlabel("Recall (attr-specific correctly classified)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_threshold_sweep(metrics: dict, title: str, out_path: str):
    fpr = np.array(metrics["roc_fpr"])
    tpr = np.array(metrics["roc_tpr"])
    thr = np.array(metrics["roc_thr"])
    fnr = 1 - tpr
    # roc_curve returns thresholds in decreasing order; reverse for plotting
    order = np.argsort(thr)
    thr = thr[order]
    fpr = fpr[order]
    fnr = fnr[order]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(thr, fpr, label="FPR (false skip)", color="C0")
    ax.plot(thr, fnr, label="FNR (false debias)", color="C3")
    if "midpoint_thr" in metrics:
        ax.axvline(metrics["midpoint_thr"], color="k", ls="--", lw=0.7,
                   label=f"midpoint thr = {metrics['midpoint_thr']:.3f}")
    ax.set_xlabel("Gate threshold (dot-product score)")
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_dp_distribution(df_layer: pd.DataFrame, midpoint_thr: float | None,
                          title: str, out_path: str):
    pos = df_layer[df_layer["label"] == 1]["dp"].to_numpy()
    neg = df_layer[df_layer["label"] == 0]["dp"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bins = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 50)
    ax.hist(neg, bins=bins, alpha=0.5, label=f"neutral (n={len(neg)})", color="C0")
    ax.hist(pos, bins=bins, alpha=0.5, label=f"attr-specific (n={len(pos)})", color="C3")
    if midpoint_thr is not None:
        ax.axvline(midpoint_thr, color="k", ls="--", lw=0.7,
                   label=f"midpoint thr = {midpoint_thr:.3f}")
    ax.set_xlabel("dp = max_token <ca, s>")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_heatmap(heat: pd.DataFrame, title: str, out_path: str,
                  highlight_block: int | None = None):
    if heat.empty:
        return
    pivot = heat.pivot(index="step", columns="block", values="auroc")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   vmin=0.5, vmax=1.0, cmap="viridis",
                   extent=[pivot.columns.min() - 0.5, pivot.columns.max() + 0.5,
                           pivot.index.min() - 0.5, pivot.index.max() + 0.5])
    ax.set_xlabel("CA block index")
    ax.set_ylabel("denoising step")
    ax.set_title(title)
    if highlight_block is not None:
        ax.axvline(highlight_block, color="red", lw=0.8, alpha=0.7,
                   label=f"chosen l^gate = {highlight_block}")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", nargs="+", required=True,
                        help="Direction subdirs under tmp/vectors/{model}/, e.g. male_female female_male")
    parser.add_argument("--root", default="/gpfs/scratch/acw685/CA_diffusion_debiasing-main")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--attribute", default="gender",
                        choices=["gender", "race", "eyeglasses", "age", "custom"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gate_key = GATE_BLOCKS[args.model]

    summary = {"model": args.model, "attribute": args.attribute,
               "gate_key": list(gate_key), "per_direction": {}}

    all_dfs = []
    direction_thrs: dict[str, float | None] = {}
    for direction in args.directions:
        df = collect_scores(args.model, direction, args.root, workers=args.workers)
        if df.empty:
            print(f"[warn] no data for {args.model}/{direction}")
            continue
        all_dfs.append(df)

        # Save raw aggregated scores
        df.to_parquet(os.path.join(args.output_dir, f"scores_{direction.replace('/', '_')}.parquet"))

        gate_step, gate_place, gate_block = gate_key
        df_layer = df[(df["step"] == gate_step)
                      & (df["place"] == gate_place)
                      & (df["block"] == gate_block)]
        if df_layer.empty:
            print(f"[warn] no data at gate layer for {args.model}/{direction}")
            continue

        midpoint_thr = _load_midpoint_threshold(args.model, args.attribute,
                                                 direction, gate_key)
        direction_thrs[direction] = midpoint_thr
        metrics = _gate_metrics_at_layer(df_layer, midpoint_thr)

        # Plots
        title_base = f"{args.model} / {direction}  (l^gate={gate_block}, t=0)"
        _plot_roc(metrics, "ROC " + title_base,
                  os.path.join(args.output_dir, f"roc_{direction.replace('/', '_')}.png"))
        _plot_pr(metrics, "PR " + title_base,
                 os.path.join(args.output_dir, f"pr_{direction.replace('/', '_')}.png"))
        _plot_threshold_sweep(metrics, "Threshold sweep " + title_base,
                              os.path.join(args.output_dir,
                                           f"threshold_sweep_{direction.replace('/', '_')}.png"))
        _plot_dp_distribution(df_layer, midpoint_thr,
                              "dp distribution " + title_base,
                              os.path.join(args.output_dir,
                                           f"dp_dist_{direction.replace('/', '_')}.png"))

        heat_place = gate_place if args.model != "sana15" else "sana"
        heat = _auroc_heatmap(df, heat_place)
        if not heat.empty:
            heat.to_csv(os.path.join(args.output_dir,
                                     f"auroc_heatmap_{direction.replace('/', '_')}.csv"),
                        index=False)
            _plot_heatmap(heat, f"AUROC across (step, block) "
                                f"@place={heat_place} | {args.model}/{direction}",
                          os.path.join(args.output_dir,
                                       f"auroc_heatmap_{direction.replace('/', '_')}.png"),
                          highlight_block=gate_block)

        # JSON-friendly metric summary (drop the long curve arrays)
        slim = {k: v for k, v in metrics.items()
                if k not in {"roc_fpr", "roc_tpr", "roc_thr", "pr_prec", "pr_rec"}}
        summary["per_direction"][direction] = slim
        print(f"[done] {direction}: AUROC={metrics.get('auroc', float('nan')):.3f}"
              f" AUPRC={metrics.get('auprc', float('nan')):.3f}"
              f" midpoint_thr={metrics.get('midpoint_thr', float('nan'))}")

    # Combined-pooled: pool all (direction, image) rows. NOT the inference
    # ensemble; just an aggregated single-direction discriminability number.
    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        gate_step, gate_place, gate_block = gate_key
        df_layer = big[(big["step"] == gate_step)
                       & (big["place"] == gate_place)
                       & (big["block"] == gate_block)]
        if not df_layer.empty:
            metrics = _gate_metrics_at_layer(df_layer, None)
            _plot_roc(metrics, f"ROC pooled  ({args.model}/{args.attribute})",
                      os.path.join(args.output_dir, "roc_pooled.png"))
            slim = {k: v for k, v in metrics.items()
                    if k not in {"roc_fpr", "roc_tpr", "roc_thr",
                                 "pr_prec", "pr_rec"}}
            summary["pooled"] = slim

    # Ensemble: realised inference-time gate score s = max_d (dp_d - thr_d).
    # Aligns directions per image via `counter` (deterministic seed across
    # direction dumps).
    ens = _ensemble_metrics(all_dfs, args.directions, direction_thrs, gate_key)
    if isinstance(ens, tuple):
        ens_metrics, df_ens = ens
        df_ens.to_parquet(os.path.join(args.output_dir, "ensemble_scores.parquet"))
        summary["ensemble"] = ens_metrics
        print(f"[ensemble] AUROC_shifted={ens_metrics.get('auroc_shifted', float('nan')):.3f}  "
              f"AUROC_raw={ens_metrics.get('auroc_raw', float('nan')):.3f}  "
              f"FPR@thr=0={ens_metrics.get('fpr_at_zero', float('nan')):.3f}  "
              f"FNR@thr=0={ens_metrics.get('fnr_at_zero', float('nan')):.3f}  "
              f"n_pos={ens_metrics.get('n_pos','?')} n_neg={ens_metrics.get('n_neg','?')}")
    elif ens:
        summary["ensemble"] = ens

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] summary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
