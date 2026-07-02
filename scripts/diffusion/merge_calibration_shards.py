"""Merge per-shard calibration outputs into one summary per cell.

For each cell directory <root>/{backbone}_{attribute}/shard{0,1,...}/, reads the
per-image CSV, concatenates them, and recomputes the same metrics that
calibrate_classifier produces for a single run. Writes:

  <root>/{backbone}_{attribute}/calibration_<bb>_<attr>.json
  <root>/{backbone}_{attribute}/calibration_<bb>_<attr>.csv         (concatenated)
  <root>/{backbone}_{attribute}/reliability_<bb>_<attr>.csv
  <root>/{backbone}_{attribute}/reliability_<bb>_<attr>.png
  <root>/{backbone}_{attribute}/confusion_<bb>_<attr>.png
"""
import argparse
import json
import os
import sys
from glob import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_classifier import (
    ATTRIBUTE_CONFIG,
    confusion,
    parity_gap,
    plot_confusion,
    plot_reliability,
    reliability_bins,
)


def merge_cell(cell_dir: str):
    cell_name = os.path.basename(os.path.normpath(cell_dir))
    # cell_name = "{bb}_{attr}", e.g. "sdxl_gender"
    bb, attr = cell_name.split("_", 1)
    cfg = ATTRIBUTE_CONFIG[attr]
    classes = cfg["classes"]
    target_uniform = cfg["target_uniform"]

    shard_csvs = sorted(glob(os.path.join(cell_dir, "shard*", f"calibration_{bb}_{attr}.csv")))
    if not shard_csvs:
        print(f"[skip] no shard CSVs in {cell_dir}")
        return

    print(f"[merge] {cell_dir}  ->  {len(shard_csvs)} shards")
    df = pd.concat([pd.read_csv(c) for c in shard_csvs], ignore_index=True)
    df = df.drop_duplicates(subset=["path"])  # belt-and-braces
    df.to_csv(os.path.join(cell_dir, f"calibration_{bb}_{attr}.csv"), index=False)

    df_clf = df[df["oracle_pred"].isin(classes)]

    overall_agreement = float((df_clf["clip_pred"] == df_clf["oracle_pred"]).mean()) \
        if len(df_clf) else float("nan")
    per_mode_agreement = {m: float((g["clip_pred"] == g["oracle_pred"]).mean())
                          for m, g in df_clf.groupby("mode")}
    per_cell_agreement = {f"{m}/{p}": {"agreement": float(
                              (g["clip_pred"] == g["oracle_pred"]).mean()),
                                       "n": int(len(g))}
                          for (m, p), g in df_clf.groupby(["mode", "profession"])}

    cm = confusion(df_clf, classes)
    rel = reliability_bins(df_clf, n_bins=10)
    rel.to_csv(os.path.join(cell_dir, f"reliability_{bb}_{attr}.csv"), index=False)

    delta_clip   = parity_gap(df_clf, "clip_pred",   classes, target_uniform)
    delta_oracle = parity_gap(df_clf, "oracle_pred", classes, target_uniform)

    per_class = {}
    for cls in classes:
        sub = df_clf[df_clf["oracle_pred"] == cls]
        if len(sub):
            per_class[cls] = {"n": int(len(sub)),
                              "clip_recall": float((sub["clip_pred"] == cls).mean())}

    summary = {
        "backbone": bb, "attribute": attr,
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
        "n_shards": len(shard_csvs),
    }
    with open(os.path.join(cell_dir, f"calibration_{bb}_{attr}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not rel.empty:
        plot_reliability(rel,
                         os.path.join(cell_dir, f"reliability_{bb}_{attr}.png"),
                         f"CLIP calibration vs GPT-4O ({bb} / {attr}, merged)")
    if not cm.empty:
        plot_confusion(cm,
                       os.path.join(cell_dir, f"confusion_{bb}_{attr}.png"),
                       f"CLIP vs GPT-4O ({bb} / {attr}, merged)")

    print(f"[done] {bb}/{attr}: n={len(df)} ({len(df_clf)} clf)  "
          f"agreement={overall_agreement:.3f}  "
          f"delta_clip(van)={delta_clip.get('vanilla',{}).get('avg_delta','?')}  "
          f"delta_oracle(van)={delta_oracle.get('vanilla',{}).get('avg_delta','?')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/gpfs/scratch/acw685/CA_diffusion_debiasing-main/calibration")
    args = p.parse_args()

    for cell_dir in sorted(glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(cell_dir):
            continue
        merge_cell(cell_dir)
