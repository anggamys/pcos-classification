"""Evaluate preprocessing quality for H1 (PCD side, no training needed).

For a deterministic sample of N images per class:
- PSNR/SSIM of raw vs BayesShrink-denoised (E2 pipeline)
- PSNR/SSIM of raw vs denoise+CLAHE (E3 pipeline)
- ROI stats (num regions, coverage, mean/max area) for otsu & adaptive
  segmentation on the denoise+CLAHE image (pipeline used in E4-E6)

Outputs (default experiments/quality/):
- quality.csv      : one row per image
- summary.txt      : mean/std per metric, per class
- figures/         : sample overlays (raw | denoised | enhanced | mask overlay)

Usage:
    python experiments/evaluate_quality.py --data-dir datasets --samples-per-class 100
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.image_quality import enhance, psnr, ssim, summarize
from src.preprocessing import wavelet_denoise
from src.segmentation import analyze_regions, overlay_mask, segment

CLASSES = ["infected", "noninfected"]
SEG_METHODS = ["otsu", "adaptive"]


def sample_images(data_dir, samples_per_class, seed=42):
    rng = np.random.RandomState(seed)
    picked = {}
    for cls in CLASSES:
        files = sorted((PROJECT_ROOT / data_dir / cls).glob("*.jpg"))
        if len(files) <= samples_per_class:
            picked[cls] = files
        else:
            idx = rng.choice(len(files), samples_per_class, replace=False)
            picked[cls] = [files[i] for i in sorted(idx)]
    return picked


def process_one(img_path):
    raw = Image.open(img_path).convert("RGB")
    denoised = wavelet_denoise(raw)
    enhanced = enhance(denoised, "clahe")
    row = {"file": img_path.name}
    row["psnr_denoise"] = psnr(raw, denoised)
    row["ssim_denoise"] = ssim(raw, denoised)
    row["psnr_enhanced"] = psnr(raw, enhanced)
    row["ssim_enhanced"] = ssim(raw, enhanced)
    for method in SEG_METHODS:
        mask = segment(enhanced, method=method)
        stats = analyze_regions(mask)
        row[f"{method}_regions"] = stats["num_regions"]
        row[f"{method}_coverage"] = round(stats["coverage"], 4)
        row[f"{method}_mean_area"] = round(stats["mean_area"], 1)
        row[f"{method}_max_area"] = round(stats["max_area"], 1)
    return row, raw, denoised, enhanced


def save_figure(raw, denoised, enhanced, save_path, n_overlays=1):
    mask = segment(enhanced, method="otsu")
    overlay = overlay_mask(enhanced, mask)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(
        axes,
        [raw, denoised, enhanced, overlay],
        ["Raw", "Denoised", "Denoise+CLAHE", "Otsu overlay"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="H1 quality evaluation (no training)")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--out-dir", default="experiments/quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--figures", type=int, default=3, help="Sample overlay figures saved per class"
    )
    args = parser.parse_args()

    out_root = PROJECT_ROOT / args.out_dir
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    picked = sample_images(args.data_dir, args.samples_per_class, args.seed)
    rows = []
    for cls in CLASSES:
        for i, path in enumerate(picked[cls]):
            row, raw, denoised, enhanced = process_one(path)
            row["class"] = cls
            rows.append(row)
            if i < args.figures:
                save_figure(raw, denoised, enhanced, fig_dir / f"{cls}_{i:02d}.png")
            print(f"[{cls} {i + 1}/{len(picked[cls])}] {path.name}")
    # Per-image CSV
    csv_path = out_root / "quality.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary per class + overall
    lines = []
    for cls in CLASSES + ["all"]:
        subset = [r for r in rows if cls == "all" or r["class"] == cls]
        lines.append(f"== {cls} (n={len(subset)}) ==")
        for key in rows[0]:
            if key in ("file", "class"):
                continue
            s = summarize([r[key] for r in subset])
            lines.append(
                f"  {key}: mean={s['mean']:.4f} std={s['std']:.4f} "
                f"min={s['min']:.4f} max={s['max']:.4f}"
            )
    summary = "\n".join(lines)
    (out_root / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
