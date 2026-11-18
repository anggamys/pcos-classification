"""Evaluate preprocessing quality for H1 (PCD side, no training needed).

For a deterministic sample of N images per class:
- PSNR/SSIM of raw vs BayesShrink-denoised (E2 pipeline)
- PSNR/SSIM of raw vs denoise+CLAHE (E3 pipeline)
- ROI stats (num regions, coverage, mean/max area) for otsu & adaptive
  segmentation on the denoise+CLAHE image (pipeline used in E4-E6)

Outputs (default experiments/quality/):
- quality.csv      : one row per image
- summary.txt      : mean/std per metric, per class (human readable)
- summary.json     : same summary, machine readable
- figures/         : sample overlays (raw | denoised | enhanced | mask overlay)

Usage:
    python experiments/evaluate_quality.py --data-dir datasets --samples-per-class 100
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.quality import enhance, psnr, ssim, summarize
from src.preprocessing.segmentation import (
    SEGMENT_WORK_SIZE,
    analyze_regions,
    overlay_mask,
    segment,
    segment_at_scale,
)
from src.preprocessing.wavelet import wavelet_denoise

CLASSES = ["infected", "noninfected"]
SEG_METHODS = ["otsu", "adaptive"]


def describe_data_dir(data_dir: str) -> str:
    """Ringkas isi direktori dataset untuk pesan error yang diagnostik.

    Args:
        data_dir (str): Direktori dataset (relatif terhadap root proyek atau absolut).

    Returns:
        str: Beberapa baris berisi path, cacah *.jpg, ekstensi yang ditemukan,
            dan contoh nama file per kelas.
    """
    root = PROJECT_ROOT / data_dir
    lines = [f"Direktori dataset: {root}"]

    if not root.is_dir():
        lines.append("  direktori tidak ditemukan")
        return "\n".join(lines)

    for cls in CLASSES:
        cls_dir = root / cls

        if not cls_dir.is_dir():
            lines.append(f"  {cls}/: direktori tidak ditemukan")
            continue

        files = sorted(p for p in cls_dir.iterdir() if p.is_file())
        jpg_count = sum(1 for p in files if p.suffix == ".jpg")
        extensions = sorted({p.suffix.lower() or "<tanpa ekstensi>" for p in files})
        sample = ", ".join(p.name for p in files[:3])

        lines.append(
            f"  {cls}/: {jpg_count} *.jpg, {len(files)} file total"
            + (f", ekstensi: {', '.join(extensions)}" if extensions else "")
            + (f", contoh: {sample}" if sample else "")
        )

    return "\n".join(lines)


def sample_images(data_dir: str, samples_per_class: int, seed: int = 42) -> dict:
    """Ambil sampel citra acak deterministik per kelas.

    Salinan byte-identik (md5, keep-first) dibuang sebelum sampling agar
    statistik PSNR/SSIM tidak didominasi salinan duplikat, konsisten
    dengan dedup di PCOSDataset.

    Args:
        data_dir (str): Direktori dataset (berisi infected/noninfected).
        samples_per_class (int): Jumlah sampel per kelas (minimal 1).
        seed (int): Seed acak agar sampel reprodusibel.

    Returns:
        dict: Daftar path citra per kelas ("infected", "noninfected").

    Raises:
        ValueError: Jika samples_per_class < 1, atau tidak ada citra .jpg
            pada salah satu kelas (path salah, ekstensi lain, atau folder kosong).
    """
    if samples_per_class < 1:
        raise ValueError(
            f"samples_per_class harus >= 1, diterima {samples_per_class}"
        )

    rng = np.random.RandomState(seed)
    picked = {}

    for cls in CLASSES:
        files = sorted((PROJECT_ROOT / data_dir / cls).glob("*.jpg"))

        seen = set()
        unique = []

        for file_path in files:
            digest = hashlib.md5(file_path.read_bytes()).hexdigest()

            if digest not in seen:
                seen.add(digest)
                unique.append(file_path)

        if len(unique) <= samples_per_class:
            picked[cls] = unique
        else:
            idx = rng.choice(len(unique), samples_per_class, replace=False)
            picked[cls] = [unique[i] for i in sorted(idx)]

    empty_classes = [cls for cls in CLASSES if not picked[cls]]

    if empty_classes:
        raise ValueError(
            "Tidak ada citra .jpg yang bisa disampling untuk kelas "
            f"{', '.join(empty_classes)}.\n"
            f"{describe_data_dir(data_dir)}\n"
            "Periksa: (1) --data-dir mengarah ke folder yang berisi "
            "infected/ dan noninfected/ (mis. datasets/PCOS, bukan datasets); "
            "(2) nama kelas huruf kecil; (3) ekstensi .jpg huruf kecil "
            "(glob *.jpg case-sensitive, .JPG/.png tidak terbaca)."
        )

    return picked


def process_one(img_path: Path) -> tuple:
    """Proses satu citra: metrik terpisah, gabungan, dan statistik ROI.

    Metrik denoise dan CLAHE dihitung terpisah terhadap citra asli. Citra
    gabungan mengikuti urutan training (CLAHE lalu denoise), dan segmentasi
    dihitung pada skala kerja tetap agar sebanding antar resolusi.

    Args:
        img_path (Path): Path file citra.

    Returns:
        tuple: (baris metrik quality.csv, citra asli, hasil denoise,
            hasil CLAHE, gabungan denoise+CLAHE) untuk figure.
    """
    raw = Image.open(img_path).convert("RGB")
    denoised = wavelet_denoise(raw)
    enhanced = enhance(raw, "clahe")
    combined = wavelet_denoise(enhanced)

    row: dict[str, Any] = {"file": img_path.name}
    row["psnr_denoise"] = psnr(raw, denoised)
    row["ssim_denoise"] = ssim(raw, denoised)
    row["psnr_enhanced"] = psnr(raw, enhanced)
    row["ssim_enhanced"] = ssim(raw, enhanced)

    for method in SEG_METHODS:
        mask = segment_at_scale(combined, method=method, size=SEGMENT_WORK_SIZE)
        stats = analyze_regions(mask)

        row[f"{method}_regions"] = stats["num_regions"]
        row[f"{method}_coverage"] = round(stats["coverage"], 4)
        row[f"{method}_mean_area"] = round(stats["mean_area"], 1)
        row[f"{method}_max_area"] = round(stats["max_area"], 1)

    return row, raw, denoised, enhanced, combined


def save_figure(
    raw: Image.Image,
    denoised: Image.Image,
    enhanced: Image.Image,
    combined: Image.Image,
    save_path: Path,
    n_overlays: int = 1,
) -> None:
    """Simpan figure 4 panel (asli, denoise, CLAHE, overlay Otsu).

    Args:
        raw (PIL.Image.Image): Citra asli.
        denoised (PIL.Image.Image): Citra hasil denoise saja.
        enhanced (PIL.Image.Image): Citra hasil CLAHE saja.
        combined (PIL.Image.Image): Citra CLAHE lalu denoise (urutan training).
        save_path (Path): Path PNG tujuan.
        n_overlays (int): Parameter cadangan jumlah overlay (saat ini 1).
    """
    mask = segment(combined, method="otsu")
    overlay = overlay_mask(combined, mask)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, img, title in zip(
        axes,
        [raw, denoised, enhanced, overlay],
        ["Raw", "Denoised", "CLAHE", "Denoise+CLAHE overlay"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Orkestrasi evaluasi kualitas: sampling, proses, tulis CSV/JSON/figure."""
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
    picked = sample_images(args.data_dir, args.samples_per_class, args.seed)
    print("Sampel: " + ", ".join(f"{cls}={len(picked[cls])}" for cls in CLASSES))

    # Dibuat setelah sampling valid agar run yang gagal karena data-dir salah
    # tidak meninggalkan direktori keluaran kosong.
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for cls in CLASSES:
        for i, path in enumerate(picked[cls]):
            row, raw, denoised, enhanced, combined = process_one(path)
            row["class"] = cls
            rows.append(row)

            if i < args.figures:
                save_figure(
                    raw,
                    denoised,
                    enhanced,
                    combined,
                    fig_dir / f"{cls}_{i:02d}.png",
                )

            print(f"{cls} {i + 1}/{len(picked[cls])}: {path.name}")

    if not rows:
        raise ValueError(
            "Tidak ada baris kualitas yang dihasilkan; periksa data-dir "
            "dan samples-per-class."
        )

    # Per-image CSV
    csv_path = out_root / "quality.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary per class + overall (txt for humans, json for documentation)
    lines = []
    structured = {"samples_per_class": args.samples_per_class, "classes": {}}

    for cls in CLASSES + ["all"]:
        subset = [r for r in rows if cls == "all" or r["class"] == cls]
        lines.append(f"{cls} (n={len(subset)})")
        structured["classes"][cls] = {"n": len(subset), "metrics": {}}

        for key in rows[0]:
            if key in ("file", "class"):
                continue

            s = summarize([r[key] for r in subset])
            lines.append(
                f"{key}: mean={s['mean']:.4f} std={s['std']:.4f} "
                f"min={s['min']:.4f} max={s['max']:.4f}"
            )

            structured["classes"][cls]["metrics"][key] = s

    summary = "\n".join(lines)
    (out_root / "summary.txt").write_text(summary)
    (out_root / "summary.json").write_text(json.dumps(structured, indent=2))

    print(summary)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
