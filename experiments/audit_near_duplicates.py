"""Audit near-duplicate leakage under the project's image-level split.

Reproduces PCOSDataset sample ordering, md5 deduplication, and
torch.randperm(seed=42), then compares a 64-bit perceptual hash of every
test image against every training image.

Usage:
    .venv/bin/python experiments/audit_near_duplicates.py \
        --data-dir /path/to/PCOS

Output defaults to experiments/quality/near_duplicate_audit.json.
"""

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fftpack import dct

SIZE = 32
HASH_BITS = 64
RANDOM_PAIRS = 200_000
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def list_unique_samples(root: Path) -> tuple[list[str], np.ndarray, dict]:
    """Recreate Dataset ordering and md5 keep-first deduplication.

    Args:
        root (Path): Dataset root with `noninfected` and `infected` folders.

    Returns:
        tuple: (paths, labels, dedup_stats).
    """
    samples = []
    for label, subdir in enumerate(["noninfected", "infected"]):
        for path in sorted((root / subdir).glob("*.jpg")):
            samples.append((str(path), label))

    seen = set()
    unique = []
    for path, label in samples:
        digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((path, label))

    paths = [path for path, _ in unique]
    labels = np.array([label for _, label in unique], dtype=np.int8)
    stats = {
        "raw": len(samples),
        "unique": len(unique),
        "removed": len(samples) - len(unique),
    }

    return paths, labels, stats


def phash(path: str) -> int:
    """Return a 64-bit DCT perceptual hash for one image."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        array = np.asarray(gray, dtype=np.float64)

    coeffs = dct(dct(array, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:8, :8].flatten()
    median = np.median(low[1:])

    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)

    return value


def gray128(path: str) -> np.ndarray:
    """Load a 128x128 grayscale matrix for pixel-level comparison."""
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize((128, 128), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )


def popcount(values: np.ndarray) -> np.ndarray:
    """Hamming weight of a uint64 array."""
    return np.bitwise_count(values).astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument(
        "--out",
        default="experiments/quality/near_duplicate_audit.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.data_dir)
    paths, labels, dedup_stats = list_unique_samples(root)
    total = len(paths)

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    order = torch.randperm(total, generator=generator).numpy()
    train_n = int(TRAIN_RATIO * total)
    val_n = int(VAL_RATIO * total)
    train = order[:train_n]
    val = order[train_n : train_n + val_n]
    test = order[train_n + val_n :]

    with Pool(args.workers) as pool:
        hashes = np.array(pool.map(phash, paths, chunksize=64), dtype=np.uint64)
        train_gray = pool.map(gray128, [paths[i] for i in train], chunksize=16)

    test_hash = hashes[test]
    train_hash = hashes[train]
    distances = popcount(test_hash[:, None] ^ train_hash[None, :])
    nearest = distances.min(axis=1)
    nearest_index = distances.argmin(axis=1)
    nearest_class = labels[train][nearest_index]

    rng = np.random.default_rng(0)
    random_test = rng.integers(0, len(test), RANDOM_PAIRS)
    random_train = rng.integers(0, len(train), RANDOM_PAIRS)
    random_distances = popcount(test_hash[random_test] ^ train_hash[random_train])

    thresholds = [0, 2, 4, 6, 8, 10, 12, 16, 20]
    distribution = {}
    for threshold in thresholds:
        count = int((nearest <= threshold).sum())
        distribution[str(threshold)] = {
            "count": count,
            "fraction": count / len(test),
            "random_fraction": float((random_distances <= threshold).mean()),
        }

    # Pixel-level check for the closest hash matches.
    pixel_checks = {}
    for max_distance in [0, 2]:
        indices = np.where(nearest <= max_distance)[0]
        if len(indices) > 200:
            indices = rng.choice(indices, 200, replace=False)

        absolute_errors = []
        correlations = []
        for index in indices:
            test_gray = gray128(paths[test[index]])
            train_image = train_gray[nearest_index[index]]
            absolute_errors.append(float(np.mean(np.abs(test_gray - train_image))))
            correlations.append(
                float(np.corrcoef(test_gray.ravel(), train_image.ravel())[0, 1])
            )

        absolute_errors = np.array(absolute_errors)
        correlations = np.array(correlations)
        pixel_checks[str(max_distance)] = {
            "n": len(indices),
            "mae_median": float(np.median(absolute_errors)),
            "mae_min": float(absolute_errors.min()),
            "mae_max": float(absolute_errors.max()),
            "mae_below_1": int((absolute_errors < 1).sum()),
            "mae_below_3": int((absolute_errors < 3).sum()),
            "corr_median": float(np.median(correlations)),
            "corr_above_0_99": int((correlations > 0.99).sum()),
        }

    same_class = nearest_class == labels[test]
    per_class = {}
    for class_id, name in [(0, "healthy"), (1, "pcos")]:
        mask = labels[test] == class_id
        per_class[name] = {
            "n": int(mask.sum()),
            "median_nearest_distance": float(np.median(nearest[mask])),
            "nearest_le_6": int((nearest[mask] <= 6).sum()),
            "nearest_gt_10": int((nearest[mask] > 10).sum()),
        }

    clean = nearest > 10
    result = {
        "method": {
            "hash": f"DCT perceptual hash {HASH_BITS}-bit ({SIZE}x{SIZE} grayscale)",
            "split_seed": SPLIT_SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "random_pairs": RANDOM_PAIRS,
        },
        "dedup": dedup_stats,
        "split": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "test_healthy": int((labels[test] == 0).sum()),
            "test_pcos": int((labels[test] == 1).sum()),
        },
        "distribution": distribution,
        "random_distance_quantiles": {
            "q05": float(np.quantile(random_distances, 0.05)),
            "q50": float(np.quantile(random_distances, 0.50)),
            "q95": float(np.quantile(random_distances, 0.95)),
        },
        "nearest_distance": {
            "median": float(np.median(nearest)),
            "min": int(nearest.min()),
            "max": int(nearest.max()),
        },
        "nearest_class_agreement": {
            "count": int(same_class.sum()),
            "fraction": float(same_class.mean()),
        },
        "per_class": per_class,
        "pixel_checks": pixel_checks,
        "clean_subset": {
            "rule": "nearest distance > 10",
            "n": int(clean.sum()),
            "healthy": int((labels[test][clean] == 0).sum()),
            "pcos": int((labels[test][clean] == 1).sum()),
        },
        "arrays": {
            "test_indices": test.tolist(),
            "test_labels": labels[test].tolist(),
            "nearest_distances": nearest.tolist(),
        },
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print(
        f"unique={total} train={len(train)} val={len(val)} test={len(test)} "
        f"({result['split']['test_healthy']} healthy, "
        f"{result['split']['test_pcos']} PCOS)"
    )
    for threshold in thresholds:
        item = distribution[str(threshold)]
        print(
            f"  <= {threshold:2d}: {item['count']:3d} "
            f"({100 * item['fraction']:.1f}%) "
            f"random={100 * item['random_fraction']:.2f}%"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
