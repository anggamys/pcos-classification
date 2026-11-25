"""Sensitivity analysis for the manuscript's near-duplicate audit.

Loads the saved checkpoints, reproduces the test split from the audit JSON,
and reports metrics on the full test set and on images whose nearest training
neighbour has a perceptual-hash distance above a threshold. Also computes
paired bootstrap confidence intervals and exact McNemar tests.

Usage:
    .venv/bin/python experiments/sensitivity_analysis.py \
        --data-dir /path/to/PCOS

Requires experiments/quality/near_duplicate_audit.json (run
audit_near_duplicates.py first).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from exp_config import load_config

from src.dataset import PCOSDataset, get_transforms
from src.models.densenet121 import DenseNet121Attention

PAIRS = [("E2", "E1"), ("E4", "E3"), ("E5", "E4"), ("E6", "E4")]


@torch.inference_mode()
def predict(model, dataset, indices, batch_size, device):
    """Run inference on selected dataset indices and return probabilities."""
    probabilities = []
    labels = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = []
        batch_labels = []
        for index in batch_indices:
            image, label = dataset[int(index)]
            images.append(image)
            batch_labels.append(label)

        batch = torch.stack(images).to(device)
        logits = model(batch)
        probabilities.extend(torch.sigmoid(logits).flatten().cpu().tolist())
        labels.extend(batch_labels)

    return np.array(probabilities), np.array(labels)


def metric_row(labels, probabilities):
    """Compute accuracy, F1, AUC, and confusion counts."""
    predictions = (probabilities > 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, probabilities))

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())

    return {
        "n": len(labels),
        "healthy": int((labels == 0).sum()),
        "pcos": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "auc": auc,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def bootstrap(labels, probabilities, repetitions, seed=42):
    """Percentile bootstrap for metrics and paired differences."""
    rng = np.random.default_rng(seed)
    model_ids = list(probabilities)
    samples = {
        model_id: {"accuracy": [], "f1": [], "auc": []} for model_id in model_ids
    }

    differences = {f"{a}-{b}": {"accuracy": [], "f1": [], "auc": []} for a, b in PAIRS}
    n = len(labels)

    for _ in range(repetitions):
        index = rng.integers(0, n, n)
        resampled_labels = labels[index]
        if len(np.unique(resampled_labels)) < 2:
            continue

        values = {}
        for model_id in model_ids:
            probability = probabilities[model_id][index]
            prediction = (probability > 0.5).astype(int)
            values[model_id] = {
                "accuracy": accuracy_score(resampled_labels, prediction),
                "f1": f1_score(resampled_labels, prediction),
                "auc": roc_auc_score(resampled_labels, probability),
            }

            for metric in values[model_id]:
                samples[model_id][metric].append(values[model_id][metric])

        for a, b in PAIRS:
            if a not in values or b not in values:
                continue

            for metric in values[a]:
                differences[f"{a}-{b}"][metric].append(
                    values[a][metric] - values[b][metric]
                )

    def summarize(values):
        array = np.array(values)
        return {
            "mean": float(array.mean()),
            "low": float(np.quantile(array, 0.025)),
            "high": float(np.quantile(array, 0.975)),
            "p_gt_0": float((array > 0).mean()),
        }

    return {
        "resamples": repetitions,
        "models": {
            model_id: {
                metric: summarize(values)
                for metric, values in samples[model_id].items()
            }
            for model_id in model_ids
        },
        "differences": {
            pair: {metric: summarize(values) for metric, values in metrics.items()}
            for pair, metrics in differences.items()
        },
    }


def mcnemar(labels, probabilities, a, b):
    """Exact McNemar test comparing two models on the same samples."""
    prediction_a = (probabilities[a] > 0.5).astype(int)
    prediction_b = (probabilities[b] > 0.5).astype(int)
    a_correct = prediction_a == labels
    b_correct = prediction_b == labels
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    total = a_only + b_only

    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "p_value": float(binomtest(a_only, total, 0.5).pvalue) if total else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--config", default="experiments/experiments.yml")
    parser.add_argument("--results", default="experiments/results")
    parser.add_argument(
        "--audit", default="experiments/quality/near_duplicate_audit.json"
    )

    parser.add_argument(
        "--out", default="experiments/quality/sensitivity_analysis.json"
    )

    parser.add_argument("--cache-dir", default="experiments/quality/predictions")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cpu")

    audit = json.loads(Path(args.audit).read_text())
    test_indices = np.array(audit["arrays"]["test_indices"], dtype=int)
    nearest = np.array(audit["arrays"]["nearest_distances"], dtype=int)
    _, experiments = load_config(args.config)

    probabilities = {}
    labels = None
    rows = {}

    for experiment in experiments:
        experiment_id = experiment["id"]
        checkpoint_path = Path(args.results) / experiment_id / "best_model.pth"

        if not checkpoint_path.exists():
            print(f"Skipping {experiment_id}: {checkpoint_path} not found")
            continue

        cache_path = Path(args.cache_dir) / f"{experiment_id}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            probability = cached["probabilities"]
            current_labels = cached["labels"]
            print(f"{experiment_id}: loaded cache {cache_path}", flush=True)
        else:
            dataset = PCOSDataset(
                args.data_dir,
                transform=get_transforms(train=False),
                denoise=experiment["denoise"],
                enhance=experiment["enhance"],
                segment=experiment["segment"],
                input_mode=experiment["input_mode"],
                seg_pad=experiment.get("seg_pad", 8),
                dedupe=True,
            )

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

            model = DenseNet121Attention(
                pretrained=False,
                dropout=experiment.get("dropout", 0.3),
                attention_type=experiment["attention"],
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            probability, current_labels = predict(
                model, dataset, test_indices, args.batch_size, device
            )

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                probabilities=probability,
                labels=current_labels,
            )

            del dataset, model

        probabilities[experiment_id] = probability
        labels = current_labels

        rows[experiment_id] = {
            "full": metric_row(labels, probability),
            "clean": metric_row(
                labels[nearest > args.threshold], probability[nearest > args.threshold]
            ),
        }

        print(
            f"{experiment_id}: full acc={rows[experiment_id]['full']['accuracy']:.4f} "
            f"clean acc={rows[experiment_id]['clean']['accuracy']:.4f}",
            flush=True,
        )

    if labels is None:
        raise SystemExit("No checkpoints were evaluated.")

    clean = nearest > args.threshold
    comparisons = {}

    for a, b in PAIRS:
        if a not in probabilities or b not in probabilities:
            continue

        comparisons[f"{a}-{b}"] = {
            "full": mcnemar(labels, probabilities, a, b),
            "clean": mcnemar(
                labels[clean],
                {a: probabilities[a][clean], b: probabilities[b][clean]},
                a,
                b,
            ),
        }

    result = {
        "threshold_distance": args.threshold,
        "clean_rule": f"nearest perceptual-hash distance > {args.threshold}",
        "clean_n": int(clean.sum()),
        "clean_healthy": int((labels[clean] == 0).sum()),
        "clean_pcos": int((labels[clean] == 1).sum()),
        "models": rows,
        "mcnemar": comparisons,
        "bootstrap": bootstrap(labels, probabilities, args.bootstrap),
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
