"""Run the 6-experiment matrix for the dual-course final project.

Configuration lives in a YAML file (single source of truth):

    python experiments/run_matrix.py --config experiments/experiments.yml
    python experiments/run_matrix.py --config experiments/experiments.yml --only E1,E2

CLI flags (--data-dir, --epochs, --batch-size, --out-dir, --gradcam-samples)
override the YAML defaults when explicitly passed.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

try:
    from experiments.exp_config import (
        load_config,
        project_root,
        resolve_overrides,
        select_experiments,
    )
    from experiments.hypotheses import evaluate_hypotheses
except ImportError:  # dijalankan langsung: python experiments/run_matrix.py
    from exp_config import (
        load_config,
        project_root,
        resolve_overrides,
        select_experiments,
    )
    from hypotheses import evaluate_hypotheses

PROJECT_ROOT = project_root()
sys.path.insert(0, str(PROJECT_ROOT))

MAIN_FLAGS = [
    ("--data-dir", "data_dir"),
    ("--epochs", "epochs"),
    ("--batch-size", "batch_size"),
    ("--image-size", "image_size"),
    ("--lr", "lr"),
    ("--weight-decay", "weight_decay"),
    ("--patience", "patience"),
    ("--dropout", "dropout"),
    ("--seg-pad", "seg_pad"),
    ("--num-workers", "num_workers"),
    ("--attention", "attention"),
    ("--enhance", "enhance"),
    ("--segment", "segment"),
    ("--input-mode", "input_mode"),
    ("--gradcam-samples", "gradcam_samples"),
]

METRIC_RE = {
    "accuracy": re.compile(r"Accuracy:\s+([0-9.]+)"),
    "precision": re.compile(r"Precision:\s+([0-9.]+)"),
    "recall": re.compile(r"Recall:\s+([0-9.]+)"),
    "f1": re.compile(r"F1-Score:\s+([0-9.]+)"),
    "specificity": re.compile(r"Specificity:\s+([0-9.]+)"),
    "auc": re.compile(r"AUC:\s+([0-9.]+)"),
}


def parse_metrics(stdout):
    metrics = {}
    for key, pattern in METRIC_RE.items():
        match = pattern.search(stdout)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


def model_stats(attention, repeats=20):
    """Total params + CPU inference latency (efficiency evidence for H3)."""
    import torch

    from src.models.densenet121 import DenseNet121Attention

    model = DenseNet121Attention(
        num_classes=1, pretrained=False, attention_type=attention
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    sample_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)
        start_time = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample_input)
        elapsed_ms = (time.perf_counter() - start_time) / repeats * 1000.0
    return total_params, round(elapsed_ms, 2)


def build_command(experiment, save_dir):
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py")]
    for flag, key in MAIN_FLAGS:
        value = experiment.get(key)
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    cmd.extend(["--save-dir", str(save_dir), "--gradcam"])
    if experiment["denoise"]:
        cmd.append("--denoise")
    return cmd


def run_experiment(experiment, out_root):
    save_dir = out_root / experiment["id"]
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(experiment, save_dir)
    log_path = save_dir / "stdout.log"
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False
    )
    log_path.write_text(proc.stdout + "\n\n===== STDERR =====\n" + proc.stderr)
    result = {
        "id": experiment["id"],
        "desc": experiment["desc"],
        "denoise": experiment["denoise"],
        "enhance": experiment["enhance"],
        "segment": experiment["segment"],
        "input_mode": experiment["input_mode"],
        "attention": experiment["attention"],
        "returncode": proc.returncode,
    }
    result.update(parse_metrics(proc.stdout))
    total_params, infer_ms = model_stats(experiment["attention"])
    result["total_params"] = total_params
    result["infer_ms_cpu"] = infer_ms
    status = "OK" if proc.returncode == 0 else "FAIL"
    print(f"[{status}] {experiment['id']}: {experiment['desc']}")
    return result


def save_matrix_summary(out_root, rows):
    summaries = {}
    for row in rows:
        summary_path = out_root / row["id"] / "run_summary.json"
        if summary_path.exists():
            summaries[row["id"]] = json.loads(summary_path.read_text())
    matrix = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": rows,
        "hypotheses": evaluate_hypotheses(rows),
        "run_summaries": summaries,
    }
    output_path = out_root / "matrix_summary.json"
    output_path.write_text(json.dumps(matrix, indent=2))
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run PCD+ACM experiment matrix")
    parser.add_argument("--config", default="experiments/experiments.yml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradcam-samples", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--only", default="", help="Comma-separated subset, e.g. E1,E2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the main.py commands without running them",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    _, experiments = load_config(config_path)
    experiments = resolve_overrides(args, experiments)
    selected = select_experiments(experiments, args.only)
    out_root = PROJECT_ROOT / selected[0]["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for experiment in selected:
            print(f"[{experiment['id']}] {experiment['desc']}")
        return

    rows = [run_experiment(experiment, out_root) for experiment in selected]
    csv_path = out_root / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")
    save_matrix_summary(out_root, rows)


if __name__ == "__main__":
    main()
