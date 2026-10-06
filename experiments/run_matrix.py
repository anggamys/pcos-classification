"""Run the 6-experiment matrix for the dual-course final project.

Each experiment calls main.py as a subprocess with its own --save-dir so
checkpoints, plots, and logs stay separated. Metrics printed by main.py
(Accuracy / AUC lines) are parsed from stdout into results.csv.

Usage:
    python experiments/run_matrix.py --data-dir datasets --epochs 30
    python experiments/run_matrix.py --data-dir /path/to/data --epochs 30 --only E1,E2
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS = [
    {"id": "E1", "denoise": False, "enhance": "none", "segment": "none",
     "input_mode": "full", "attention": "self_attention",
     "desc": "Baseline tanpa denoise"},
    {"id": "E2", "denoise": True, "enhance": "none", "segment": "none",
     "input_mode": "full", "attention": "self_attention",
     "desc": "BayesShrink wavelet denoising"},
    {"id": "E3", "denoise": True, "enhance": "clahe", "segment": "none",
     "input_mode": "full", "attention": "self_attention",
     "desc": "Denoise + CLAHE enhancement"},
    {"id": "E4", "denoise": True, "enhance": "clahe", "segment": "otsu",
     "input_mode": "roi", "attention": "self_attention",
     "desc": "Preprocessing terbaik + ROI crop Otsu"},
    {"id": "E5", "denoise": True, "enhance": "clahe", "segment": "otsu",
     "input_mode": "masked", "attention": "self_attention",
     "desc": "Preprocessing terbaik + masked input Otsu"},
    {"id": "E6", "denoise": True, "enhance": "clahe", "segment": "otsu",
     "input_mode": "roi", "attention": "cbam",
     "desc": "Konfigurasi terbaik + CBAM attention"},
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
    out = {}
    for key, rx in METRIC_RE.items():
        m = rx.search(stdout)
        out[key] = float(m.group(1)) if m else None
    return out


def model_stats(attention, repeats=20):
    """Total params + CPU inference latency (efficiency evidence for H3)."""
    import torch
    from src.models.densenet121 import DenseNet121Attention
    model = DenseNet121Attention(num_classes=1, pretrained=False,
                                attention_type=attention)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        dt = (time.perf_counter() - t0) / repeats * 1000.0
    return total, round(dt, 2)


def run_experiment(exp, args, out_root):
    save_dir = out_root / exp["id"]
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"),
           "--data-dir", args.data_dir,
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--save-dir", str(save_dir),
           "--attention", exp["attention"],
           "--enhance", exp["enhance"],
           "--segment", exp["segment"],
           "--input-mode", exp["input_mode"],
           "--gradcam", "--gradcam-samples", str(args.gradcam_samples)]
    if exp["denoise"]:
        cmd.append("--denoise")
    log_path = save_dir / "stdout.log"
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n\n===== STDERR =====\n" + proc.stderr)
    result = {"id": exp["id"], "desc": exp["desc"],
              "denoise": exp["denoise"], "enhance": exp["enhance"],
              "segment": exp["segment"], "input_mode": exp["input_mode"],
              "attention": exp["attention"],
              "returncode": proc.returncode}
    result.update(parse_metrics(proc.stdout))
    total_params, infer_ms = model_stats(exp["attention"])
    result["total_params"] = total_params
    result["infer_ms_cpu"] = infer_ms
    status = "OK" if proc.returncode == 0 else "FAIL"
    print(f"[{status}] {exp['id']}: {exp['desc']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run PCD+ACM experiment matrix")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradcam-samples", type=int, default=5)
    parser.add_argument("--out-dir", default="experiments/results")
    parser.add_argument("--only", default="",
                        help="Comma-separated subset, e.g. E1,E2")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = [e for e in EXPERIMENTS if not only or e["id"] in only]
    out_root = PROJECT_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [run_experiment(e, args, out_root) for e in selected]
    csv_path = out_root / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
