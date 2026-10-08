"""Machine-readable run documentation (run_summary.json)."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

SUMMARY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "auc",
    "average_precision",
)

SUMMARY_ARTIFACTS = [
    "best_model.pth",
    "training_history.png",
    "confusion_matrix.png",
    "roc_curve.png",
    "gradcam_visualization.png",
]


def get_git_commit() -> str:
    """Ambil hash commit git pendek untuk keterlacakan run.

    Returns:
        str: Hash commit pendek, atau "unknown" bila git tak tersedia.
    """
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def save_run_summary(
    save_dir: str | Path,
    args: argparse.Namespace,
    config: dict,
    dataset_sizes: dict,
    model_info: dict,
    history: dict,
    test_metrics: dict,
    device: torch.device,
) -> None:
    """Tulis dokumentasi run machine-readable ke run_summary.json.

    Mencatat timestamp, commit git, environment, config, ukuran dataset,
    info model, history per-epoch, epoch terbaik, status early stopping,
    metrik test, dan daftar artefak yang benar-benar tersimpan.

    Args:
        save_dir (str atau Path): Direktori keluaran run.
        args (argparse.Namespace): Argumen CLI (attention, dropout, ...).
        config (dict): Config training dari `build_config`.
        dataset_sizes (dict): Cacah train/val/test.
        model_info (dict): total_params dan trainable_params.
        history (dict): History train/val loss dan akurasi per epoch.
        test_metrics (dict): Metrik hasil `compute_metrics`.
        device (torch.device): Perangkat komputasi yang dipakai.
    """
    checkpoint = torch.load(
        Path(save_dir) / "best_model.pth", map_location="cpu", weights_only=True
    )

    best_epoch = int(checkpoint.get("epoch", -1)) + 1
    stopped_epoch = len(history["train_loss"])
    artifacts = [name for name in SUMMARY_ARTIFACTS if (Path(save_dir) / name).exists()]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "environment": {
            "device": device.type,
            "gpu_name": torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None,
            "torch_version": torch.__version__,
        },
        "config": config,
        "model": {
            "name": "densenet121",
            "attention": args.attention,
            "dropout": args.dropout,
            "pretrained": not args.no_pretrained,
            **model_info,
        },
        "dataset": dataset_sizes,
        "history": {
            key: [float(value) for value in values] for key, values in history.items()
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "early_stopped": stopped_epoch < config["epochs"],
        "test_metrics": {key: float(test_metrics[key]) for key in SUMMARY_METRICS},
        "artifacts": artifacts,
    }

    output_path = Path(save_dir) / "run_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))

    print(f"\nRun summary saved to {output_path}")
