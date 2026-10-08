import argparse
from collections.abc import Sized
from typing import cast

import torch

from src.cli import build_config, parse_args
from src.dataset import create_dataloaders
from src.evaluate import (
    compute_metrics,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    print_report,
)
from src.gradcam import visualize_gradcam
from src.models.densenet121 import DenseNet121Attention
from src.run_summary import save_run_summary
from src.train import train


def get_model(
    pretrained: bool = True,
    dropout: float = 0.3,
    attention_type: str = "self_attention",
) -> DenseNet121Attention:
    """Bangun model DenseNet-121 + Attention sesuai judul kerja.

    Args:
        pretrained (bool): Muat bobot ImageNet.
        dropout (float): Laju dropout classifier.
        attention_type (str): Jenis mekanisme attention.

    Returns:
        DenseNet121Attention: Model klasifikasi biner PCOS.
    """
    return DenseNet121Attention(
        pretrained=pretrained,
        dropout=dropout,
        attention_type=attention_type,
    )


def print_config(config: dict, args: argparse.Namespace) -> None:
    """Cetak ringkasan config dan identitas model ke terminal.

    Args:
        config (dict): Config training.
        args (argparse.Namespace): Argumen CLI (model, attention, dropout).
    """
    print("Config:")

    for key, value in config.items():
        print(f"  {key}: {value}")

    print("  model: densenet121")
    print(f"  attention: {args.attention}")
    print(f"  dropout: {args.dropout}")
    print()


def print_gpu_info() -> None:
    """Cetak nama GPU dan VRAM bila CUDA tersedia; diam bila CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"GPU: {gpu_name}")
        print(f"VRAM: {total_mem:.1f} GB | AMP: enabled (float16)")
        print()


def main() -> None:
    """Orkestrasi satu run: data, model, training, evaluasi, dokumentasi."""
    args = parse_args()
    config = build_config(args)

    print_config(config, args)
    print_gpu_info()

    print("Loading dataset...")

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        denoise=config["denoise"],
        enhance=config["enhance"],
        segment=config["segment"],
        input_mode=config["input_mode"],
        seg_pad=config["seg_pad"],
        num_workers=args.num_workers,
    )

    print(f"Train: {len(cast(Sized, train_loader.dataset))}")
    print(f"Val:   {len(cast(Sized, val_loader.dataset))}")
    print(f"Test:  {len(cast(Sized, test_loader.dataset))}")

    dataset_sizes = {
        "train": len(cast(Sized, train_loader.dataset)),
        "val": len(cast(Sized, val_loader.dataset)),
        "test": len(cast(Sized, test_loader.dataset)),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nInitializing densenet121 with {args.attention} attention...")

    model = get_model(
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        attention_type=args.attention,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    model_info = {"total_params": total_params, "trainable_params": trainable_params}
    print("\nStarting training...")

    history = train(model, train_loader, val_loader, config)

    plot_training_history(history, save_path=f"{args.save_dir}/training_history.png")

    print("\nLoading best model for evaluation...")

    checkpoint = torch.load(f"{args.save_dir}/best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(results)

    print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")

    print_report(results)

    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        save_path=f"{args.save_dir}/confusion_matrix.png",
    )

    plot_roc_curve(
        metrics["false_positive_rate"],
        metrics["true_positive_rate"],
        metrics["auc"],
        save_path=f"{args.save_dir}/roc_curve.png",
    )

    if args.gradcam:
        print("\nGenerating Grad-CAM visualization...")

        visualize_gradcam(
            model,
            test_loader,
            device,
            num_samples=args.gradcam_samples,
            save_dir=args.save_dir,
        )

    save_run_summary(
        args.save_dir,
        args,
        config,
        dataset_sizes,
        model_info,
        history,
        metrics,
        device,
    )


if __name__ == "__main__":
    main()
