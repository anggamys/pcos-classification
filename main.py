import argparse
from collections.abc import Sized
from typing import cast

import torch

from src.dataset import create_dataloaders
from src.evaluate import (
    compute_metrics,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    print_report,
)
from src.models.densenet121 import DenseNet121Attention
from src.train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pengaruh Wavelet Denoising dan Segmentasi Folikel "
        "terhadap Klasifikasi PCOS dengan DenseNet-121 + Attention"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Path to dataset directory (default: datasets)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image resize size (default: 224)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Max training epochs (default: 30)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--denoise", action="store_true", help="Enable wavelet denoising preprocessing"
    )
    parser.add_argument(
        "--enhance",
        type=str,
        default="none",
        choices=["none", "clahe"],
        help="Contrast enhancement (default: none)",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default="none",
        choices=["none", "otsu", "adaptive"],
        help="Follicle segmentation method (default: none)",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="full",
        choices=["full", "roi", "masked"],
        help="Model input: full image, ROI crop, or masked (default: full)",
    )
    parser.add_argument(
        "--seg-pad",
        type=int,
        default=8,
        help="Padding around ROI crop in pixels (default: 8)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader num_workers (default: 2)"
    )

    parser.add_argument(
        "--attention",
        type=str,
        default="self_attention",
        choices=["self_attention", "se_net", "cbam", "transformer"],
        help="Attention mechanism (default: self_attention)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--gradcam", action="store_true", help="Generate Grad-CAM visualization"
    )
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=5,
        help="Number of Grad-CAM samples (default: 5)",
    )

    return parser.parse_args()


def get_model(pretrained=True, dropout=0.3, attention_type="self_attention"):
    return DenseNet121Attention(
        num_classes=1,
        pretrained=pretrained,
        dropout=dropout,
        attention_type=attention_type,
    )


def main():
    args = parse_args()

    config = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "save_dir": args.save_dir,
        "denoise": args.denoise,
        "enhance": args.enhance,
        "segment": args.segment,
        "input_mode": args.input_mode,
        "seg_pad": args.seg_pad,
    }

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("  model: densenet121")
    print(f"  attention: {args.attention}")
    print(f"  dropout: {args.dropout}")
    print()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {total_mem:.1f} GB | AMP: enabled (float16)")
        print()

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
        from src.gradcam import visualize_gradcam

        visualize_gradcam(
            model,
            test_loader,
            device,
            num_samples=args.gradcam_samples,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
