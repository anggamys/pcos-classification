import argparse

import torch

from src.dataset import create_dataloaders
from src.models.densenet121 import DenseNet121Attention
from src.train import train
from src.evaluate import (
    evaluate_model, compute_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_training_history, print_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCOS Classification using DenseNet-121 + Self-Attention"
    )

    parser.add_argument("--data-dir", type=str, default="datasets",
                        help="Path to dataset directory (default: datasets)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image resize size (default: 224)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max training epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--denoise", action="store_true",
                        help="Enable wavelet denoising preprocessing")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Disable ImageNet pretrained weights")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader num_workers (default: 2)")

    return parser.parse_args()


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
    }

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        denoise=config["denoise"],
        num_workers=args.num_workers,
    )

    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val:   {len(val_loader.dataset)}")
    print(f"Test:  {len(test_loader.dataset)}")

    print("\nInitializing DenseNet-121 + Attention...")
    model = DenseNet121Attention(
        num_classes=1, pretrained=not args.no_pretrained
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        results["labels"], results["preds"],
        save_path=f"{args.save_dir}/confusion_matrix.png"
    )
    plot_roc_curve(
        metrics["fpr"], metrics["tpr"], metrics["auc"],
        save_path=f"{args.save_dir}/roc_curve.png"
    )


if __name__ == "__main__":
    main()
