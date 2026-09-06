import argparse

import torch

from src.dataset import create_dataloaders
from src.models.densenet121 import DenseNet121Attention
from src.models.efficientnet_b3 import EfficientNetB3Attention
from src.train import train
from src.evaluate import (
    evaluate_model, compute_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_training_history, print_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCOS Classification using Deep Learning"
    )

    parser.add_argument("--data-dir", type=str, default="datasets",
                        help="Path to dataset directory (default: datasets)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
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

    parser.add_argument("--model", type=str, default="densenet121",
                        choices=["densenet121", "efficientnet_b3"],
                        help="Model architecture (default: densenet121)")
    parser.add_argument("--attention", type=str, default="self_attention",
                        choices=["self_attention", "se_net", "cbam", "transformer"],
                        help="Attention mechanism (default: self_attention)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run Bayesian optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualization")
    parser.add_argument("--gradcam-samples", type=int, default=5,
                        help="Number of Grad-CAM samples (default: 5)")
    parser.add_argument("--smote", action="store_true",
                        help="Apply SMOTE for class imbalance")

    return parser.parse_args()


def get_model(model_name, pretrained=True, dropout=0.3, attention_type="self_attention"):
    if model_name == "efficientnet_b3":
        return EfficientNetB3Attention(
            num_classes=1, pretrained=pretrained, dropout=dropout
        )
    else:
        return DenseNet121Attention(
            num_classes=1, pretrained=pretrained, dropout=dropout,
            attention_type=attention_type
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
    }

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  model: {args.model}")
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
        num_workers=args.num_workers,
    )

    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val:   {len(val_loader.dataset)}")
    print(f"Test:  {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.optimize:
        print("\nRunning Bayesian Optimization...")
        from src.optimize import run_optuna
        best_params = run_optuna(
            model_class=lambda **kw: get_model(args.model, pretrained=not args.no_pretrained,
                                                attention_type=args.attention, **kw),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config_base=config,
            n_trials=args.n_trials,
        )
        config["lr"] = best_params["lr"]
        config["weight_decay"] = best_params["weight_decay"]
        args.dropout = best_params["dropout"]
        args.batch_size = best_params["batch_size"]
        print(f"\nBest params: {best_params}")

    print(f"\nInitializing {args.model} with {args.attention} attention...")
    model = get_model(args.model, pretrained=not args.no_pretrained,
                      dropout=args.dropout, attention_type=args.attention)
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
        results["labels"], results["preds"],
        save_path=f"{args.save_dir}/confusion_matrix.png"
    )
    plot_roc_curve(
        metrics["fpr"], metrics["tpr"], metrics["auc"],
        save_path=f"{args.save_dir}/roc_curve.png"
    )

    if args.gradcam:
        print("\nGenerating Grad-CAM visualization...")
        from src.gradcam import visualize_gradcam
        visualize_gradcam(model, test_loader, device,
                          num_samples=args.gradcam_samples,
                          save_dir=args.save_dir)


if __name__ == "__main__":
    main()
