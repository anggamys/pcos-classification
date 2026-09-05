import torch

from src.dataset import create_dataloaders
from src.models.densenet121 import DenseNet121Attention
from src.train import train
from src.evaluate import (
    evaluate_model, compute_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_training_history, print_report,
)


def main():
    config = {
        "data_dir": "datasets",
        "batch_size": 32,
        "image_size": 224,
        "epochs": 30,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patience": 10,
        "save_dir": "checkpoints",
        "denoise": False,
    }

    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        denoise=config["denoise"],
        num_workers=2,
    )

    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val: {len(val_loader.dataset)}")
    print(f"Test: {len(test_loader.dataset)}")

    print("\nInitializing DenseNet-121 + Attention...")
    model = DenseNet121Attention(num_classes=1, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    print("\nStarting training...")
    history = train(model, train_loader, val_loader, config)

    plot_training_history(history, save_path="checkpoints/training_history.png")

    print("\nLoading best model for evaluation...")
    checkpoint = torch.load("checkpoints/best_model.pth", weights_only=True)
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

    plot_confusion_matrix(results["labels"], results["preds"],
                          save_path="checkpoints/confusion_matrix.png")
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["auc"],
                   save_path="checkpoints/roc_curve.png")


if __name__ == "__main__":
    main()
