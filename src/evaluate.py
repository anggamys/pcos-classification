import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.amp import autocast


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_probabilities = []
    all_labels = []
    use_amp = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_probabilities.extend(probabilities)
        all_labels.extend(labels.numpy())

    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    all_predictions = (all_probabilities > 0.5).astype(int)

    return {
        "labels": all_labels,
        "probabilities": all_probabilities,
        "predictions": all_predictions,
    }


def compute_metrics(results):
    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    specificity = recall_score(labels, predictions, pos_label=0, zero_division=0)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "auc": roc_auc,
        "average_precision": average_precision,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }


def plot_confusion_matrix(labels, predictions, save_path=None):
    confusion_values = confusion_matrix(labels, predictions)
    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion_values, cmap="Blues")

    for row in range(confusion_values.shape[0]):
        for col in range(confusion_values.shape[1]):
            color = (
                "white"
                if confusion_values[row, col] > confusion_values.max() / 2
                else "black"
            )
            ax.text(
                col,
                row,
                str(confusion_values[row, col]),
                ha="center",
                va="center",
                color=color,
                fontsize=14,
            )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Infected", "Infected"])
    ax.set_yticklabels(["Non-Infected", "Infected"])
    plt.colorbar(im)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(false_positive_rate, true_positive_rate, auc_value, save_path=None):
    _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        "b-",
        linewidth=2,
        label=f"AUC = {auc_value:.4f}",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pr_curve(precision_curve, recall_curve, average_precision, save_path=None):
    _, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        recall_curve,
        precision_curve,
        "b-",
        linewidth=2,
        label=f"AP = {average_precision:.4f}",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history, save_path=None):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_report(results):
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(
        classification_report(
            results["labels"],
            results["predictions"],
            target_names=["Non-Infected", "Infected"],
        )
    )
