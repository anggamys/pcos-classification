import time
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    """Satu epoch pelatihan dengan mixed precision.

    Args:
        model (torch.nn.Module): Model yang dilatih.
        loader (DataLoader): DataLoader training.
        criterion: Fungsi loss (BCEWithLogitsLoss).
        optimizer: Optimizer PyTorch.
        device (torch.device): CPU atau CUDA.
        scaler (torch.amp.GradScaler): Scaler mixed precision.
        use_amp (bool): Aktifkan autocast float16.

    Returns:
        tuple: (rata-rata loss, akurasi) epoch ini.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    """Evaluasi model pada data validasi tanpa gradient.

    Args:
        model (torch.nn.Module): Model yang dievaluasi.
        loader (DataLoader): DataLoader validasi.
        criterion: Fungsi loss.
        device (torch.device): CPU atau CUDA.
        use_amp (bool): Aktifkan autocast float16.

    Returns:
        tuple: (rata-rata loss, akurasi) validasi.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True).unsqueeze(1)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, config):
    """Loop pelatihan: Adam + ReduceLROnPlateau + early stopping.

    Model terbaik (val loss terkecil) disimpan ke `best_model.pth` di
    direktori save_dir pada config.

    Args:
        model (torch.nn.Module): Model yang dilatih.
        train_loader (DataLoader): DataLoader training.
        val_loader (DataLoader): DataLoader validasi.
        config (dict): Berisi epochs, lr, weight_decay, patience,
            batch_size, dan save_dir.

    Returns:
        dict: History train_loss, train_acc, val_loss, val_acc per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = model.to(device)

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    save_dir = Path(config.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_mem:.1f} GB | AMP: enabled (float16)")
    print(
        f"Epochs: {config['epochs']}, LR: {config['lr']}, Batch: {config['batch_size']}"
    )
    print("-" * 60)

    for epoch in range(config["epochs"]):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1}/{config['epochs']} ({elapsed:.1f}s) - "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                save_dir / "best_model.pth",
            )
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 10):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history
